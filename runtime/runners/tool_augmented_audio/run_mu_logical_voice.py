import bootstrap
import signal
#!/usr/bin/env python3
"""Isolated v14 single-session logical-time entrypoint; no Probe execution."""
from run_tau_mu_voice_realtime import *  # frozen baseline helpers, no main side effects
import os
import sys


def _load_json(path: Path, default=None):
    """Best-effort loader for an immutable prior partial artifact."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default
def run_case(
    args: argparse.Namespace, case_id: str, bridge: Any, values: dict[str, str]
):
    """Execute one causally scheduled multi-user Realtime episode."""
    from experiments.tau_multiuser_voice.attribution import (
        apply_attribution_evidence,
    )
    from experiments.tau_multiuser_voice.authorization import AuthorizationOracle
    from experiments.tau_multiuser_voice.cases import CaseCatalog
    from experiments.tau_multiuser_voice.enrollment import InSessionVoiceEnrollment
    from experiments.tau_multiuser_voice.evaluator import MultiUserEvaluator
    from experiments.tau_multiuser_voice.models import (
        ObservationKind,
        SpeakerAttributionTrace,
        ToolActionTrace,
        TranscriptTurn,
        WorldObservation,
    )
    from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
    from experiments.tau_multiuser_voice.version import (
        BENCHMARK_VERSION,
        EXECUTION_PROTOCOL_VERSION,
        VOICE_PIPELINE_VERSION,
        WEAK_COMPREHENSION_SCHEDULE_VERSION,
    )
    from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT

    case = with_enrollment_transport(
        CaseCatalog.get(case_id), args.enrollment_transport
    )
    openweight = getattr(args, "openweight_endpoint", None) is not None
    from experiments.tau_multiuser_voice.scene_context import verify_transport
    initial_context_receipt = verify_transport(case, openweight=openweight)
    realtime_provider = getattr(args, "realtime_provider", "openai")
    moss = realtime_provider == "moss"
    pcm_transport = openweight or moss
    if openweight and realtime_provider != "openai":
        raise ValueError("openweight endpoint and closed provider selection are mutually exclusive")
    if not openweight:
        from experiments.tau_multiuser_voice.spoken_protocol import with_spoken_session_rules
        case = with_spoken_session_rules(case)
    observation_mode = getattr(args, "planner_observation", "auto")
    if observation_mode == "auto":
        observation_mode = "audible_asr"
    if observation_mode != "audible_asr":
        raise ValueError("v14.2 scored sessions require audible_asr for all providers; "
                         "proportional generated text is diagnostic, not played evidence")
    output_dir = args.output_dir / case_id
    tts = None
    from experiments.tau_multiuser_voice.fact_contract import input_contract, review_for_case
    from experiments.tau_multiuser_voice.task_evidence import build_task_evidence_contract
    case_contract = input_contract(case)
    if initial_context_receipt["transport"] == "post_registration_pre_task_tts":
        case_contract["input_transport"] = "post_registration_pre_task_tts_plus_user_audio"
        case_contract["system_prompt_fact_injection"] = False
    task_scoring_contract = build_task_evidence_contract(case, review_for_case(case))
    if output_dir.exists() and any(output_dir.iterdir()):
        from experiments.mu_bench.logical_time.integrity import OutputArtifactConflictError
        raise OutputArtifactConflictError("case output is nonempty; use a new versioned output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    from experiments.mu_bench.logical_time.integrity import INTEGRITY_VERSION
    write_json(output_dir / "attempt_phase.json", {"phase": "simulation", "integrity_version": INTEGRITY_VERSION})
    if args.frozen_plan_dir or getattr(args, "pre_rendered_user_track", None):
        raise ValueError("single-session logical mode forbids frozen plans/replays")
    # The case contract is unchanged; only execution/simulation is replaced.
    write_json(output_dir / "case_input_contract.json", case_contract)
    write_json(output_dir / "initial_assistant_context.json", initial_context_receipt)
    write_json(output_dir / "reviewed_case.json", case.model_dump(mode="json"))
    voice_manifest = args.voice_manifest
    selected_tts_backend = "manifest_voice_bank" if voice_manifest is not None else None
    if args.voice_bank_dir is not None:
        voice_manifest = args.voice_bank_dir / case_id / "manifest.json"
        selected_tts_backend = "manifest_voice_bank"
    if voice_manifest is not None:
        from experiments.tau_multiuser_voice.voice_bank import ManifestVoiceBankTTS

        tts = ManifestVoiceBankTTS(
            voice_manifest,
            require_complete_quality=not args.allow_incomplete_voice_quality,
        )
        raise ValueError("online wording requires a dynamic TTS backend, not a fixed voice manifest")
    else:
        requested_tts_backend = getattr(args, "tts_backend", "auto")
        has_local_cosyvoice_config = (
            args.cosyvoice_model_dir is not None and args.voice_catalog is not None
        )
        if requested_tts_backend == "cosyvoice_http":
            from experiments.tau_multiuser_voice.tts_http import RemoteCosyVoiceTTS

            selected_endpoint = resolve_remote_tts_endpoint(args)
            tts = RemoteCosyVoiceTTS(
                endpoint_file=selected_endpoint,
                run_namespace=resolve_tts_run_namespace(args),
                case_id=case_id,
                assignment_seed=case.seed,
                timeout_seconds=args.tts_http_timeout_seconds,
            )
            selected_tts_backend = "cosyvoice_http"
        elif requested_tts_backend == "cosyvoice" or (
            requested_tts_backend == "auto" and has_local_cosyvoice_config
        ):
            if not has_local_cosyvoice_config:
                raise ValueError(
                    "CosyVoice TTS requires both --cosyvoice-model-dir and "
                    "--voice-catalog; refusing an implicit API fallback"
                )
            if args.voice_binding_manifest is None:
                raise ValueError(
                    "formal CosyVoice rendering requires --voice-binding-manifest; "
                    "the legacy per-case assignment is smoke-only"
                )
            from experiments.tau_multiuser_voice.tts_backends import CosyVoiceTTS

            tts = CosyVoiceTTS(
                model_dir=args.cosyvoice_model_dir,
                catalog_path=args.voice_catalog,
                binding_manifest_path=args.voice_binding_manifest,
                speaker_feature_cache_path=args.speaker_feature_cache,
                case_id=case_id,
                assignment_seed=case.seed,
                cache_dir=args.tts_cache_dir,
                device=args.cosyvoice_device,
            )
            selected_tts_backend = "cosyvoice"
        elif requested_tts_backend not in {"auto", "gpt_audio"}:
            raise ValueError(f"unsupported TTS backend: {requested_tts_backend}")
        else:
            if requested_tts_backend == "auto" and (
                args.cosyvoice_model_dir is not None or args.voice_catalog is not None
            ):
                raise ValueError(
                    "--cosyvoice-model-dir and --voice-catalog must be supplied "
                    "together"
                )
            tts = GPTAudioTTS(
                bridge,
                values["GPT_URL"],
                values["GPT_KEY"],
                args.tts_model,
                args.tts_cache_dir,
            )
            selected_tts_backend = "gpt_audio_smoke"
    tts.register_profiles(case.users)
    voice_provenance = tts.provenance()
    voice_provenance["requested_backend"] = getattr(args, "tts_backend", "auto")
    voice_provenance["selected_backend"] = selected_tts_backend
    write_json(output_dir / "voice_provenance.json", voice_provenance)

    if not openweight and realtime_provider == "openai":
        bridge.install_realtime_adapter(
            values["REALTIME_GPT_URL"], values["REALTIME_GPT_KEY"], args.realtime_model)
        import tau2.agent.discrete_time_audio_native_agent as agent_module

    from experiments.mu_bench.logical_time.online_users import (
        OnlineUserSession, OnlineMultiUserSimulator, OnlineClosurePlanner,
        LogicalPauseLedger, PROTOCOL_VERSION,
    )
    from experiments.mu_bench.logical_time.pacing import build_case_pacing
    from experiments.mu_bench.logical_time.completion import (
        CompletionConfig, CompletionGate, CompletionSnapshot,
    )
    pacing_plan = build_case_pacing(case)
    write_json(output_dir / "event_pacing_plan.json", pacing_plan.to_dict())
    prior_partial_dir = getattr(args, "prior_partial_dir", None)
    prior_partial_dir = Path(prior_partial_dir) if prior_partial_dir else None
    restored_planner_users: list[str] = []
    sessions = {
        user.user_id: OnlineUserSession(case=case, profile=user,
            backend=BridgeJSONProposalBackend(bridge, values["GPT_URL"], values["GPT_KEY"], args.planner_model),
            timing_alignment_ms=args.logical_tick_ms)
        for user in case.users
    }
    simulator = OnlineMultiUserSimulator(case=case, sessions=sessions, tts=tts,
        tick_ms=args.logical_tick_ms, pacing_plan=pacing_plan)
    tail_orchestrator = OnlineClosurePlanner(BridgeJSONProposalBackend(
        bridge, values["GPT_URL"], values["GPT_KEY"], args.planner_model))
    pause_ledger = LogicalPauseLedger()
    tail_audit = []
    tail_round = 0
    last_assistant_audio_end_ms = 0
    enrollment = InSessionVoiceEnrollment(
        case=case,
        tts=tts,
        tick_ms=simulator.tick_ms,
        separate_session_protocol=not openweight,
    )
    trace = simulator.trace
    write_json(output_dir / "registration_audio_segments.json", enrollment.registration_audio_segments)
    if not openweight:
        from experiments.tau_multiuser_voice.spoken_protocol import receipt as spoken_protocol_receipt
        spoken_rules = spoken_protocol_receipt(enrollment)
        write_json(output_dir / "session_protocol_audio.json", spoken_rules)
    source_phase_end_ms = enrollment.enrollment_complete_ms
    if initial_context_receipt['transport'] == 'post_registration_pre_task_tts':
        from experiments.tau_multiuser_voice.scene_audio import (
            source_text, render_source, install_source, record_source_tick)
        initial_source_text = source_text(case)
        initial_source_pcm, initial_source_voice = render_source(case, tts, initial_source_text)
        source_phase_end_ms = install_source(enrollment, initial_source_pcm,
            initial_source_text, output_dir, initial_context_receipt, initial_source_voice)
    trace.voice_enrollment_mode = (
        "in_session" if case.voice_enrollment.mode == "in_session" else "not_run"
    )
    trace.task_start_ms = enrollment.task_start_ms
    trace.voice_enrollments = list(enrollment.traces)
    trace.assistant_activation = enrollment.activation_trace
    trace.execution_protocol = PROTOCOL_VERSION
    trace.frozen_event_plan_sha256 = None
    trace.probe_count = 0
    if prior_partial_dir is not None:
        # A partial trace cannot restore the SUT's hidden KV/cache state. It
        # can restore accepted private user-planner turns and provide
        # immutable provenance for this fresh larger-budget session.
        source_files = {}
        for rel in (
            "trace.partial.json",
            "conversation_completion.json",
            "audio/user_mixed_native.wav",
            "audio/both.wav",
        ):
            source = prior_partial_dir / rel
            if source.exists():
                stat = source.stat()
                source_files[rel] = {
                    "path": str(source),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
        write_json(output_dir / "recovery_source.json", {
            "mode": "fresh_model_session_with_prior_user_planner_context",
            "source_dir": str(prior_partial_dir),
            "source_files": source_files,
            "hidden_model_state_restored": False,
            "user_planner_context_restored": False,
            "policy": "partial_audio_and_planner_context_are_provenance; no model KV continuation",
        })
    if prior_partial_dir is not None:
        for uid, user_session in sessions.items():
            source = prior_partial_dir / "planner_sessions" / f"{uid}.json"
            payload = _load_json(source, {})
            messages = payload.get("messages") if isinstance(payload, dict) else None
            if not isinstance(messages, list) or not messages:
                continue
            clean_messages = [
                item for item in messages
                if isinstance(item, dict)
                and item.get("role") in {"system", "user", "assistant"}
                and isinstance(item.get("content"), str)
            ]
            if not clean_messages:
                continue
            # Keep the current system contract if the prior artifact is from
            # an older runtime; otherwise preserve the accepted history.
            if clean_messages[0].get("role") == "system":
                user_session.messages = clean_messages
            else:
                user_session.messages = [user_session.messages[0], *clean_messages]
            user_session._proposal_index = sum(
                1 for item in clean_messages if item.get("role") == "assistant"
            )
            restored_planner_users.append(uid)
        recovery_source = _load_json(output_dir / "recovery_source.json", {})
        if isinstance(recovery_source, dict):
            recovery_source["user_planner_context_restored"] = bool(restored_planner_users)
            recovery_source["restored_planner_users"] = restored_planner_users
            write_json(output_dir / "recovery_source.json", recovery_source)
    environment = create_virtual_environment(case.environment)
    tool_names = required_tool_names(case)
    if openweight and tool_names:
        raise ValueError("openweight track supports only the existing no-tool cases")
    from resource_id_protocol import BusinessResourceIDs
    resource_ids = BusinessResourceIDs(environment) if tool_names else None
    if resource_ids is not None:
        write_json(output_dir / "resource_id_protocol.json", resource_ids.describe())
    from experiments.tau_multiuser_voice.business_participant_identifiers import configure_schemas, metadata, public_result
    if tool_names and not openweight:
        configure_schemas(environment, tool_names)
        visible_identity_tools = [{"name": n, "parameters": environment.definitions[n].parameters}
                                  for n in sorted(tool_names)]
        write_json(output_dir / "business_participant_id_protocol.json", metadata(case, visible_identity_tools))
    tools = create_tools(environment, tool_names) if not openweight else []
    authorization_oracle = AuthorizationOracle(case)
    # Closed-source Realtime receives the stable benchmark protocol and tool
    # contract. Identity and initial permissions are still spoken by each
    # participant in that participant's task voice.
    system_prompt = build_system_prompt(case, environment, enrollment, tool_names) if not openweight else ""
    if resource_ids is not None:
        system_prompt += resource_ids.prompt_suffix()
    system_prompt_path = output_dir / "system_prompt.txt"
    system_prompt_path.write_text(system_prompt, encoding="utf-8")
    write_json(
        output_dir / "system_prompt_policy.json",
        {
            "policy": "native_official_defaults_only" if openweight else "closed_source_attribution_alias_revision_v4_20261114",
            "custom_system_prompt_sent": not openweight,
            "session_rules_transport": "spoken_tts_registration",
            "transcription_keyterms": False,
            "prompt_path": str(system_prompt_path.resolve()),
            "prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "prompt_excludes_future_event_oracle": True,
            "prompt_excludes_internal_user_ids": True,
            "prompt_excludes_identity_and_acl_text": True,
            "prompt_excludes_non_tool_behavior_rules": True,
            "identity_delivery": "spoken_tts_self_introduction",
        },
    )

    if moss:
        from adapters.providers.moss.tools import MossToolAdapter
        adapter = MossToolAdapter(args.moss_endpoint, tick_duration_ms=simulator.tick_ms,
            seed=case.seed, input_sample_rate=tts.sample_rate,
            max_media_ms=enrollment.task_start_ms + int(args.logical_max_seconds * 1000) + simulator.tick_ms)
        adapter.tool_effects = {name: environment.definitions[name].effect for name in tool_names}
    elif openweight:
        from experiments.mu_bench.logical_time.openweight_bridge import OpenweightLogicalAdapter
        adapter = OpenweightLogicalAdapter(args.openweight_endpoint,
            tick_duration_ms=simulator.tick_ms, seed=case.seed,
            input_sample_rate=tts.sample_rate,
            max_media_ms=enrollment.task_start_ms + int(args.logical_max_seconds * 1000) + simulator.tick_ms)
    elif realtime_provider == "seeduplex":
        from experiments.mu_bench.logical_time.seeduplex_adapter import make_seeduplex_logical_adapter
        adapter = make_seeduplex_logical_adapter(
            key=values["REALTIME_SEEDUPLEX_KEY"], model=args.realtime_model, bridge=bridge,
            realtime_url=values["REALTIME_SEEDUPLEX_URL"], tick_duration_ms=simulator.tick_ms,
            voice=getattr(args, "seeduplex_voice", "zh_female_vv_jupiter_bigtts"),
            proxy=os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY"))
    elif realtime_provider == "gemini":
        from experiments.mu_bench.logical_time.gemini_adapter import make_gemini_logical_adapter
        adapter = make_gemini_logical_adapter(
            key=values["REALTIME_GEMINI_KEY"], model=args.realtime_model, bridge=bridge,
            tick_duration_ms=simulator.tick_ms,
            proxy=os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY"))
    elif realtime_provider == "xai":
        from experiments.mu_bench.logical_time.grok_adapter import make_grok_logical_adapter
        adapter = make_grok_logical_adapter(
            realtime_url=values["REALTIME_GROK_URL"], realtime_key=values["REALTIME_GROK_KEY"],
            model=args.realtime_model, tick_duration_ms=simulator.tick_ms,
            audio_format=TELEPHONY_AUDIO_FORMAT, bridge=bridge)

    else:
        from experiments.mu_bench.logical_time.openai_adapter import make_logical_adapter
        adapter = make_logical_adapter(agent_module, tick_duration_ms=simulator.tick_ms,
                                      audio_format=TELEPHONY_AUDIO_FORMAT)
    adapter.connect(system_prompt, tools, modality="audio")
    # Supervisor kills use SIGTERM on the process group. Release the
    # max_sessions=1 worker slot before the interpreter exits.
    def _release_openweight_on_signal(signum, _frame):
        try:
            if hasattr(adapter, "force_abort"):
                adapter.force_abort(reason=f"signal_{signum}")
            elif getattr(adapter, "is_connected", False):
                adapter.disconnect()
        finally:
            raise SystemExit(128 + int(signum))
    try:
        signal.signal(signal.SIGTERM, _release_openweight_on_signal)
    except Exception:
        pass

    final_session_id = getattr(adapter.provider, "session_id", None)
    write_json(output_dir / "session_topology.json", {
        "protocol": PROTOCOL_VERSION, "model_session_count": 1, "probe_count": 0,
        "provider_session_id": final_session_id, "scored_session_id": final_session_id,
        "fresh_final_session": False, "replay_used": False,
    })
    def configure_auto_response(enabled):
        if not openweight:
            adapter.configure_auto_response(enabled=enabled)
    try:
        configure_auto_response(False)
    except Exception:
        write_json(output_dir / "adapter_audit.json", adapter.audit_records)
        if adapter.is_connected:
            adapter.disconnect()
        raise

    converter = ULAWConverter()
    user_ulaw_ticks: list[bytes] = []
    agent_ulaw_ticks: list[bytes] = []
    user_recording_rate = tts.sample_rate if pcm_transport else 8000
    assistant_recording_rate = adapter.output_sample_rate if pcm_transport else 8000
    def recorded_pcm(chunks):
        raw = b"".join(chunks)
        return raw if pcm_transport else audioop.ulaw2lin(raw, 2)
    def save_audio_tracks():
        user_pcm, assistant_pcm = recorded_pcm(user_ulaw_ticks), recorded_pcm(agent_ulaw_ticks)
        write_mono_wav(output_dir / "audio" / ("user_mixed_native.wav" if pcm_transport else "user_mixed_8k.wav"),
                       user_pcm, user_recording_rate)
        write_mono_wav(output_dir / "audio" / ("assistant_native.wav" if pcm_transport else "assistant_8k.wav"),
                       assistant_pcm, assistant_recording_rate)
        if assistant_recording_rate != user_recording_rate:
            assistant_pcm, _ = audioop.ratecv(assistant_pcm, 2, 1, assistant_recording_rate, user_recording_rate, None)
        write_stereo_wav(output_dir / "audio" / "both.wav", user_pcm, assistant_pcm, user_recording_rate)
    tick_records: list[dict] = []
    assistant_intervals_session: list[dict] = []
    assistant_turns_session: list[dict] = []
    current_interval: dict | None = None
    current_text = ""
    assistant_speaking = False
    next_kind = ObservationKind.CONVERSATION_START
    pending_tool_name: str | None = None
    pending_tool_result: str | None = None
    pending_protocol_observations: list[dict] = []
    current_oracle_actor_id: str | None = None
    current_sut_actor_id: str | None = None
    previous_task_active_speakers: set[str] = set()
    attributed_oracle_proposal_ids: set[str] = set()
    public_transcript: list[TranscriptTurn] = []
    tool_log: list[dict] = []
    prior_completed: set[str] = set()
    enrollment_complete_ms = enrollment.enrollment_complete_ms
    task_start_ms = enrollment.task_start_ms
    audible_observer = None
    if observation_mode == "audible_asr":
        from experiments.mu_bench.logical_time.openweight_bridge import PlayedPrefixObserver
        def played_snapshot():
            intervals = list(assistant_intervals_session)
            if current_interval is not None:
                intervals.append({"start_ms": current_interval["start_ms"],
                                  "end_ms": len(agent_ulaw_ticks) * simulator.tick_ms})
            return (recorded_pcm(agent_ulaw_ticks), assistant_recording_rate, task_start_ms, intervals)
        audible_observer = PlayedPrefixObserver(artifact_dir=output_dir / "played_prefix_observer",
                                               bridge=bridge, values=values, snapshot=played_snapshot)
        for user_session in sessions.values():
            user_session.observation_enricher = audible_observer.observe
        def public_played_evidence(observation):
            # Called after the ASR enricher. Every item has a task-clock
            # interval so a late joiner cannot inherit earlier conversation.
            fragments = (audible_observer.last_result or {}).get("assistant_fragments", [])
            assistant = [{"speaker": "assistant", "text": f["text"],
                          "start_ms": math.floor(f["context_start_ms"] - task_start_ms),
                          "end_ms": math.ceil(f["context_end_ms"] - task_start_ms)}
                         for f in fragments]
            users = [turn.model_dump(mode="json") for turn in observation.public_transcript]
            return sorted([*users, *assistant], key=lambda item: (item["start_ms"], item["end_ms"]))
        for user_session in sessions.values():
            user_session.public_evidence_provider = public_played_evidence
    task_end_ms = int(args.logical_max_seconds * 1000)
    session_end_ms = task_start_ms + task_end_ms
    total_ticks = math.ceil(session_end_ms / simulator.tick_ms)
    # Quiet auto-end after assistant speech: OFF for openweight (0), ON for
    # closed-source Grok/OpenAI (10s). Shared CompletionConfig default is 0.
    bilateral_silence_limit_ms = int(getattr(args, "bilateral_silence_seconds", 0) * 1000)
    completion_protocol = ("Silence600-v1" if bilateral_silence_limit_ms else "pre-Silence600")
    write_json(output_dir / "completion_protocol_marker.json", {
        "label": completion_protocol,
        "bilateral_silence_limit_ms": bilateral_silence_limit_ms,
        "clock": "logical_time",
        "counting_phase": "formal_task",
        "eligibility": "both audio tracks idle; output drained; no pending response, tools or protocol",
        "end_state": "completed_bilateral_silence_limit",
        "task_success_determined_by": "existing J-Full and separate-authorization evaluator"})
    completion_config = CompletionConfig(
        bilateral_silence_limit_ms=bilateral_silence_limit_ms,
        quiet_guard_ms=max(2000, case.iterative_replay.probe_completion_guard_ms),
        # Preserve the advertised --final-drain-seconds minimum. It was
        # previously parsed but ignored by this single-session runner.
        minimum_response_opportunity_ms=max(5000, int(args.final_drain_seconds * 1000)),
        post_decision_settle_ms=2000,
        post_assistant_user_silence_ms=(10_000 if not pcm_transport else 0),
        max_logical_ms=total_ticks * simulator.tick_ms)
    completion_gate = CompletionGate(completion_config)
    completion_snapshot = None
    completion_assessment = None
    completion_evidence_key = None
    completion_evidence_revision = 0
    closure_end_denials = 0
    closure_forced_after_denials = False
    termination = {"state": "running", "normal_conversation_end": False}
    # Reaching the explicit logical-duration cap is a bounded, scoreable
    # conversation end. It is not a transport, simulator, or judge failure.
    bounded_logical_completion = False
    bounded_bilateral_silence_completion = False
    # proposal_contract_exhausted forced-eval (Jay/Critic 2026-10-12): whitelist
    # only this SimulatorInvalid reason; score + enter denominator like logical cap.
    bounded_proposal_contract_completion = False
    write_json(output_dir / "session_spec.json", {
        "schema_version": PROTOCOL_VERSION, "case_id": case.case_id,
        "execution_mode": "single_session_logical", "measurement_track": "logical_time",
        "tick_ms": simulator.tick_ms, "task_start_ms": task_start_ms,
        "model_name": args.realtime_model, "system_prompt_path": str(system_prompt_path),
        "enrollment_transport": case.voice_enrollment.transport,
        "identity_control_events": [], "user_audio_source": "causal_online_tts",
        "future_audio_visible": False, "planner_location": "networked_development_machine",
        "model_generation_freezes_during_planning": openweight,
        "planner_observation": observation_mode,
        "prior_partial_recovery": prior_partial_dir is not None,
        "prior_partial_dir": str(prior_partial_dir) if prior_partial_dir else None,
        "restored_planner_users": restored_planner_users,
        "transcript_alignment": "audio_bounded_asr_not_word_aligned" if observation_mode == "audible_asr" else "upstream_proportional_not_word_aligned",
        "execution_implementation_version": INTEGRITY_VERSION,
        "event_pacing_plan": "event_pacing_plan.json",
        "completion_policy": dict(vars(completion_config)),
    })
    task_auto_response_enabled = False
    enrollment_response_created_count = 0
    delivered_control_events: list[dict[str, Any]] = []
    session_wall_started = time.monotonic()
    final_completed = False
    cleanup_failure = None
    tick = -1
    try:
        for tick in range(total_ticks):
            tick_wall_started = time.monotonic()
            session_time_ms = tick * simulator.tick_ms
            in_enrollment = session_time_ms < enrollment_complete_ms
            in_initial_source = enrollment_complete_ms <= session_time_ms < source_phase_end_ms
            in_activation = source_phase_end_ms <= session_time_ms < task_start_ms
            in_task = session_time_ms >= task_start_ms
            task_time_ms = session_time_ms - task_start_ms
            if in_enrollment:
                phase = "voice_enrollment"
                observation_kind = "voice_enrollment"
                audio_window = enrollment.mix_window(
                    session_time_ms,
                    session_time_ms + simulator.tick_ms,
                )
            elif in_initial_source:
                phase = "initial_assistant_context"
                observation_kind = "initial_assistant_context"
                audio_window = enrollment.mix_window(
                    session_time_ms, session_time_ms + simulator.tick_ms)
            elif in_activation:
                phase = "assistant_activation"
                observation_kind = "assistant_activation"
                if (
                    case.assistant_activation.allow_brief_acknowledgement
                    and not task_auto_response_enabled
                ):
                    configure_auto_response(True)
                    task_auto_response_enabled = True
                audio_window = enrollment.mix_window(
                    session_time_ms,
                    session_time_ms + simulator.tick_ms,
                )
            else:
                phase = "task"
                if not task_auto_response_enabled:
                    configure_auto_response(True)
                    task_auto_response_enabled = True
                for spoken in simulator.trace.spoken_events:
                    if (
                        spoken.end_ms <= task_time_ms
                        and spoken.event_id not in prior_completed
                    ):
                        public_transcript.append(
                            TranscriptTurn(
                                speaker=spoken.speaker_id,
                                text=spoken.utterance,
                                start_ms=spoken.start_ms,
                                end_ms=spoken.end_ms,
                            )
                        )
                        if spoken.event_id:
                            prior_completed.add(spoken.event_id)

                task_turn_texts = [
                    turn["text"]
                    for turn in assistant_turns_session
                    if turn["start_ms"] >= task_start_ms
                ]
                observation = WorldObservation(
                    sim_time_ms=task_time_ms,
                    kind=next_kind,
                    assistant_speaking=assistant_speaking,
                    assistant_committed_prefix=" ".join(task_turn_texts)
                    + (" " if task_turn_texts and current_text else "")
                    + current_text,
                    public_transcript=public_transcript,
                    completed_event_ids=simulator.event_graph.completed_event_ids,
                    tool_name=pending_tool_name,
                    tool_result=pending_tool_result,
                    active_thread_ids={
                        event.thread_id for event in simulator.trace.spoken_events
                    },
                )
                # No append/receive/playback/tick advance occurs inside planning.
                adapter.pause_logical_clock("user_planner_and_tts", session_time_ms)
                try:
                    with pause_ledger.planning(session_time_ms):
                        # Observe all tool transitions at the next logical
                        # boundary without advancing/rendering audio twice.
                        # Tool invocation, authorization and results themselves
                        # remain exactly the baseline dispatch block below.
                        for transition in pending_protocol_observations:
                            simulator.event_graph.ready_events(observation.model_copy(update=transition))
                        pending_protocol_observations.clear()
                        # Native models without text streams observe only an
                        # ASR of already-delivered audio. Prefix-trigger checks
                        # must use the same causal observation as user planning.
                        if audible_observer is not None:
                            pending = next((e for e in simulator.event_graph._events
                                if e.event_id not in simulator.event_graph.completed_event_ids), None)
                            if pending is not None and pending.trigger.kind.value == "assistant_prefix":
                                observation = audible_observer.observe(observation)
                        simulation_tick = simulator.update(observation)
                finally:
                    adapter.resume_logical_clock("user_planner_and_tts", session_time_ms)
                pending_tool_name = None
                pending_tool_result = None
                observation_kind = observation.kind.value
                audio_window = simulation_tick.audio
            final_user_pcm = audio_window.mixed_pcm_s16le
            user_ulaw = final_user_pcm if pcm_transport else converter.convert(final_user_pcm, adapter.bytes_per_tick)
            active_task_speakers = set(audio_window.active_speakers) if in_task else set()
            if active_task_speakers - previous_task_active_speakers:
                # A new utterance invalidates the requester context from the
                # preceding turn.  A business call now needs a fresh explicit
                # attribution matched to a completed speech event.
                current_sut_actor_id = None
                current_oracle_actor_id = None
            previous_task_active_speakers = active_task_speakers
            result = adapter.run_tick(user_ulaw, tick_number=tick)
            if not openweight and session_time_ms < enrollment_complete_ms:
                spoken_rules["delivered_until_ms"] = min(session_time_ms + simulator.tick_ms, spoken_rules["session_end_ms"])
                spoken_rules["delivery_status"] = ("audio_submitted" if spoken_rules["delivered_until_ms"] >= spoken_rules["session_end_ms"] else "submitting")
                write_json(output_dir / "session_protocol_audio.json", spoken_rules)
            if in_initial_source:
                record_source_tick(output_dir, initial_context_receipt,
                    session_time_ms + simulator.tick_ms)
            tick_processing_ms = (time.monotonic() - tick_wall_started) * 1000
            event_types = [getattr(event, "type", "unknown") for event in result.events]
            if in_enrollment:
                enrollment_response_created_count += event_types.count(
                    "response.created"
                )
            agent_ulaw = result.get_played_agent_audio()
            user_ulaw_ticks.append(user_ulaw)
            agent_ulaw_ticks.append(agent_ulaw)

            fragment = result.proportional_transcript or ""
            current_text += fragment
            agent_pcm = agent_ulaw if pcm_transport else audioop.ulaw2lin(agent_ulaw, 2)
            now_speaking = audioop.rms(agent_pcm, 2) > 80
            if now_speaking:
                last_assistant_audio_end_ms = session_time_ms + simulator.tick_ms
            if now_speaking and not assistant_speaking:
                current_interval = {"start_ms": session_time_ms, "end_ms": None}
                if in_task:
                    next_kind = ObservationKind.ASSISTANT_SPEECH_START
            elif not now_speaking and assistant_speaking:
                if current_interval is not None:
                    current_interval["end_ms"] = session_time_ms
                    assistant_intervals_session.append(current_interval)
                    assistant_turns_session.append(
                        {
                            "start_ms": current_interval["start_ms"],
                            "end_ms": session_time_ms,
                            "text": current_text.strip(),
                        }
                    )
                current_interval = None
                current_text = ""
                if in_task:
                    next_kind = ObservationKind.ASSISTANT_SPEECH_STOP
            elif in_task:
                next_kind = ObservationKind.CLOCK
            assistant_speaking = now_speaking

            calls_this_tick = []
            for call in result.tool_calls:
                arguments = dict(call.arguments)
                if call.name == "report_speaker_attribution":
                    error = None
                    current_sut_actor_id = None
                    current_oracle_actor_id = None
                    reported_speaker_reference = str(
                        arguments.get("speaker_id", "")
                    )
                    reported_addressee_reference = str(
                        arguments.get("addressee", "")
                    )
                    predicted_speaker_id = normalize_user_reference(
                        case, reported_speaker_reference
                    )
                    predicted_addressee = normalize_addressee_reference(
                        case, reported_addressee_reference
                    )
                    predicted_thread_id = str(arguments.get("thread_id", ""))
                    try:
                        confidence = float(arguments.get("confidence", 0.0))
                        if not in_task:
                            raise ValueError(
                                "attribution tool called before formal task start"
                            )
                        if predicted_speaker_id not in {*sessions, "unknown"}:
                            raise ValueError("speaker_id is not enrolled or unknown")
                        if predicted_addressee not in {
                            *sessions,
                            "assistant",
                            "group",
                            "unknown",
                        }:
                            raise ValueError("addressee is not a known participant")
                        attribution = SpeakerAttributionTrace(
                            timestamp_ms=session_time_ms,
                            task_timestamp_ms=task_time_ms,
                            predicted_speaker_id=predicted_speaker_id,
                            predicted_addressee=predicted_addressee,
                            reported_speaker_reference=(
                                reported_speaker_reference
                            ),
                            reported_addressee_reference=(
                                reported_addressee_reference
                            ),
                            predicted_thread_id=predicted_thread_id,
                            confidence=confidence,
                        )
                        trace.speaker_attributions.append(attribution)
                        oracle_event = match_oracle_event_online(
                            trace,
                            task_time_ms=task_time_ms,
                            used_proposal_ids=attributed_oracle_proposal_ids,
                        )
                        # No correctness or hidden-source feedback to the SUT.
                        # Event matching is private routing evidence only.
                        current_sut_actor_id = predicted_speaker_id
                        if oracle_event is not None:
                            current_oracle_actor_id = oracle_event.speaker_id
                            attribution.matched_event_id = oracle_event.event_id
                            attribution.matched_proposal_id = oracle_event.proposal_id
                            attributed_oracle_proposal_ids.add(oracle_event.proposal_id)
                        tool_result = {"recorded": True}
                    except (TypeError, ValueError, ValidationError) as exc:
                        error = str(exc)
                        tool_result = {"error": error}
                        if not in_task:
                            trace.simulator_violations.append(
                                "attribution_tool_before_task"
                            )
                    # Official xAI continuation: function_call_output, then
                    # response.create. The provider adds a one-turn instruction
                    # to speak and not repeat this attribution call.
                    adapter.send_tool_result(
                        call.id,
                        json.dumps(tool_result, ensure_ascii=False),
                        request_response=True,
                        is_error=bool(error),
                    )
                    calls_this_tick.append(
                        {
                            "kind": "speaker_attribution",
                            "session_timestamp_ms": session_time_ms,
                            "task_timestamp_ms": task_time_ms,
                            "arguments": arguments,
                            "error": error,
                        }
                    )
                    continue

                tool_definition = environment.definitions.get(call.name)
                requester_is_runner_bound = bool(
                    tool_definition
                    and tool_definition.requester_source
                    == "runner_bound_speaker_context"
                )
                from experiments.tau_multiuser_voice.caller_protocol import resolve_caller, REQUESTER_SOURCE
                explicit_caller = bool(tool_definition and tool_definition.requester_source == REQUESTER_SOURCE)
                caller_id = None
                caller_error = None
                if explicit_caller:
                    declared_actor, caller_id, arguments, caller_error = resolve_caller(case, arguments)
                supplied_actor_argument = arguments.pop("actor_id", None)
                if explicit_caller:
                    actor_id = declared_actor
                elif requester_is_runner_bound:
                    actor_id = current_sut_actor_id or ""
                else:
                    actor_id = normalize_user_reference(
                        case, str(supplied_actor_argument or "")
                    )
                raw_business_arguments = dict(call.arguments)
                arguments = normalize_tool_arguments(call.name, arguments, case=case)
                resource_resolution = resource_parse_error = None
                if resource_ids is not None:
                    arguments, resource_resolution, resource_parse_error = resource_ids.resolve(call.name, arguments)
                    resource_ids.record(output_dir, call.id, raw_business_arguments, arguments, resource_resolution, task_time_ms)
                resource_id = infer_resource_id(case, call.name, actor_id, arguments)
                oracle_actor_id = infer_oracle_actor_id(
                    case,
                    trace,
                    tool_name=call.name,
                    resource_id=resource_id,
                    task_time_ms=task_time_ms,
                    attributed_actor_id=current_oracle_actor_id,
                )
                if explicit_caller:
                    oracle_actor_id = None  # no invented ground-truth requester
                oracle_authorized = False
                authorization_evidence: list[str] = []
                executed = False
                error = None
                tool_result: Any
                if not in_task:
                    error = "domain tool called before formal task start"
                    tool_result = {"error": error}
                    trace.simulator_violations.append("domain_tool_before_task")
                elif explicit_caller and caller_error:
                    error = caller_error
                    tool_result = {"error": error}
                elif requester_is_runner_bound and supplied_actor_argument is not None:
                    error = (
                        "actor_id is runner-bound context and must not appear in "
                        "business arguments"
                    )
                    tool_result = {"error": error}
                    authorization_evidence = ["identity_argument_rejected"]
                elif actor_id not in sessions:
                    error = "unknown or missing actor_id"
                    tool_result = {"error": error}
                elif not explicit_caller and oracle_actor_id is None:
                    error = "cannot bind tool call to an actual requesting speaker"
                    tool_result = {"error": error}
                elif not explicit_caller and actor_id != oracle_actor_id:
                    error = (
                        "speaker/actor mismatch: claimed "
                        f"{actor_id}, actual {oracle_actor_id}"
                    )
                    tool_result = {"error": error}
                    authorization_evidence = ["speaker_actor_mismatch"]
                elif resource_parse_error is not None:
                    error = resource_parse_error["code"] + ": " + resource_parse_error["message"]
                    tool_result = {"error": error, "resource_id_error": resource_parse_error}
                    authorization_evidence = ["resource_id_resolution_failed_before_authorization"]
                else:
                    decision = authorization_oracle.decide(
                        actor_id=actor_id if explicit_caller else oracle_actor_id,
                        action=call.name,
                        resource_id=resource_id,
                        runtime=sessions[actor_id if explicit_caller else oracle_actor_id].runtime,
                    )
                    oracle_authorized = decision.allowed
                    authorization_evidence = [
                        decision.reason,
                        *decision.matching_grant_ids,
                    ]
                    if not decision.allowed:
                        error = f"authorization denied: {decision.reason}"
                        tool_result = {
                            "error": error,
                            "actor_id": actor_id,
                            "resource_id": resource_id,
                        }
                    else:
                        try:
                            tool_result = environment.invoke(
                                call.name, actor_id, **arguments
                            )
                            executed = True
                        except (TypeError, ValueError) as exc:
                            error = str(exc)
                            tool_result = {"error": error}
                model_visible_tool_result = public_result(case, call.name, tool_result)
                adapter.send_tool_result(
                    call.id,
                    json.dumps(model_visible_tool_result, ensure_ascii=False),
                    request_response=True,
                    is_error=bool(error),
                )
                record = {
                    "timestamp_ms": task_time_ms,
                    "result_available_ms": task_time_ms + simulator.tick_ms if executed else None,
                    "result_availability_basis": "next_tick_submission_bound" if executed else None,
                    "session_timestamp_ms": session_time_ms,
                    "tool_name": call.name,
                    "raw_arguments": raw_business_arguments,
                    "model_visible_result": model_visible_tool_result,
                    "participant_id_protocol": "mu-business-public-participant-api-v2-20261115",
                    "resource_id_resolution": resource_resolution,
                    "actor_id": actor_id,
                    "oracle_actor_id": oracle_actor_id,
                    "caller_id": caller_id,
                    "authorization_basis": REQUESTER_SOURCE if explicit_caller else "oracle_speaker_context",
                    "resource_id": resource_id,
                    "arguments": arguments,
                    "executed": executed,
                    "result": tool_result,
                }
                calls_this_tick.append(record)
                tool_log.append(record)
                simulator.trace.tool_actions.append(
                    ToolActionTrace(
                        tool_name=call.name,
                        resource_id=resource_id,
                        timestamp_ms=task_time_ms,
                        result_available_ms=task_time_ms + simulator.tick_ms if executed else None,
                        result_availability_basis="next_tick_submission_bound" if executed else None,
                        sut_attributed_actor_id=actor_id or None,
                        oracle_actor_id=oracle_actor_id,
                        caller_id=caller_id,
                        authorization_basis=REQUESTER_SOURCE if explicit_caller else "oracle_speaker_context",
                        oracle_authorized=oracle_authorized,
                        executed=executed,
                        arguments=arguments,
                        authorization_evidence=authorization_evidence,
                    )
                )
                pending_tool_name = call.name
                pending_tool_result = json.dumps(tool_result, ensure_ascii=False)
            if pending_tool_name is not None and in_task:
                next_kind = ObservationKind.TOOL_RESULT
            if in_task:
                for record in calls_this_tick:
                    # Attribution is bookkeeping, not a domain-tool transition.
                    # Its original record intentionally has no tool_name/result.
                    if record.get("kind") == "speaker_attribution":
                        continue
                    for kind in (ObservationKind.TOOL_CALL, ObservationKind.TOOL_RESULT):
                        pending_protocol_observations.append({"kind": kind,
                            "tool_name": record["tool_name"],
                            "tool_result": json.dumps(record.get("result"), ensure_ascii=False)})

            tick_records.append(
                {
                    "tick": tick,
                    "phase": phase,
                    "session_time_ms": session_time_ms,
                    "task_time_ms": None if not in_task else task_time_ms,
                    "wall_elapsed_ms": (time.monotonic() - session_wall_started) * 1000,
                    "tick_processing_ms": tick_processing_ms,
                    "observation_kind": observation_kind,
                    "active_user_speakers": audio_window.active_speakers,
                    "assistant_speaking": now_speaking,
                    "user_audio_source": "causal_online_tts",
                    "logical_time_paused_during_planning": True,
                    "assistant_transcript_delta": fragment,
                    "realtime_event_types": event_types,
                    "vad_events": result.vad_events,
                    "was_truncated": result.was_truncated,
                    "truncated_audio_bytes": result.truncated_audio_bytes,
                    "tool_calls": calls_this_tick,
                }
            )
            if in_task:
                # Count only voiced, delivered task media. Planner/TTS wall
                # time and silence never earn spoken recovery opportunity.
                simulator.event_graph.pacing.observe_media(
                    task_time_ms + simulator.tick_ms, now_speaking,
                    response_in_flight=not adapter.is_output_drained(),
                    played_duration_ms=simulator.tick_ms if now_speaking else 0)
                played_end_ms = session_time_ms + simulator.tick_ms
                latest_user_end_task_ms = max((e.end_ms for e in trace.spoken_events), default=0)
                latest_user_end_session_ms = task_start_ms + latest_user_end_task_ms
                delivered_text = " ".join(turn["text"] for turn in assistant_turns_session
                    if turn["start_ms"] >= task_start_ms) + current_text
                evidence_key = (len(trace.spoken_events), latest_user_end_session_ms,
                    last_assistant_audio_end_ms, delivered_text, len(tool_log))
                if evidence_key != completion_evidence_key:
                    completion_evidence_revision += 1
                    completion_evidence_key = evidence_key
                completion_snapshot = CompletionSnapshot(
                    logical_ms=played_end_ms,
                    all_events_completed=simulator.all_events_completed,
                    latest_user_audio_end_ms=latest_user_end_session_ms,
                    user_audio_active=latest_user_end_session_ms > played_end_ms,
                    assistant_speaking=now_speaking,
                    last_assistant_audio_end_ms=last_assistant_audio_end_ms,
                    output_drained=adapter.is_output_drained(),
                    pending_protocol=bool(pending_protocol_observations),
                    pending_tools=pending_tool_name is not None,
                    evidence_revision=completion_evidence_revision)
                completion_assessment = completion_gate.evaluate(completion_snapshot)
                if completion_assessment.terminal_incomplete:
                    if completion_snapshot.logical_ms >= completion_config.max_logical_ms:
                        bounded_logical_completion = True
                        termination = {"state": "completed_logical_limit",
                            "normal_conversation_end": True,
                            "completion_mode": "logical_duration_cap",
                            "logical_duration_cap_ms": completion_config.max_logical_ms,
                            "assessment": completion_assessment.as_dict()}
                        final_completed = True
                        break
                    termination = {"state": "incomplete_logical_budget",
                        "normal_conversation_end": False,
                        "assessment": completion_assessment.as_dict()}
                    raise RuntimeError("logical session budget exhausted before confirmed bilateral completion")
                if completion_assessment.can_request_closure:
                    # Rebuild the current, fully delivered public user prefix;
                    # the final user can finish in this tick, after observation.
                    closure_public = [TranscriptTurn(speaker=e.speaker_id, text=e.utterance,
                        start_ms=e.start_ms, end_ms=e.end_ms) for e in trace.spoken_events
                        if e.end_ms <= task_time_ms + simulator.tick_ms]
                    closure_observation = observation.model_copy(update={
                        "sim_time_ms": task_time_ms + simulator.tick_ms,
                        "assistant_speaking": now_speaking,
                        "assistant_committed_prefix": delivered_text,
                        "public_transcript": closure_public,
                        "completed_event_ids": simulator.event_graph.completed_event_ids,
                    })
                    adapter.pause_logical_clock("closure_planner", played_end_ms)
                    try:
                        with pause_ledger.planning(played_end_ms, "closure_planner"):
                            if audible_observer is not None:
                                closure_observation = audible_observer.observe(closure_observation)
                            asr_result = getattr(audible_observer, "last_result", None)
                            assistant_timeline_source = ((asr_result or {}).get("assistant_fragments", [])
                                if audible_observer is not None else assistant_turns_session)
                            played_timeline = [{"speaker": e.speaker_id, "start_ms": e.start_ms,
                                "end_ms": e.end_ms, "text": e.utterance, "source": "played_user_tts"}
                                for e in trace.spoken_events
                                if e.end_ms <= task_time_ms + simulator.tick_ms]
                            for fragment in assistant_timeline_source:
                                context_start = fragment.get("context_start_ms", fragment.get("start_ms", 0))
                                context_end = fragment.get("context_end_ms", fragment.get("end_ms", played_end_ms + 1))
                                if (fragment.get("text") and context_start >= task_start_ms
                                        and context_end <= played_end_ms):
                                    played_timeline.append({"speaker": "assistant",
                                        "start_ms": math.floor(context_start - task_start_ms),
                                        "end_ms": math.ceil(context_end - task_start_ms),
                                        "text": fragment["text"],
                                        "source": "played_audio_asr_interval" if audible_observer is not None
                                            else "played_audio_proportional_transcript_interval"})
                            played_timeline.sort(key=lambda row: (row["start_ms"], row["end_ms"]))
                            voiced_after_user_ms = sum(max(0, interval["end_ms"] -
                                max(interval["start_ms"], latest_user_end_session_ms))
                                for interval in assistant_intervals_session)
                            closure_context = {"timestamp_basis": "task_logical_media_ms",
                                "played_end_ms": task_time_ms + simulator.tick_ms,
                                "last_user_audio_end_ms": latest_user_end_task_ms,
                                "last_assistant_audio_end_ms": max(0, last_assistant_audio_end_ms - task_start_ms),
                                "assistant_voiced_after_last_user_ms": voiced_after_user_ms,
                                "assistant_output_drained": completion_snapshot.output_drained,
                                "assistant_speaking": now_speaking,
                                "minimum_response_opportunity_ms": completion_config.minimum_response_opportunity_ms,
                                "asr_word_alignment_available": False}
                            write_json(output_dir / "closure_observations" / f"{len(tail_audit):03d}.json",
                                {"played_timeline": played_timeline, "completion_context": closure_context,
                                 "evidence_revision": completion_evidence_revision})
                            from experiments.tau_multiuser_voice.iterative_replay import TailDecision
                            try:
                                decision = tail_orchestrator.decide_online(case=case,
                                    observation=closure_observation, spoken_events=trace.spoken_events,
                                    round_index=tail_round, played_timeline=played_timeline,
                                    completion_context=closure_context)
                            except Exception as exc:
                                is_closure_contract = (type(exc).__name__ == "SimulatorInvalid"
                                    and getattr(exc, "reason_code", "") == "closure_contract_exhausted")
                                if not is_closure_contract:
                                    raise
                                closure_end_denials += 1
                                audit_payload = exc.as_dict() if hasattr(exc, "as_dict") else {"error": str(exc)}
                                if closure_end_denials >= 4:
                                    closure_forced_after_denials = True
                                    decision = TailDecision(action="end", actor_id=None,
                                        reason=("forced benchmark closure after three rejected closure decisions; "
                                                "final scoring will use the preserved played evidence"))
                                    write_json(output_dir / "forced_closure_after_denials.json", {
                                        "closure_end_denials": closure_end_denials,
                                        "forced": True,
                                        "last_error": audit_payload,
                                        "policy": "three denied closure decisions are allowed; fourth forces Final evaluation"})
                                else:
                                    decision = TailDecision(action="respond",
                                        actor_id=case.users[0].user_id,
                                        intent=("briefly ask the assistant to complete any remaining unanswered "
                                                "request; do not add new facts or new tasks"),
                                        addressee="assistant", thread_id="main",
                                        speech_act="repair_request", urgency=3,
                                        overlap_allowed=False, max_delay_ms=3000,
                                        requested_delay_ms=0,
                                        reason=("closure denied by semantic user contract; retrying without "
                                                "consuming a successful closure"))
                                    write_json(output_dir / f"closure_denial_retry_{closure_end_denials}.json", {
                                        "closure_end_denials": closure_end_denials,
                                        "forced": False,
                                        "last_error": audit_payload,
                                        "policy": "denied closure is converted into a bounded repair request"})
                    finally:
                        adapter.resume_logical_clock("closure_planner", played_end_ms)
                    tail_audit.append({"logical_ms": task_time_ms + simulator.tick_ms,
                        "evidence_revision": completion_evidence_revision,
                        **decision.model_dump(mode="json")})
                    completion_gate.record_decision(completion_snapshot, action=decision.action,
                        outstanding_question=False if decision.action == "end" else True,
                        reason=(decision.reason + ("; forced_after_three_closure_denials" if closure_forced_after_denials and decision.action == "end" else "")))
                    if decision.action == "respond":
                        if tail_round >= case.iterative_replay.tail_probe_max_rounds:
                            termination = {"state": "incomplete_closure_budget",
                                "normal_conversation_end": False,
                                "reason": "outstanding interaction remains after bounded tail replies"}
                            final_completed = True
                            break
                        simulator.append_tail(tail_orchestrator.to_event(decision, tail_round,
                            task_time_ms + simulator.tick_ms))
                        tail_round += 1
                    # Even a semantic end is provisional. Keep advancing the
                    # SAME session so delayed speech invalidates stale closure.
                if completion_assessment.completed:
                    final_completed = True
                    reasons = completion_assessment.reasons
                    if "bilateral_silence_limit_reached" in reasons:
                        bounded_bilateral_silence_completion = True
                        termination = {
                            "state": "completed_bilateral_silence_limit",
                            "normal_conversation_end": True,
                            "completion_mode": "bilateral_silence_cap",
                            "bilateral_silence_limit_ms": bilateral_silence_limit_ms,
                            "assessment": completion_assessment.as_dict()}
                    elif "incomplete_events_quiet_after_assistant_speech" in reasons:
                        termination = {
                            "state": "completed_post_assistant_user_silence",
                            "normal_conversation_end": True,
                            "completion_mode": "post_assistant_user_silence",
                            "post_assistant_user_silence_ms": (
                                completion_config.post_assistant_user_silence_ms),
                            "assessment": completion_assessment.as_dict()}
                    else:
                        termination = {"state": "completed", "normal_conversation_end": True,
                            "assessment": completion_assessment.as_dict()}
                    break
        if not final_completed:
            reached_cap = len(agent_ulaw_ticks) * simulator.tick_ms >= completion_config.max_logical_ms
            if reached_cap:
                bounded_logical_completion = True
                termination = {"state": "completed_logical_limit",
                    "normal_conversation_end": True,
                    "completion_mode": "logical_duration_cap",
                    "logical_duration_cap_ms": completion_config.max_logical_ms,
                    "assessment": (completion_assessment.as_dict()
                        if completion_assessment is not None else None)}
                final_completed = True
            else:
                raise RuntimeError("logical session budget exhausted before event/closure completion")
    except BaseException as exc:
        # Whitelist ONLY proposal_contract_exhausted (Critic MUST-EDIT). Catch inside
        # the sim loop; do NOT reraise to outer failure_record. Align to closure
        # forced-eval: preserve played evidence and continue to Final scoring.
        is_proposal_contract = (
            type(exc).__name__ == "SimulatorInvalid"
            and getattr(exc, "reason_code", "") == "proposal_contract_exhausted"
        )
        if is_proposal_contract:
            bounded_proposal_contract_completion = True
            audit_payload = exc.as_dict() if hasattr(exc, "as_dict") else {"error": str(exc)}
            termination = {
                "state": "completed_proposal_contract_exhausted",
                "normal_conversation_end": True,
                "completion_mode": "proposal_contract_exhausted_forced_eval",
                "reason_code": "proposal_contract_exhausted",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "event_id": getattr(exc, "event_id", None),
            }
            write_json(output_dir / "proposal_contract_exhausted_forced_eval.json", {
                "forced": True,
                "reason_code": "proposal_contract_exhausted",
                "last_error": audit_payload,
                "policy": (
                    "Default SimulatorInvalid remains excluded from model score; "
                    "proposal_contract_exhausted is the documented exception. "
                    "Final scoring uses preserved played evidence (closure-forced pattern)."
                ),
            })
            final_completed = True
            # do NOT raise — fall through finally → Judge/score
        else:
            if termination["state"] == "running":
                termination = {"state": "aborted_error", "normal_conversation_end": False,
                    "error_type": type(exc).__name__, "error": str(exc)}
                from experiments.mu_bench.logical_time.pacing import CausalWindowUnrealized
                if isinstance(exc, CausalWindowUnrealized):
                    termination.update(state="challenge_unrealized", challenge=exc.as_dict())
                    write_json(output_dir / "challenge_unrealized.json", exc.as_dict())
            raise
    finally:
        # A planner/transport failure may happen before the attempted tick was
        # delivered. Never add that unplayed tick to the saved WAV time axis.
        session_end_ms = len(agent_ulaw_ticks) * simulator.tick_ms
        task_end_ms = max(0, session_end_ms - task_start_ms)
        # Save played evidence even if planning, transport, closure or Judge fails.
        write_json(output_dir / "trace.partial.json", trace.model_dump(mode="json"))
        write_json(output_dir / "ticks.json", tick_records)
        write_json(output_dir / "logical_pauses.json", pause_ledger.records)
        write_json(output_dir / "tail_orchestrator_audit.json", tail_audit)
        write_json(output_dir / "closure_schema_audit.json", getattr(tail_orchestrator, "audit_log", []))
        write_json(output_dir / "event_pacing_audit.json", simulator.event_graph.pacing_audit)
        write_json(output_dir / "conversation_completion.json", {
            **termination, "session_end_ms": session_end_ms,
            "attempted_tick_index": tick, "recorded_audio_tick_count": len(agent_ulaw_ticks),
            "policy": dict(vars(completion_config)),
            "last_snapshot": dict(vars(completion_snapshot)) if completion_snapshot else None,
            "audit": completion_gate.audit_records,
            "score_produced_only_after_normal_conversation_end": True,
            "logical_duration_cap_is_scoreable": bounded_logical_completion,
            "completion_protocol": completion_protocol,
            "bilateral_silence_cap_is_scoreable": bounded_bilateral_silence_completion,
            "proposal_contract_exhausted_is_scoreable": bounded_proposal_contract_completion})
        write_json(output_dir / "adapter_audit.json", adapter.audit_records)
        for uid, user_session in sessions.items():
            write_json(output_dir / "planner_sessions" / f"{uid}.json", {
                "user_id": uid, "isolated_from_other_users": True, "messages": user_session.messages})
            write_json(output_dir / "simulator_contract_audits" / f"{uid}.json",
                       getattr(user_session, "contract_audit", []))
        save_audio_tracks()
        if adapter.is_connected:
            try:
                adapter.disconnect()
            except Exception as close_error:
                # Preserve the original simulation failure, if any. A close
                # failure must not mask it or prevent a completed Final bundle
                # from being sealed for later evaluation-only recovery.
                cleanup_failure = {"error_type": type(close_error).__name__,
                                   "error": "adapter disconnect failed",
                                   "normal_conversation_end": final_completed}
                write_json(output_dir / "session_cleanup_error.json", cleanup_failure)
        release_case = getattr(tts, "release_case", None) if tts is not None else None
        if callable(release_case):
            try:
                write_json(output_dir / "tts_release.json", release_case())
            except Exception as release_error:
                write_json(output_dir / "tts_cleanup_error.json", {
                    "error_type": type(release_error).__name__,
                    "error": str(release_error),
                    "normal_conversation_end": final_completed,
                })

    if current_interval is not None:
        current_interval["end_ms"] = session_end_ms
        assistant_intervals_session.append(current_interval)
        assistant_turns_session.append(
            {
                "start_ms": current_interval["start_ms"],
                "end_ms": session_end_ms,
                "text": current_text.strip(),
            }
        )

    enrollment_turns = [
        turn
        for turn in assistant_turns_session
        if turn["start_ms"] < enrollment_complete_ms
    ]
    activation_turns = [
        turn
        for turn in assistant_turns_session
        if turn["end_ms"] > source_phase_end_ms and turn["start_ms"] < task_start_ms
    ]
    source_turns = [turn for turn in assistant_turns_session
        if turn['end_ms'] > enrollment_complete_ms and turn['start_ms'] < source_phase_end_ms]
    write_json(output_dir / 'initial_context_assistant_turns.json', source_turns)
    assistant_turns = [
        {
            **turn,
            "start_ms": max(0, turn["start_ms"] - task_start_ms),
            "end_ms": min(task_end_ms, turn["end_ms"] - task_start_ms),
        }
        for turn in assistant_turns_session
        if turn["end_ms"] > task_start_ms
    ]
    assistant_intervals = [
        {
            "start_ms": max(0, interval["start_ms"] - task_start_ms),
            "end_ms": min(task_end_ms, interval["end_ms"] - task_start_ms),
        }
        for interval in assistant_intervals_session
        if interval["end_ms"] > task_start_ms
    ]
    # Persist a complete, unscored Final for every catalog case before any ASR
    # or Judge call. Evaluation repair must never require replaying the SUT.
    save_audio_tracks()
    write_json(output_dir / "attempt_phase.json", {"phase": "evaluation", "integrity_version": INTEGRITY_VERSION})
    write_json(output_dir / "final_trace.before_judge.json", trace.model_dump(mode="json"))
    write_json(output_dir / "tool_log.before_judge.json", tool_log)
    write_json(output_dir / "assistant_turns.proportional_diagnostic.json", assistant_turns)
    write_json(output_dir / "voice_enrollment.before_judge.json", {
        "task_start_ms": task_start_ms, "enrollment_complete_ms": enrollment_complete_ms,
        "transport": case.voice_enrollment.transport, "oracle_audio_bindings": enrollment.prompt_bindings()})
    if cleanup_failure is not None:
        raise RuntimeError("session cleanup failed; complete unscored Final evidence preserved")
    if audible_observer is not None:
        asr_result = audible_observer.observe()
        write_json(output_dir / "audible_asr_result.json", asr_result)
        from experiments.mu_bench.logical_time.integrity import played_assistant_evidence
        assistant_turns = played_assistant_evidence(asr_result,
            task_start_ms=task_start_ms, session_end_ms=session_end_ms)
    trace.enrollment_assistant_response_count = len(enrollment_turns)
    trace.activation_assistant_response_count = len(activation_turns)
    trace.enrollment_response_created_count = enrollment_response_created_count
    trace.session_wall_duration_ms = (time.monotonic() - session_wall_started) * 1000
    trace.session_media_duration_ms = float(session_end_ms)
    trace.max_tick_processing_ms = max(
        (record["tick_processing_ms"] for record in tick_records),
        default=0.0,
    )
    trace.stalled_tick_count = sum(
        record["tick_processing_ms"] > simulator.tick_ms * 2 for record in tick_records
    )

    enrollment.write_session_audio(
        simulator.queues,
        output_dir / "audio" / "oracle",
        task_end_ms,
    )
    save_audio_tracks()

    trace.final_environment_state = environment.snapshot()
    assess_interaction_realization(case, trace, assistant_intervals)
    measure_timings(case, trace, assistant_intervals)
    apply_attribution_evidence(trace, assistant_intervals, tool_log)
    write_json(output_dir / "final_trace.before_judge.json", trace.model_dump(mode="json"))
    write_json(output_dir / "assistant_turns.before_judge.json", assistant_turns)
    write_json(output_dir / "tool_log.before_judge.json", tool_log)
    write_json(output_dir / "reviewed_audio_integrity.json", simulator.audio_integrity())
    from experiments.tau_multiuser_voice.judge_evidence import JudgeEvidenceInvalid
    last_judge_error = None
    for judge_attempt in range(1, 4):
        judge_audit_dir = output_dir if judge_attempt == 1 else output_dir / f"semantic_judge_retry_{judge_attempt}"
        if judge_attempt > 1:
            from mubench.judge_retry_io import prepare_retry
            prepare_retry(output_dir, judge_audit_dir)
        try:
            capabilities, goals, privacy_leaks = run_semantic_judge(
                bridge, values, case, trace, assistant_turns, tool_log, audit_dir=judge_audit_dir
            )
            if judge_attempt > 1:
                from mubench.judge_retry_io import promote_retry
                promote_retry(output_dir, judge_audit_dir, judge_attempt)
                write_json(output_dir / "semantic_judge_retry_success.json", {"judge_attempt": judge_attempt})
            break
        except JudgeEvidenceInvalid as exc:
            last_judge_error = exc
            write_json(output_dir / f"semantic_judge_retry_failure_{judge_attempt}.json", {
                "judge_attempt": judge_attempt,
                "error": str(exc),
                "audit": getattr(exc, "audit", None),
            })
            if any(str(i.get('path', '')).startswith('source_guard') for i in getattr(exc, 'issues', [])):
                raise  # V4 guard is bounded; preserve evidence for review.
            if judge_attempt == 3:
                raise
    else:
        raise last_judge_error
    trace.capability_results = capabilities
    trace.goal_results = goals
    trace.privacy_leaks = privacy_leaks
    measure_floor_split_timings(case, trace, assistant_turns)
    if not openweight:
        enforce_hybrid_rule_assertions(case, trace)
    report = MultiUserEvaluator().evaluate(case, trace)
    score_payload = report.model_dump(mode="json")
    from experiments.tau_multiuser_voice.caller_protocol import apply_score_policy
    score_payload = apply_score_policy(case, score_payload)
    if openweight:
        # Keep the previous openweight eligibility policy byte-for-byte. Its
        # historical "realtime" label means causal input, not wall-clock speed.
        from experiments.mu_bench.logical_time.openweight_scoring import compatibility_score
        from experiments.mu_bench.runtime.contracts import CausalInputIsolation, ExecutionMode
        native_compatibility = compatibility_score(report, case_id=case.case_id,
            model=args.realtime_model, tool_required=False, leaderboard_track="core_no_tool",
            execution_mode=ExecutionMode.CAUSAL_STREAMING,
            causal_input_isolation=CausalInputIsolation.ENFORCED,
            plan_source="causal_online_user_sessions")
        write_json(output_dir / "score_full_contract.json", score_payload)
        write_json(output_dir / "score_compatibility_original_policy.json", native_compatibility)
        score_payload.update(native_compatibility)
        score_payload["causal_interaction_eligible"] = native_compatibility["realtime_eligible"]
    score_payload["task_evidence_contract_version"] = task_scoring_contract["format_version"]
    score_payload["judge_goal_scope_policy"] = "mu-case-design-goal-scope-v1-20261114"
    score_payload["judge_full_review_policy"] = "mu-full-semantic-review-v1-20261114"
    score_payload.update({
        "termination_state": termination.get("state"),
        "normal_conversation_end": bool(termination.get("normal_conversation_end")),
        "completion_protocol": completion_protocol,
        "bilateral_silence_limit_ms": bilateral_silence_limit_ms,
        "bilateral_silence_cap_is_scoreable": bounded_bilateral_silence_completion,
        "logical_duration_cap_applied": bounded_logical_completion,
        "logical_duration_cap_is_scoreable": bounded_logical_completion,
        "logical_duration_cap_ms": (completion_config.max_logical_ms
            if bounded_logical_completion else None),
        "proposal_contract_exhausted_forced_eval": bounded_proposal_contract_completion,
        "proposal_contract_exhausted_is_scoreable": bounded_proposal_contract_completion,
    })
    # B1 (Critic PASS): label cap-unreached events; do NOT change coverage formula
    # or historical interaction/floor numbers. Parallel fields only.
    try:
        _req = {e.event_id for e in case.events if getattr(e, "required", False)}
        _obs = {e.event_id for e in trace.spoken_events if e.event_id is not None}
        _unreached = sorted(_req - _obs)
    except Exception:
        _unreached = []
    if bounded_logical_completion and _unreached:
        score_payload["events_unreached_due_to_logical_cap"] = _unreached
        score_payload["required_events_unreached_due_to_cap"] = _unreached
        score_payload["incompleteness_stratum"] = "logical_duration_cap"
    elif bounded_bilateral_silence_completion:
        score_payload["events_unreached_due_to_bilateral_silence_cap"] = _unreached
        score_payload["incompleteness_stratum"] = "bilateral_silence_cap"
    elif bounded_proposal_contract_completion:
        score_payload["events_unreached_due_to_proposal_contract"] = _unreached
        score_payload["incompleteness_stratum"] = "proposal_contract_exhausted"
    # Keep missing_required_events diagnostics untouched (no coverage formula change).
    score_payload["task_contract_origin"] = task_scoring_contract.get("contract_origin", "reviewed")
    score_payload["task_review_sha256"] = case_contract.get("review_sha256")
    score_payload["case_input_sha256"] = hashlib.sha256(json.dumps(
        case.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    score_payload["privacy_evidence_contract_version"] = "mu-bench-privacy-evidence-v1"
    score_payload.update({
        "integrity_version": INTEGRITY_VERSION,
        "evaluation_valid": True,
        "valid_model_result": bool(report.run_valid) or bounded_logical_completion or bounded_proposal_contract_completion or bounded_bilateral_silence_completion,
        "eligible_for_model_score_denominator": bool(report.run_valid) or bounded_logical_completion or bounded_proposal_contract_completion or bounded_bilateral_silence_completion,
        "assistant_evidence_source": "played_assistant_waveform_asr",
        "asr_accuracy_certified": False,
        "measurement_track": "logical_time", "execution_mode": "single_session_logical",
        "execution_protocol_version": PROTOCOL_VERSION,
        "realtime_eligible": False, "wall_clock_eligible": False,
        "plan_source": "causal_online_user_sessions", "probe_execution": "none",
        "task_contract_supported": True, "attribution_supported": not openweight,
        "interaction_score_supported_contract": True, "interaction_success_supported_contract": True,
        "strict_joint_success_supported_contract": True,
        "pause_equivalence_claim": "same_native_context_no_steps" if openweight else "conditional_not_wall_clock_equivalence",
        "planner_observation": observation_mode,
    })
    audit = adapter.audit_records
    history_risks = [r for r in audit if
        r.get("kind") == "provider_event_before_truncate_ack"
        or (r.get("kind") == "native_interruption" and r.get("native_history_alignment_unverified"))
        or (r.get("kind") == "truncate_ack" and not r.get("matched", False))]
    score_payload["logical_history_risk"] = bool(history_risks)
    score_payload["logical_history_risk_event_count"] = len(history_risks)
    score_payload["logical_tool_race_event_count"] = sum(
        r.get("kind") == "tool_call_interruption_race" for r in audit)
    # These are protocol qualification diagnostics, not extra terms secretly
    # inserted into the existing Task/Interaction/run_valid formulas.
    if openweight:
        for key in ("speaker_attribution_accuracy", "addressee_accuracy", "speaker_success", "addressee_success", "attribution_success"):
            score_payload[key] = None
    write_json(output_dir / "trace.json", trace.model_dump(mode="json"))
    write_json(output_dir / "score.json", score_payload)
    write_json(output_dir / "ticks.json", tick_records)
    write_json(output_dir / "clock_diagnostics.json", {
        "tick_ms": simulator.tick_ms, "session_clock_duration_ms": session_end_ms,
        "session_wall_duration_ms": trace.session_wall_duration_ms,
        "planning_wall_ms": sum(r["wall_elapsed_ms"] for r in pause_ledger.records),
        "interaction_timing_clock": "logical_media_clock", "wall_timing_eligible": False,
        "cloud_generation_frozen": False, "max_tick_processing_ms": trace.max_tick_processing_ms,
        "stalled_tick_count": trace.stalled_tick_count,
    })
    write_json(output_dir / "assistant_turns.json", assistant_turns)
    write_json(output_dir / "assistant_turns_session.json", assistant_turns_session)
    write_json(output_dir / "enrollment_assistant_turns.json", enrollment_turns)
    write_json(output_dir / "assistant_activation_turns.json", activation_turns)
    write_json(output_dir / "session_control_events.json", delivered_control_events)
    write_json(
        output_dir / "voice_enrollment.json",
        {
            "mode": trace.voice_enrollment_mode,
            "transport": case.voice_enrollment.transport,
            "assistant_name": case.assistant_activation.assistant_name,
            "session_preamble": case.voice_enrollment.session_preamble_template,
            "formal_start_instruction": case.voice_enrollment.self_identification_activation_template,
            "uses_dynamic_control_items": (
                case.voice_enrollment.uses_dynamic_control_items
            ),
            "enrollment_complete_ms": enrollment_complete_ms,
            "task_start_ms": task_start_ms,
            "oracle_audio_bindings": enrollment.prompt_bindings(),
            "assistant_activation": (
                trace.assistant_activation.model_dump(mode="json")
                if trace.assistant_activation is not None
                else None
            ),
            "control_events_file": "session_control_events.json",
            "assistant_response_count": trace.enrollment_assistant_response_count,
            "activation_assistant_response_count": (
                trace.activation_assistant_response_count
            ),
            "response_created_event_count": (trace.enrollment_response_created_count),
        },
    )
    write_json(output_dir / "tool_log.json", tool_log)
    write_json(output_dir / "session_result.json", {
        "schema_version": PROTOCOL_VERSION, "case_id": case.case_id,
        "provider_session_id": final_session_id, "state": "succeeded",
        "model_session_count": 1, "probe_count": 0,
        "measurement_track": "logical_time", "media_duration_ms": session_end_ms,
        "task_start_ms": task_start_ms, "model_name": args.realtime_model,
        "realtime_provider": "openweight" if openweight else realtime_provider,
        "assistant_turns": assistant_turns_session,
        "score_json": str(output_dir / "score.json"),
    })

    # --- E6 formal post-score (MUST-2 migrate → floor A → rice); never fail case ---
    if os.environ.get("MUBENCH_FORMAL_POST_SCORE") == "1":
        try:
            from pathlib import Path as _Path
            _runtime = _Path(
                os.environ.get(
                    "MUBENCH_RUNTIME_ROOT",
                    str(_Path(__file__).resolve().parent),
                )
            )
            if str(_runtime) not in sys.path:
                sys.path.insert(0, str(_runtime))
            from formal_post_score_rice import (
                resolve_formal_case_dir,
                run_after_score_seal,
            )
            _formal_root = _Path(os.environ["MUBENCH_FORMAL_ROOT"])
            _model = os.environ["MUBENCH_MODEL"]
            _formal_case = resolve_formal_case_dir(
                output_dir,
                formal_root=_formal_root,
                model=_model,
                case_id=case_id,
            )
            _res = run_after_score_seal(
                output_dir,
                formal_case_dir=_formal_case,
                runtime_root=_runtime,
                model=_model,
                bridge=_Path(os.environ["MUBENCH_BRIDGE"])
                if os.environ.get("MUBENCH_BRIDGE")
                else None,
                env_file=_Path(os.environ["MUBENCH_ENV"])
                if os.environ.get("MUBENCH_ENV")
                else None,
                migrate_mode=os.environ.get("MUBENCH_MIGRATE_MODE", "hardlink"),
            )
            try:
                write_json(output_dir / "_formal_post_score_receipt.json", _res)
            except Exception:
                pass
        except Exception as _e6_exc:
            try:
                write_json(
                    output_dir / "_formal_post_score_receipt.json",
                    {
                        "status": "hook_error",
                        "error": str(_e6_exc),
                        "error_type": type(_e6_exc).__name__,
                    },
                )
            except Exception:
                pass
    else:
        # B2 attempt-local RICE (A_OUT / HOLD): score already written; never migrate.
        # Prefer MUBENCH_ATTEMPT_LOCAL_RICE=1; if unset and SEAL_CLEANUP=1 → ON.
        # Order: attempt-local rice → cleanup (require_rice stays False).
        _want = os.environ.get("MUBENCH_ATTEMPT_LOCAL_RICE")
        if _want is None:
            _want = "1" if os.environ.get("MUBENCH_ATTEMPT_SEAL_CLEANUP") == "1" else "0"
        if _want == "1":
            try:
                from pathlib import Path as _Path
                _runtime = _Path(
                    os.environ.get(
                        "MUBENCH_RUNTIME_ROOT",
                        str(_Path(__file__).resolve().parent),
                    )
                )
                if str(_runtime) not in sys.path:
                    sys.path.insert(0, str(_runtime))
                from formal_post_score_rice import run_attempt_local_rice
                _b2 = run_attempt_local_rice(
                    output_dir,
                    runtime_root=_runtime,
                    bridge=_Path(os.environ["MUBENCH_BRIDGE"])
                    if os.environ.get("MUBENCH_BRIDGE")
                    else _Path(),
                    env_file=_Path(os.environ["MUBENCH_ENV"])
                    if os.environ.get("MUBENCH_ENV")
                    else _Path(),
                    model=os.environ.get("MUBENCH_MODEL") or "",
                )
                try:
                    write_json(output_dir / "_attempt_local_rice_receipt.json", _b2)
                except Exception:
                    pass
            except Exception as _b2_exc:
                try:
                    write_json(
                        output_dir / "_attempt_local_rice_receipt.json",
                        {
                            "status": "hook_error",
                            "error": str(_b2_exc),
                            "error_type": type(_b2_exc).__name__,
                            "migrated": False,
                        },
                    )
                except Exception:
                    pass
        if os.environ.get("MUBENCH_ATTEMPT_SEAL_CLEANUP") == "1":
            # A-class / no-flat: after score (+optional B2 rice), drop played+mono on attempt.
            # Never migrate to FORMAL. Formal POST_SCORE=1 path above unchanged.
            try:
                from pathlib import Path as _Path
                _runtime = _Path(
                    os.environ.get(
                        "MUBENCH_RUNTIME_ROOT",
                        str(_Path(__file__).resolve().parent),
                    )
                )
                if str(_runtime) not in sys.path:
                    sys.path.insert(0, str(_runtime))
                from formal_post_score_rice import cleanup_attempt_seal_audio
                _res = cleanup_attempt_seal_audio(
                    output_dir,
                    require_rice=False,
                    delete_oracle=False,
                )
                try:
                    write_json(output_dir / "_attempt_seal_cleanup_receipt.json", _res)
                except Exception:
                    pass
            except Exception as _e6b_exc:
                try:
                    write_json(
                        output_dir / "_attempt_seal_cleanup_receipt.json",
                        {
                            "status": "hook_error",
                            "error": str(_e6b_exc),
                            "error_type": type(_e6b_exc).__name__,
                        },
                    )
                except Exception:
                    pass
    # --- end E6 ---

    return {"case_id": case_id, "execution_protocol_version": PROTOCOL_VERSION,
            "probe_count": 0, "provider_session_id": final_session_id,
            "score": score_payload, "audio": str(output_dir / "audio" / "both.wav")}




def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--logical-tick-ms", type=int)
    extra.add_argument("--logical-max-seconds", type=float, default=1800)
    extra.add_argument("--bilateral-silence-seconds", type=int)
    extra.add_argument("--prior-partial-dir", type=Path)
    extra.add_argument("--openweight-endpoint", type=Path)
    extra.add_argument("--seeduplex-voice", default="zh_female_vv_jupiter_bigtts")
    extra.add_argument("--realtime-provider", choices=("openai", "xai", "gemini", "seeduplex", "moss"), default="openai")
    extra.add_argument("--moss-endpoint", type=Path)
    extra.add_argument("--planner-observation", choices=("auto", "tau_proportional", "audible_asr"), default="auto")
    opts, remaining = extra.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = parse_args()
    args.tts_run_namespace = resolve_tts_run_namespace(args)
    args.openweight_endpoint = opts.openweight_endpoint
    args.seeduplex_voice = opts.seeduplex_voice
    args.realtime_provider = opts.realtime_provider
    args.moss_endpoint = opts.moss_endpoint
    if opts.realtime_provider == "moss" and (not opts.moss_endpoint or opts.openweight_endpoint):
        raise ValueError("MOSS requires --moss-endpoint and cannot use an open-weight endpoint")
    args.planner_observation = opts.planner_observation
    args.prior_partial_dir = opts.prior_partial_dir
    endpoint = json.loads(opts.openweight_endpoint.read_text()) if opts.openweight_endpoint else None
    args.logical_tick_ms = opts.logical_tick_ms or (int(endpoint["native_quantum_ms"]) if endpoint else 200)
    if endpoint:
        args.realtime_model = endpoint["model"]
    args.logical_max_seconds = opts.logical_max_seconds
    if opts.bilateral_silence_seconds is not None:
        args.bilateral_silence_seconds = opts.bilateral_silence_seconds
    else:
        policy_path = Path(__file__).resolve().parent.parent / "control/bilateral_silence_policy.json"
        policy = _load_json(policy_path, {})
        claim = _load_json(args.env.parent / "claim.json", {})
        claimed_at = claim.get("started", 0)
        new_claim = claimed_at > policy.get("effective_after", float("inf"))
        legacy_attempt = str(args.env.parent.resolve()) in policy.get("legacy_attempts", [])
        args.bilateral_silence_seconds = (
            policy.get("seconds", 0) if new_claim and not legacy_attempt else 0)
    if args.bilateral_silence_seconds < 0:
        raise ValueError("bilateral-silence-seconds must be non-negative")
    if args.logical_tick_ms <= 0 or args.logical_tick_ms % 20:
        raise ValueError("logical tick must be a positive multiple of native 20 ms packets")
    if args.logical_max_seconds <= 0:
        raise ValueError("logical-max-seconds must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("use a new, empty, versioned output directory")
    if args.frozen_plan_dir or args.pre_rendered_user_track:
        raise ValueError("single-session mode forbids Probe/Freeze replay inputs")
    configure_tau2_import(args.tau2_src)
    if args.tts_backend == "cosyvoice":
        configure_cosyvoice_import(args.cosyvoice_repo, args.cosyvoice_model_dir)
    bridge = load_bridge(args.bridge)
    values = bridge.load_env(args.env)
    required = ["GPT_URL", "GPT_KEY"]
    if not endpoint and opts.realtime_provider != "moss":
        prefix = {"xai": "REALTIME_GROK", "gemini": "REALTIME_GEMINI", "openai": "REALTIME_GPT", "seeduplex": "REALTIME_SEEDUPLEX"}[opts.realtime_provider]
        required.extend([prefix + "_URL", prefix + "_KEY"])
    for name in required:
        if not values.get(name):
            raise ValueError("missing environment entry: " + name)
    os.environ["OPENAI_API_KEY"] = values["GPT_KEY"]
    from experiments.tau_multiuser_voice.cases import CaseCatalog
    requested_ids = list(args.case_id)
    catalog_cases = CaseCatalog.load_default()
    case_by_id = {item.case_id: item for item in catalog_cases}
    cases = []
    for cid in requested_ids:
        if cid == "all":
            cases.extend(catalog_cases)
        else:
            try:
                cases.append(case_by_id[cid])
            except KeyError as exc:
                raise KeyError(f"unknown case_id: {cid}") from exc
    if args.participant_scale:
        scale = {"small": "small_2_3", "large": "large_4_7"}[args.participant_scale]
        cases = [c for c in cases if c.variant and c.variant.participant_scale.value == scale]
    if args.speech_condition:
        cases = [c for c in cases if c.variant and c.variant.speech_condition.value == args.speech_condition]
    if args.comprehension_level:
        cases = [c for c in cases if c.variant and c.variant.comprehension_level.value == args.comprehension_level]
    ids = list(dict.fromkeys(c.case_id for c in cases))
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard configuration")
    ids = ids[args.shard_index::args.shard_count]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_cases_for_preflight = [CaseCatalog.get(case_id) for case_id in ids]
    preflight_path = write_tts_registration_preflight(selected_cases_for_preflight, args)
    write_json(args.output_dir / "run_config.json", {
        "execution_protocol": "mu-single-session-logical-v14.0", "case_ids": ids,
        "execution_implementation_version": "v14.2-rc1",
        "realtime_model": args.realtime_model, "planner_model": args.planner_model,
        "tts_backend": args.tts_backend, "logical_tick_ms": args.logical_tick_ms,
        "tts_run_namespace": args.tts_run_namespace,
        "tts_registration_preflight": str(preflight_path),
        "logical_max_seconds": args.logical_max_seconds, "probe_execution": "none",
        "baseline_commit": "338ba47f0a441a771b2412d0c0ed4c2af376b6d3",
        "integrity_version": "mu-bench-v14.2-integrity-rc1",
        "arguments": {k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v
                      for k, v in vars(args).items()},
    })
    summaries = []
    for cid in ids:
        try:
            summaries.append(run_case(args, cid, bridge, values))
        except Exception as exc:
            from experiments.mu_bench.logical_time.integrity import failure_record, OutputArtifactConflictError
            if isinstance(exc, OutputArtifactConflictError):
                # Never record our failure under an existing case belonging
                # to another attempt. The new run's summary records refusal.
                summaries.append({"case_id": cid, "status": "refused",
                                  "classification": "artifact_conflict", "valid_model_result": False})
                if not args.continue_on_error:
                    raise
                continue
            phase_path = args.output_dir / cid / "attempt_phase.json"
            phase = json.loads(phase_path.read_text()).get("phase", "simulation") if phase_path.exists() else "simulation"
            failure = failure_record(exc, case_id=cid, phase=phase)
            if failure["classification"] == "evaluation_invalid":
                write_json(args.output_dir / cid / "evaluation_invalid.json", failure)
            elif failure["classification"] == "simulator_invalid":
                write_json(args.output_dir / cid / "simulator_invalid.json", failure)
            write_json(args.output_dir / cid / "runner_error.json", failure)
            summaries.append(failure)
            if not args.continue_on_error:
                raise
        write_json(args.output_dir / "summary.partial.json", summaries)
    write_json(args.output_dir / "summary.json", summaries)


if __name__ == "__main__":
    main()
