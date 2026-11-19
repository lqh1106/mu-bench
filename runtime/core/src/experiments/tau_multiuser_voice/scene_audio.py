"""Post-enrollment, pre-activation delivery of read-only source audio.

Only explicitly authored source content is spoken. The narrator is an audio
source, never an enrolled participant or a task event.
"""
import copy
import hashlib
import json
import wave
from pathlib import Path

from experiments.tau_multiuser_voice.audio import QueuedAudio, SpeakerAudioQueue
from experiments.tau_multiuser_voice.scene_context import get_initial_context

NARRATOR = '__initial_source_narrator__'

def source_text(case):
    context = get_initial_context(case)
    if not context:
        return ''
    return ('Background information for the assistant. This is a read-only service or shared-document recording, '
            'not a participant request. Listen to the information; the conversation will start after the activation message. '
            + context['content'] + '\nEnd of background information.')

def render_source(case, tts, text, *, short_segments=False):
    """Use a distinct existing catalog voice without rebinding task speakers."""
    from experiments.tau_multiuser_voice.cases import CaseCatalog
    from experiments.tau_multiuser_voice.tts_http import RemoteCosyVoiceTTS
    original = tts.provenance()
    occupied = set(original.get('assigned_voice_catalog_ids', {}).values())
    if not occupied:
        raise ValueError('source narrator requires the formal CosyVoice catalog binding')
    seen = set()
    for candidate in CaseCatalog.load_default():
        signature = tuple(u.voice.voice_id for u in candidate.users)
        if signature in seen:
            continue
        seen.add(signature)
        if isinstance(tts, RemoteCosyVoiceTTS):
            suffix = hashlib.sha256((tts.run_namespace + case.case_id).encode()).hexdigest()[:24]
            narrator_tts = RemoteCosyVoiceTTS(endpoint_file=tts.endpoint_file,
                run_namespace='initial-source-' + suffix, case_id=candidate.case_id,
                assignment_seed=candidate.seed, timeout_seconds=tts.timeout_seconds)
        else:
            # Share loaded weights only; register_profiles replaces binding maps
            # on this shallow copy, leaving task speaker assignments untouched.
            narrator_tts = copy.copy(tts)
            narrator_tts.case_id = candidate.case_id
            narrator_tts.assignment_seed = candidate.seed
        try:
            narrator_tts.register_profiles(candidate.users)
            provenance = narrator_tts.provenance()
            mapping = provenance.get('assigned_voice_catalog_ids', {})
            voice = next((u.voice.voice_id for u in candidate.users
                          if mapping.get(u.voice.voice_id) and mapping[u.voice.voice_id] not in occupied), None)
            if voice is None:
                continue
            if short_segments:
                from .registration_audio import synthesize
                pcm, audio_segments = synthesize(narrator_tts, text, voice)
            else:
                pcm = narrator_tts.synthesize(text, voice)
                audio_segments = []
            if not pcm or len(pcm) % 2 or not any(pcm):
                raise ValueError('initial source TTS returned empty, invalid or zero PCM')
            if narrator_tts.sample_rate != tts.sample_rate:
                raise ValueError('initial source TTS sample rate mismatch')
            return pcm, {'voice_id': voice, 'voice_catalog_id': mapping[voice],
                         'task_voice_catalog_ids': sorted(occupied),
                         'tts_binding_case_id': candidate.case_id,
                         'sample_rate': tts.sample_rate, 'audio_segments': audio_segments}
        finally:
            if isinstance(narrator_tts, RemoteCosyVoiceTTS):
                narrator_tts.release_case()
    raise ValueError('no distinct source narrator available in the existing voice catalog')

def install_source(enrollment, pcm, text, output_dir, receipt, voice):
    """Insert source audio and shift activation; keep every task-relative time."""
    if hasattr(enrollment, 'initial_source_item'):
        raise ValueError('initial source is already scheduled')
    start = enrollment.enrollment_complete_ms
    item = QueuedAudio(audio_id='initial-assistant-source', speaker_id=NARRATOR,
        text=text, start_ms=start, pcm_s16le=pcm, sample_rate=enrollment.tts.sample_rate,
        metadata={'phase': 'initial_assistant_context', 'source_kind': 'read_only_recording'})
    end = enrollment._align(item.end_ms)
    shift = end - start
    for queue in enrollment.queues.values():
        for existing in queue.items:
            if existing.metadata.get('phase') == 'assistant_activation':
                existing.start_ms += shift
    if enrollment.activation_trace is not None:
        enrollment.activation_trace = enrollment.activation_trace.model_copy(update={
            'start_ms': enrollment.activation_trace.start_ms + shift,
            'end_ms': enrollment.activation_trace.end_ms + shift})
    enrollment.task_start_ms += shift
    queue = SpeakerAudioQueue(NARRATOR, enrollment.tts.sample_rate)
    queue.schedule(item)
    enrollment.queues[NARRATOR] = queue
    enrollment.initial_source_item = item
    enrollment.initial_source_end_ms = end
    directory = Path(output_dir) / 'initial_context'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'transcript.txt').write_text(text, encoding='utf-8')
    with wave.open(str(directory / 'source.wav'), 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2)
        stream.setframerate(enrollment.tts.sample_rate); stream.writeframes(pcm)
    receipt.update(transport='post_registration_pre_task_tts',
        delivery_phase='after_enrollment_before_activation', delivery_status='prepared',
        source_start_ms=start, source_audio_end_ms=item.end_ms, source_phase_end_ms=end,
        task_start_ms=enrollment.task_start_ms, source_pcm_sha256=hashlib.sha256(pcm).hexdigest(),
        spoken_text_sha256=hashlib.sha256(text.encode()).hexdigest(), narrator=voice,
        task_event_created=False, enrolled_participant_added=False, submitted_until_ms=start)
    write_receipt(output_dir, receipt)
    return end

def write_receipt(output_dir, receipt):
    path = Path(output_dir) / 'initial_assistant_context.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)

def record_source_tick(output_dir, receipt, end_ms):
    receipt['submitted_until_ms'] = min(end_ms, receipt['source_phase_end_ms'])
    receipt['delivery_status'] = ('audio_submitted' if end_ms >= receipt['source_phase_end_ms'] else 'submitting')
    # Submission through the normal audio path is not a semantic acknowledgement.
    write_receipt(output_dir, receipt)
