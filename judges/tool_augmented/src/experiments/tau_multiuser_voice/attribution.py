"""Map observable SUT attribution outputs back to oracle speech events."""

from __future__ import annotations

from typing import Any

from experiments.tau_multiuser_voice.models import MultiUserEpisodeTrace


def match_explicit_attributions(trace: MultiUserEpisodeTrace) -> None:
    """Match explicit SUT reports to the latest causally available user event."""
    unmatched = list(trace.spoken_events)
    for attribution in sorted(
        trace.speaker_attributions, key=lambda item: item.task_timestamp_ms
    ):
        completed = [
            event
            for event in unmatched
            if event.end_ms <= attribution.task_timestamp_ms
        ]
        if completed:
            event = max(completed, key=lambda item: item.end_ms)
        else:
            active = [
                event
                for event in unmatched
                if event.start_ms
                <= attribution.task_timestamp_ms
                <= event.end_ms + 2000
            ]
            if not active:
                continue
            event = max(active, key=lambda item: item.start_ms)
        attribution.matched_event_id = event.event_id
        attribution.matched_proposal_id = event.proposal_id
        event.sut_attributed_speaker_id = attribution.predicted_speaker_id
        event.sut_inferred_addressee = attribution.predicted_addressee
        event.sut_attribution_source = "explicit"
        unmatched.remove(event)


def apply_attribution_evidence(
    trace: MultiUserEpisodeTrace,
    assistant_intervals: list[dict[str, Any]],
    tool_log: list[dict[str, Any]],
) -> None:
    """Use explicit reports first, then observable tool/response evidence."""
    match_explicit_attributions(trace)
    events = sorted(trace.spoken_events, key=lambda item: item.start_ms)
    for index, event in enumerate(events):
        window_end = (
            events[index + 1].start_ms
            if index + 1 < len(events)
            else event.end_ms + 5000
        )
        actor_calls = [
            call
            for call in tool_log
            if event.start_ms <= call["timestamp_ms"] < window_end
            and call.get("authorization_basis") != "model_declared_caller"
        ]
        if event.sut_attributed_speaker_id is None and actor_calls:
            event.sut_attributed_speaker_id = actor_calls[0]["actor_id"]
            event.sut_attribution_source = "tool"
        response_started = any(
            event.start_ms <= interval["start_ms"] < window_end
            for interval in assistant_intervals
        )
        if (
            event.sut_inferred_addressee is None
            and event.addressee == "assistant"
            and (actor_calls or response_started)
        ):
            event.sut_inferred_addressee = "assistant"
        elif (
            event.sut_inferred_addressee is None
            and event.addressee not in {None, "assistant"}
            and not response_started
        ):
            event.sut_inferred_addressee = event.addressee
        if event.sut_attribution_source is None and event.sut_inferred_addressee:
            event.sut_attribution_source = "heuristic"
