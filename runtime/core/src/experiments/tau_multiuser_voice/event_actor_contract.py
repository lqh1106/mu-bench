"""Keep user speech actors distinct from observed assistant/environment events."""

from __future__ import annotations

from experiments.tau_multiuser_voice.models import MultiUserVoiceCase


class InvalidUserEventActor(ValueError):
    """A planner requested speech from an actor with no real user session.

    Callers may ask the planner to reconsider its *semantic* decision using
    the same public evidence. They must not repair this by renaming the actor,
    synthesizing assistant speech, or marking a required event completed.
    """

    def __init__(self, actor_id: object, *, case: MultiUserVoiceCase,
                 event_id: str | None = None) -> None:
        self.actor_id = actor_id
        self.case_id = case.case_id
        self.event_id = event_id
        self.allowed_user_ids = tuple(user.user_id for user in case.users)
        self.reason = (
            f"user speech actor {actor_id!r} is not a canonical participant in "
            f"case {case.case_id!r}; allowed user_ids: {self.allowed_user_ids!r}. "
            "Assistant speech, tool results, and environmental observations "
            "belong in WorldObservation, not in the user speech event graph. "
            "Do not rename the actor or supply the assistant's answer through "
            "a user voice. Reconsider the user event using the same evidence."
        )
        super().__init__(f"event {event_id!r}: {self.reason}" if event_id else self.reason)


def validate_user_event_actor(actor_id: object, *, case: MultiUserVoiceCase,
                              event_id: str | None = None) -> None:
    """Validate without aliases, mutation, synthesis, or completion side effects.

    Exact and unique public aliases may be resolved at the planner boundary;
    execution only accepts the resulting canonical ID of an existing user.
    """
    if not isinstance(actor_id, str) or actor_id not in {
        user.user_id for user in case.users
    }:
        raise InvalidUserEventActor(actor_id, case=case, event_id=event_id)
