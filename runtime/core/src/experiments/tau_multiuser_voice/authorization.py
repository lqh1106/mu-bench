"""Deterministic authorization oracle kept outside the dialogue model."""

from __future__ import annotations

from experiments.tau_multiuser_voice.models import (
    AuthorizationDecision,
    MultiUserVoiceCase,
    UserRuntimeState,
)


class AuthorizationOracle:
    """Resolve active action/resource grants without trusting speaker claims."""

    def __init__(self, case: MultiUserVoiceCase) -> None:
        self.case = case
        self._profiles = {user.user_id: user for user in case.users}

    def decide(
        self,
        *,
        actor_id: str,
        action: str,
        resource_id: str,
        runtime: UserRuntimeState,
    ) -> AuthorizationDecision:
        """Return whether the actor's currently active grants cover the action."""
        if actor_id not in self._profiles:
            return AuthorizationDecision(
                actor_id=actor_id,
                action=action,
                resource_id=resource_id,
                allowed=False,
                reason="unknown_actor",
            )
        if runtime.user_id != actor_id:
            raise ValueError("runtime belongs to a different actor")

        matches = []
        for grant in self._profiles[actor_id].permissions:
            if grant.grant_id not in runtime.active_grant_ids:
                continue
            action_matches = "*" in grant.actions or action in grant.actions
            resource_matches = "*" in grant.resources or resource_id in grant.resources
            if action_matches and resource_matches:
                matches.append(grant.grant_id)
        return AuthorizationDecision(
            actor_id=actor_id,
            action=action,
            resource_id=resource_id,
            allowed=bool(matches),
            matching_grant_ids=matches,
            reason="active_grant" if matches else "no_active_matching_grant",
        )
