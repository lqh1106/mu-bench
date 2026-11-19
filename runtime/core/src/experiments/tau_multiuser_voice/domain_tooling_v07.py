"""Deprecated v0.7 fixture source retained for migration compatibility.

Do not expose the generic ``inspect_*`` / ``apply_*_resolution`` definitions in
this module to a model.  The active catalog and runner use
``business_tooling_v07``.  That module temporarily imports only the scenario
fixture facts here so existing case content remains stable during migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, Signature
from typing import Any, Callable, Literal

from experiments.tau_multiuser_voice.models import EnvironmentBinding
from experiments.tau_multiuser_voice.virtual_tools import (
    BaseVirtualEnvironment,
    VirtualToolDefinition,
)


@dataclass(frozen=True)
class DomainToolContract:
    """Names and argument vocabulary for one deployment domain."""

    domain: str
    resource_noun: str
    resource_key: str
    inspect_tool: str
    apply_tool: str
    subject_key: str
    protected_key: str
    operations: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioToolSemantics:
    """Concrete business facts that distinguish one tool scenario from another."""

    initial_context: dict[str, Any]
    resolution_payload: dict[str, Any]


def _contract(
    domain: str,
    resource_noun: str,
    resource_key: str,
    inspect_tool: str,
    apply_tool: str,
    subject_key: str,
    protected_key: str,
    operations: str,
) -> DomainToolContract:
    return DomainToolContract(
        domain=domain,
        resource_noun=resource_noun,
        resource_key=resource_key,
        inspect_tool=inspect_tool,
        apply_tool=apply_tool,
        subject_key=subject_key,
        protected_key=protected_key,
        operations=tuple(operations.split()),
    )


DOMAIN_TOOL_CONTRACTS: dict[str, DomainToolContract] = {
    item.domain: item
    for item in (
        _contract(
            "airline", "booking", "booking_id", "inspect_booking_context",
            "apply_itinerary_resolution", "passenger_ids",
            "preserved_passenger_ids",
            "split_rebooking child_seat_correction payer_traveler_cancel "
            "gate_announcement_collision mobility_repeat",
        ),
        _contract(
            "retail", "order workflow", "order_id", "inspect_order_context",
            "apply_order_resolution", "shopper_ids", "preserved_shopper_ids",
            "gift_cancel_revoke shared_address_correction comparison_backchannel "
            "return_side_talk cart_quantity_revision",
        ),
        _contract(
            "telecom", "service account", "service_account_id",
            "inspect_service_account", "apply_service_account_resolution",
            "line_user_ids", "preserved_line_user_ids",
            "family_data_lines lost_phone_number_fix router_restart_pushback "
            "roaming_interpreter credit_during_result",
        ),
        _contract(
            "finance_billing", "billing case", "billing_case_id",
            "inspect_billing_context", "apply_billing_resolution",
            "account_holder_ids", "nonconsenting_holder_ids",
            "joint_autopay_conflict two_traveler_exchange card_benefit_privacy "
            "invoice_amount_revision unsupported_transfer_side_request",
        ),
        _contract(
            "housing_location", "property search", "property_search_id",
            "inspect_property_search", "apply_property_search_resolution",
            "renter_ids", "protected_renter_ids",
            "couple_filter_conflict roommate_commutes accessibility_late_correction "
            "guarantor_not_tenant group_viewing_schedule",
        ),
        _contract(
            "travel_identity", "document case", "document_case_id",
            "inspect_document_context", "apply_document_resolution",
            "document_owner_ids", "protected_document_owner_ids",
            "passport_digit_correction multi_traveler_documents "
            "document_interpreter adult_parent_disagreement background_number_privacy",
        ),
        _contract(
            "shared_home", "household task", "household_task_id",
            "inspect_household_context", "apply_household_resolution",
            "resident_ids", "protected_resident_ids",
            "parallel_timers media_backchannel grocery_threads thermostat_consent "
            "guest_door_authority",
        ),
        _contract(
            "vehicle", "shared trip", "trip_id", "inspect_trip_context",
            "apply_trip_resolution", "rider_ids", "protected_rider_ids",
            "driver_destination music_side_talk warning_provenance "
            "late_rider_summary road_trip_fairness",
        ),
        _contract(
            "meeting", "meeting", "meeting_id", "inspect_meeting_context",
            "apply_meeting_resolution", "participant_ids",
            "dissenting_participant_ids",
            "overlap_votes assignee_correction quote_not_vote quiet_dissenter "
            "result_due_date_update",
        ),
        _contract(
            "education", "learning session", "learning_session_id",
            "inspect_learning_context", "apply_learning_resolution",
            "learner_ids", "protected_learner_ids",
            "fraction_answers peer_whisper explanation_backchannel project_roles "
            "translated_answer",
        ),
        _contract(
            "healthcare_care", "care plan", "care_plan_id",
            "inspect_care_plan", "apply_care_plan_resolution", "patient_ids",
            "protected_patient_ids",
            "appointment_consent medication_list_source care_accessibility "
            "visitor_privacy discharge_repeat",
        ),
        _contract(
            "hospitality", "reservation", "reservation_id",
            "inspect_reservation_context", "apply_reservation_resolution",
            "guest_ids", "protected_guest_ids",
            "party_size_fix allergy_ownership table_side_talk bill_split "
            "late_roommate_privacy",
        ),
        _contract(
            "public_services", "public-service case", "service_case_id",
            "inspect_public_service_case", "apply_public_service_resolution",
            "resident_ids", "protected_resident_ids",
            "transit_accessibility appointment_interpreter library_privacy "
            "community_quiet_floor alert_plain_repeat",
        ),
        _contract(
            "it_support", "incident workstream", "incident_id",
            "inspect_incident_workstream", "apply_incident_workstream_resolution",
            "service_owner_ids", "protected_service_owner_ids",
            "service_owner_incident reset_privacy rollback_correction "
            "helpdesk_backchannel incident_bridge_threads",
        ),
        _contract(
            "events_entertainment", "event plan", "event_plan_id",
            "inspect_event_plan", "apply_event_plan_resolution", "attendee_ids",
            "protected_attendee_ids",
            "ticket_ownership playlist_cheer accessible_seat child_content_filter "
            "schedule_time_fix",
        ),
        _contract(
            "family_coordination", "family plan", "family_plan_id",
            "inspect_family_plan", "apply_family_plan_resolution",
            "family_member_ids", "protected_family_member_ids",
            "pickup_assignments elder_care_schedule chore_quiet_child "
            "calendar_privacy packing_threads",
        ),
        _contract(
            "emergency_incident", "incident record", "incident_record_id",
            "inspect_emergency_record", "apply_emergency_record_resolution",
            "affected_person_ids", "protected_person_ids",
            "evacuation_accessibility lost_child_description roll_call_overlap "
            "dispatcher_relay shelter_resources",
        ),
    )
}


def _sem(
    initial_context: dict[str, Any], resolution_payload: dict[str, Any]
) -> ScenarioToolSemantics:
    return ScenarioToolSemantics(initial_context, resolution_payload)


# This table is intentionally explicit.  It is the semantic oracle that stops a
# model from passing merely by selecting the right domain tool and actor while
# ignoring the corrected number, destination, source, consent state, or item.
SCENARIO_TOOL_SEMANTICS: dict[str, ScenarioToolSemantics] = {
    "airline_01_split_rebooking": _sem(
        {"requested_scope": ["primary", "live"], "requested_action": "rebook_together"},
        {"rebook_user_ids": ["primary"], "keep_original_user_ids": ["live"]},
    ),
    "airline_02_child_seat_correction": _sem(
        {"relayed_seat": "window", "passenger_user_id": "live"},
        {"passenger_user_id": "live", "seat_preference": "aisle"},
    ),
    "airline_03_payer_traveler_cancel": _sem(
        {"payer_user_id": "primary", "traveler_user_id": "live"},
        {"traveler_user_id": "live", "cancellation_status": "denied_no_consent"},
    ),
    "retail_01_gift_cancel_revoke": _sem(
        {"order_type": "gift", "cancellation_status": "requested"},
        {"order_status": "active", "cancellation_status": "revoked_by_recipient"},
    ),
    "retail_02_shared_address_correction": _sem(
        {"requested_house_number": 81, "street": "Lake Street"},
        {"corrected_house_number": 18, "status": "pending_owner_confirmation"},
    ),
    "retail_04_return_side_talk": _sem(
        {"item_condition": "damaged", "refund_destination": "original_payment_method"},
        {"return_status": "submitted", "dessert_added": False},
    ),
    "retail_05_cart_quantity_revision": _sem(
        {"item": "filters", "obsolete_quantity": 4},
        {"item": "filters", "quantity": 2},
    ),
    "telecom_01_family_data_lines": _sem(
        {"group_request_gigabytes": 5},
        {"data_pass_gigabytes_by_user": {"primary": 5, "live": 10}},
    ),
    "telecom_02_lost_phone_number_fix": _sem(
        {"initial_lost_line_suffix": "0182"},
        {"lost_line_suffix": "0812", "preserve_active_line_suffix": "0182"},
    ),
    "telecom_03_router_restart_pushback": _sem(
        {"router_action": "restart_now"},
        {"router_action": "paused", "reason": "active_emergency_video_call"},
    ),
    "telecom_04_roaming_interpreter": _sem(
        {"request_source_user_id": "primary", "relay_user_id": "live"},
        {"line_owner_user_id": "primary", "region": "Japan", "enabled": True},
    ),
    "telecom_05_credit_during_result": _sem(
        {"diagnostic_result": "outage_confirmed", "documented_credit_available": True},
        {"credit_recipient_user_id": "live", "credit_basis": "documented_outage_policy"},
    ),
    "finance_billing_01_joint_autopay_conflict": _sem(
        {"account_type": "joint_checking", "autopay_status": "requested"},
        {"autopay_status": "unchanged", "reason": "co_owner_withheld_consent"},
    ),
    "finance_billing_02_two_traveler_exchange": _sem(
        {"primary_conversion": {"amount": 500, "target_currency": "EUR"}},
        {"conversions": {"primary": [500, "EUR"], "live": [200, "JPY"]}},
    ),
    "finance_billing_04_invoice_amount_revision": _sem(
        {"obsolete_approved_amount": 15000},
        {"approved_amount": 50000, "obsolete_amount_removed": True},
    ),
    "housing_location_01_couple_filter_conflict": _sem(
        {"location_filter": "downtown", "bedrooms": 1},
        {"wheelchair_access": "mandatory", "downtown": "preferred_not_mandatory"},
    ),
    "housing_location_02_roommate_commutes": _sem(
        {"primary": {"mode": "walking", "destination": "office"}},
        {"commutes": {"primary": ["walking", "office"], "live": ["transit", "campus"]}},
    ),
    "housing_location_03_accessibility_late_correction": _sem(
        {"existing_filters_loaded": True},
        {"step_free_entry": "mandatory", "requesting_user_id": "live"},
    ),
    "housing_location_05_group_viewing_schedule": _sem(
        {"group_slot_requested": True},
        {"separate_availability_user_ids": ["live"], "conflict_status": "unresolved"},
    ),
    "travel_identity_01_passport_digit_correction": _sem(
        {"obsolete_candidate": "P8899011"},
        {"corrected_candidate": "P8890011", "status": "pending_owner_verification"},
    ),
    "travel_identity_02_multi_traveler_documents": _sem(
        {"document_type": "passport", "group_update_requested": True},
        {"separate_owner_scopes": ["primary", "live"], "merge_numbers": False},
    ),
    "travel_identity_03_document_interpreter": _sem(
        {"document_owner_user_id": "primary", "relay_user_id": "live"},
        {"visa_number": "V123456", "source_user_id": "primary", "relay_user_id": "live"},
    ),
    "shared_home_01_parallel_timers": _sem(
        {"primary_timer": ["oven", 30], "obsolete_live_timer_minutes": 20},
        {"timers": {"primary": ["oven", 30], "live": ["stovetop", 12]}},
    ),
    "shared_home_02_media_backchannel": _sem(
        {"album": "Evening Jazz", "scope": "full_album"},
        {"playback_status": "continue", "album": "Evening Jazz"},
    ),
    "shared_home_03_grocery_threads": _sem(
        {"primary_items": ["milk", "bread"]},
        {"lists": {"primary": ["milk", "bread"], "live": ["rice"]}, "live_excluded_items": ["milk"]},
    ),
    "vehicle_01_driver_destination": _sem(
        {"passenger_destination": "airport"},
        {"active_destination": "school", "route_authority_user_id": "live"},
    ),
    "vehicle_02_music_side_talk": _sem(
        {"active_destination": "airport"},
        {"active_destination": "airport", "media_changed": False},
    ),
    "vehicle_03_warning_provenance": _sem(
        {"active_destination": "airport", "eta_minutes": 42},
        {},
    ),
    "meeting_01_overlap_votes": _sem(
        {"proposal": "budget", "primary_vote": "support"},
        {"votes": {"primary": "support", "live": "oppose"}, "consensus": False},
    ),
    "meeting_02_assignee_correction": _sem(
        {"obsolete_owner": "Blake", "obsolete_due_date": "Friday"},
        {"owner": "Cameron", "due_date": "Monday"},
    ),
    "meeting_04_quiet_dissenter": _sem(
        {"premature_consensus": "support"},
        {"live_position": "oppose", "consensus": False, "floor_received": True},
    ),
    "meeting_05_result_due_date_update": _sem(
        {"task": "audit", "recorded_due_date": "Friday"},
        {"task": "audit", "due_date": "Monday"},
    ),
    "education_01_fraction_answers": _sem(
        {"primary_answer": "1/2"},
        {"answers": {"primary": "1/2", "live": "2/3"}, "merge_speakers": False},
    ),
    "education_04_project_roles": _sem(
        {"roles": ["research", "slides", "speaking"]},
        {"live_accepts": ["slides"], "live_rejects": ["speaking"], "conflict_status": "unresolved"},
    ),
    "healthcare_care_01_appointment_consent": _sem(
        {"caregiver_requested_slot": "morning"},
        {"appointment_slot": "afternoon", "preference_owner_user_id": "live"},
    ),
    "healthcare_care_02_medication_list_source": _sem(
        {"source": "written_list"},
        {"documented_dose": "10 mg", "confirmation_status": "patient_confirmation_required"},
    ),
    "hospitality_01_party_size_fix": _sem(
        {"obsolete_party_size": 8},
        {"party_size": 6},
    ),
    "hospitality_03_table_side_talk": _sem(
        {"primary_order": "vegetarian entree"},
        {"primary_order": "vegetarian entree", "dessert_added": False},
    ),
    "hospitality_04_bill_split": _sem(
        {"obsolete_split": "equal"},
        {"live_pays_items": ["entree", "tea"], "equal_split": False},
    ),
    "public_services_01_transit_accessibility": _sem(
        {"obsolete_assistance": "standard_boarding"},
        {"entrance": "lift_equipped", "preference_owner_user_id": "live"},
    ),
    "public_services_02_appointment_interpreter": _sem(
        {"requester_user_id": "primary", "relay_user_id": "live"},
        {"requester_user_id": "primary", "slot": "Tuesday morning", "relay_user_id": "live"},
    ),
    "it_support_01_service_owner_incident": _sem(
        {"proposed_action": "restart_production_service", "active_transaction": True},
        {"restart_status": "deferred", "until": "active_transaction_finishes"},
    ),
    "it_support_03_rollback_correction": _sem(
        {"obsolete_service": "beta", "version": "4.1"},
        {"service": "alpha", "version": "4.1"},
    ),
    "events_entertainment_01_ticket_ownership": _sem(
        {"ticket_count": 4, "allocation_order_loaded": True},
        {"live_ticket_status": "retained", "assign_to_late_attendee": False},
    ),
    "events_entertainment_02_playlist_cheer": _sem(
        {"playlist_scope": "complete_set"},
        {"playback_status": "continue", "playlist_scope": "complete_set"},
    ),
    "events_entertainment_03_accessible_seat": _sem(
        {"group_preference": "closest_available"},
        {"seat_requirement": "wheelchair_accessible_row", "proximity_priority": "secondary"},
    ),
    "events_entertainment_05_schedule_time_fix": _sem(
        {"recorded_start_time": "6:00"},
        {},
    ),
    "family_coordination_01_pickup_assignments": _sem(
        {"children": ["Maya", "Leo"], "single_driver_requested": True},
        {"assignments": {"Maya": "live", "Leo": "unresolved"}},
    ),
    "family_coordination_03_chore_quiet_child": _sem(
        {"allocation_status": "draft"},
        {"live_existing_chore": "dishes_daily", "allocation_status": "reconsider_required"},
    ),
    "emergency_incident_02_lost_child_description": _sem(
        {"guardian_reported_jacket": "blue"},
        {"jacket_color": "green", "correction_source_user_id": "live"},
    ),
    "emergency_incident_03_roll_call_overlap": _sem(
        {"primary_report": "all_accounted_for"},
        {"reports": {"primary": "all_accounted_for", "live": "one_unaccounted_for"}, "overall_status": "unresolved"},
    ),
}


def semantics_for_scenario(scenario_id: str) -> ScenarioToolSemantics | None:
    return SCENARIO_TOOL_SEMANTICS.get(scenario_id)


RESOLUTION_MODES = (
    "speaker_scoped_resolution",
    "apply_live_correction",
    "continue_after_backchannel",
    "ignore_side_talk_preserve_task",
    "handle_subtask_then_resume",
    "stop_and_concise_restate",
    "pause_and_reconcile",
    "selective_common_ground",
    "preserve_independent_threads",
    "commit_latest_revision",
    "integrate_tool_result_then_resume",
    "clarify_ambiguous_recipient",
    "preserve_floor_and_authority",
    "cancel_or_leave_unchanged",
    "preserve_source_and_relay",
    "targeted_plain_restatement",
)


def contract_for_domain(domain: str) -> DomainToolContract | None:
    """Return the domain contract, accepting either raw or v0.7 binding names."""
    return DOMAIN_TOOL_CONTRACTS.get(domain.removeprefix("v07_"))


class DomainScenarioEnvironment(BaseVirtualEnvironment):
    """One case-scoped domain state machine with an auditable semantic write."""

    def __init__(self, binding: EnvironmentBinding) -> None:
        contract = contract_for_domain(binding.domain)
        if contract is None:
            raise ValueError(f"v0.7 domain has no tool contract: {binding.domain}")
        self.binding = binding
        self.contract = contract
        self.definitions = {
            contract.inspect_tool: VirtualToolDefinition(
                name=contract.inspect_tool,
                description=(
                    f"Read the current {contract.resource_noun}, pending domain "
                    "operation, and already-recorded participant scope without "
                    "changing it."
                ),
                effect="read",
            ),
            contract.apply_tool: VirtualToolDefinition(
                name=contract.apply_tool,
                description=(
                    f"Apply one interaction-aware resolution to the current "
                    f"{contract.resource_noun}. Record the authorizing actor, "
                    f"operation, resolution mode, {contract.subject_key}, and "
                    f"{contract.protected_key}, plus the operation-specific "
                    "resolution_payload; do not collapse distinct users or omit "
                    "corrected business values."
                ),
                effect="write",
            ),
        }
        super().__init__(binding.initial_state or None)

    @classmethod
    def default_state(cls) -> dict[str, Any]:
        return {"domain_record": {}}

    def invoke(self, tool_name: str, actor_id: str, **arguments: Any) -> Any:
        """Route instance-defined names to the two deterministic handlers."""
        if tool_name == self.contract.inspect_tool:
            handler_name = "_tool_inspect"
        elif tool_name == self.contract.apply_tool:
            handler_name = "_tool_apply"
        else:
            return super().invoke(tool_name, actor_id, **arguments)
        original = self.definitions
        try:
            # BaseVirtualEnvironment supplies uniform call logging and errors.
            setattr(self, f"_tool_{tool_name}", getattr(self, handler_name))
            return super().invoke(tool_name, actor_id, **arguments)
        finally:
            self.definitions = original

    def _record(self, resource_id: str) -> dict[str, Any]:
        record = self.state.get("domain_record")
        if not isinstance(record, dict) or record.get("resource_id") != resource_id:
            raise ValueError(f"{self.contract.resource_noun} not found: {resource_id}")
        return record

    def _tool_inspect(self, *, actor_id: str, **arguments: Any) -> dict[str, Any]:
        resource_id = str(arguments[self.contract.resource_key])
        record = self._record(resource_id)
        inspected = record.setdefault("inspected_by", [])
        if actor_id not in inspected:
            inspected.append(actor_id)
        return {**record, "last_inspected_by": actor_id}

    def _tool_apply(self, *, actor_id: str, **arguments: Any) -> dict[str, Any]:
        resource_id = str(arguments[self.contract.resource_key])
        record = self._record(resource_id)
        operation = str(arguments["operation"])
        if operation not in self.contract.operations:
            raise ValueError(f"unknown {self.contract.domain} operation: {operation}")
        subject_ids = list(arguments[self.contract.subject_key])
        protected_ids = list(arguments[self.contract.protected_key])
        unknown = (set(subject_ids) | set(protected_ids)) - set(self.binding.principals)
        if unknown:
            raise ValueError(f"unknown participant ids: {sorted(unknown)}")
        record.update(
            status="applied",
            applied_by=actor_id,
            operation=operation,
            resolution_mode=str(arguments["resolution_mode"]),
            resolution_payload=dict(arguments["resolution_payload"]),
            **{
                self.contract.subject_key: subject_ids,
                self.contract.protected_key: protected_ids,
            },
        )
        return record


def make_domain_tool_callable(
    environment: DomainScenarioEnvironment, tool_name: str
) -> Callable[..., dict[str, Any]]:
    """Create a τ-compatible callable with a domain-specific visible signature."""
    contract = environment.contract
    if tool_name not in {contract.inspect_tool, contract.apply_tool}:
        raise ValueError(f"not a tool in {contract.domain}: {tool_name}")

    def invoke(**arguments: Any) -> dict[str, Any]:
        actor_id = str(arguments.pop("actor_id"))
        return environment.invoke(tool_name, actor_id, **arguments)

    invoke.__name__ = tool_name
    invoke.__doc__ = environment.definitions[tool_name].description
    parameters = [
        Parameter("actor_id", Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        Parameter(
            contract.resource_key,
            Parameter.POSITIONAL_OR_KEYWORD,
            annotation=str,
        ),
    ]
    if tool_name == contract.apply_tool:
        operation_type = Literal.__getitem__(contract.operations)
        resolution_type = Literal.__getitem__(RESOLUTION_MODES)
        parameters.extend(
            [
                Parameter(
                    "operation",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=operation_type,
                ),
                Parameter(
                    "resolution_mode",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=resolution_type,
                ),
                Parameter(
                    contract.subject_key,
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=list[str],
                ),
                Parameter(
                    contract.protected_key,
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=list[str],
                ),
                Parameter(
                    "resolution_payload",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=dict[str, Any],
                ),
            ]
        )
    invoke.__signature__ = Signature(parameters, return_annotation=dict[str, Any])
    return invoke
