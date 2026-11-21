"""Production-shaped business tools for the τ-MU Voice v0.7 case grid.

This layer replaces the model-visible ``apply_*_resolution`` abstraction with
stable domain APIs.  The eight variants of a base scenario share the same
fixture and operation contract.  Requester identity is not a function
argument: the runner binds the latest explicit speaker-attribution report to
the oracle audio event and supplies the trusted actor only at execution time.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from inspect import Parameter, Signature
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from experiments.tau_multiuser_voice.domain_tooling_v07 import (
    semantics_for_scenario as legacy_semantics_for_scenario,
)
from experiments.tau_multiuser_voice.models import EnvironmentBinding
from experiments.tau_multiuser_voice.virtual_tools import (
    BaseVirtualEnvironment,
    VirtualToolDefinition,
)

ExecutionPhase = Literal["before_live", "after_live", "forbidden", "inspect_only"]


class BusinessResourceRecord(BaseModel):
    """Validated state record stored under a domain-specific resource ID."""

    model_config = ConfigDict(extra="allow")

    resource_type: str
    participant_user_ids: list[str]
    status: Literal["open", "updated"]
    business_state: dict[str, Any] = Field(default_factory=dict)
    last_operation: str | None = None
    last_modified_by: str | None = None
    last_transaction_id: str | None = None


class BusinessToolResult(BaseModel):
    """Common queryable envelope returned by every v0.7 business operation."""

    model_config = ConfigDict(extra="allow")

    transaction_id: str | None
    business_state: dict[str, Any]


class BusinessAuditEvent(BaseModel):
    """Append-only evidence for one successful state-changing operation."""

    transaction_id: str
    tool_name: str
    resource_id: str
    actor_id: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass(frozen=True)
class DomainBusinessContract:
    domain: str
    resource_noun: str
    resource_key: str
    collection_key: str
    inspect_tool: str
    id_prefix: str


@dataclass(frozen=True)
class BusinessOperationSpec:
    scenario_id: str
    domain: str
    tool_name: str
    description: str
    expected_arguments: dict[str, Any]
    effect: Literal["read", "write"] = "write"
    execution_phase: ExecutionPhase = "after_live"
    derived_state: dict[str, Any] = field(default_factory=dict)
    read_result: dict[str, Any] = field(default_factory=dict)
    reversible: bool = True
    errors: tuple[str, ...] = (
        "RESOURCE_NOT_FOUND: the requested business record does not exist.",
        "INVALID_ARGUMENT: an argument has the wrong type or violates the fixture.",
    )
    idempotency: str = (
        "An identical repeated request returns the original transaction result "
        "without applying a second state change."
    )


@dataclass(frozen=True)
class ScenarioBusinessSemantics:
    initial_context: dict[str, Any]
    operation: BusinessOperationSpec

    @property
    def resolution_payload(self) -> dict[str, Any]:
        if self.operation.effect != "write":
            return {}
        return {
            **deepcopy(self.operation.expected_arguments),
            **deepcopy(self.operation.derived_state),
        }


def _domain(
    domain: str,
    noun: str,
    key: str,
    collection: str,
    inspect: str,
    prefix: str,
) -> DomainBusinessContract:
    return DomainBusinessContract(domain, noun, key, collection, inspect, prefix)


DOMAIN_BUSINESS_CONTRACTS = {
    item.domain: item
    for item in (
        _domain("airline", "booking", "booking_id", "bookings", "get_booking_context", "BK"),
        _domain("retail", "order", "order_id", "orders", "get_order_context", "ORD"),
        _domain("telecom", "service account", "service_account_id", "service_accounts", "get_service_account_context", "TEL"),
        _domain("finance_billing", "billing case", "billing_case_id", "billing_cases", "get_billing_case", "BILL"),
        _domain("housing_location", "property search", "property_search_id", "property_searches", "get_property_search", "HOME"),
        _domain("travel_identity", "document case", "document_case_id", "document_cases", "get_document_case", "DOC"),
        _domain("shared_home", "household task", "household_task_id", "household_tasks", "get_household_task", "HSH"),
        _domain("vehicle", "shared trip", "trip_id", "trips", "get_trip_context", "TRIP"),
        _domain("meeting", "meeting", "meeting_id", "meetings", "get_meeting_record", "MTG"),
        _domain("education", "learning session", "learning_session_id", "learning_sessions", "get_learning_session", "EDU"),
        _domain("healthcare_care", "care plan", "care_plan_id", "care_plans", "get_care_plan", "CARE"),
        _domain("hospitality", "reservation", "reservation_id", "reservations", "get_reservation", "RSV"),
        _domain("public_services", "public-service case", "service_case_id", "service_cases", "get_public_service_case", "CIVIC"),
        _domain("it_support", "incident", "incident_id", "incidents", "get_incident_workstream", "INC"),
        _domain("events_entertainment", "event plan", "event_plan_id", "event_plans", "get_event_plan", "EVT"),
        _domain("family_coordination", "family plan", "family_plan_id", "family_plans", "get_family_plan", "FAM"),
        _domain("emergency_incident", "incident record", "incident_record_id", "incident_records", "get_incident_record", "EMG"),
    )
}


def _row(
    tool_name: str,
    description: str,
    arguments: dict[str, Any],
    *,
    effect: Literal["read", "write"] = "write",
    phase: ExecutionPhase = "after_live",
    derived: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    reversible: bool = True,
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "description": description,
        "arguments": arguments,
        "effect": effect,
        "phase": phase,
        "derived": derived or {},
        "result": result or {},
        "reversible": reversible,
    }


# One concrete operation per tool-bearing base scenario.  The same operation is
# reused by the small/large, clean/realistic, and strong/weak variants.
OPERATION_ROWS: dict[str, dict[str, Any]] = {
    "airline_01_split_rebooking": _row(
        "rebook_selected_travelers",
        "Move only the listed consenting travelers to an alternative flight while keeping the other listed travelers on the original itinerary.",
        {"rebook_user_ids": ["primary"], "keep_original_user_ids": ["live"], "target_flight_id": "FLT-MU-220"},
        derived={"rebooking_status": "partially_rebooked"},
    ),
    "airline_02_child_seat_correction": _row(
        "update_passenger_seat_preference",
        "Update one passenger's seat preference on an existing booking.",
        {"passenger_user_id": "live", "seat_preference": "aisle"},
        derived={"seat_preference_status": "updated"},
    ),
    "airline_03_payer_traveler_cancel": _row(
        "cancel_traveler_itinerary",
        "Cancel one adult traveler's itinerary and select a refund destination. Current traveler consent is required.",
        {"traveler_user_id": "live", "refund_destination": "payer_card"},
        phase="forbidden", derived={"itinerary_status": "cancelled"}, reversible=False,
    ),
    "retail_01_gift_cancel_revoke": _row(
        "cancel_gift_order",
        "Cancel a gift order after validating the recipient's current consent.",
        {"recipient_user_id": "live", "reason": "recipient_no_longer_wants_gift"},
        phase="forbidden", derived={"order_status": "cancelled"}, reversible=False,
    ),
    "retail_02_shared_address_correction": _row(
        "create_pending_address_change",
        "Create a pending delivery-address change for owner confirmation without immediately replacing the shipping address.",
        {"house_number": 18, "street": "Lake Street", "confirmation_status": "pending_owner_confirmation"},
        derived={"address_change_status": "pending"},
    ),
    "retail_04_return_side_talk": _row(
        "create_return_request",
        "Create a return request for a delivered item using its condition and refund destination.",
        {"item_condition": "damaged", "refund_destination": "original_payment_method"},
        derived={"return_status": "submitted"},
    ),
    "retail_05_cart_quantity_revision": _row(
        "set_cart_item_quantity",
        "Set the final quantity of one cart item, replacing the obsolete quantity.",
        {"item_name": "filters", "quantity": 2}, derived={"quantity_status": "updated"},
    ),
    "telecom_01_family_data_lines": _row(
        "set_line_data_passes",
        "Apply independent data-pass amounts to specified line owners without changing other lines.",
        {"gigabytes_by_user": {"primary": 5, "live": 10}}, derived={"data_pass_status": "updated"},
    ),
    "telecom_02_lost_phone_number_fix": _row(
        "suspend_lost_line",
        "Suspend the identified lost line while explicitly preserving the active line.",
        {"lost_line_suffix": "0812", "preserve_active_line_suffix": "0182", "reason": "lost_device"},
        derived={"line_0812_status": "suspended", "line_0182_status": "active"}, reversible=False,
    ),
    "telecom_03_router_restart_pushback": _row(
        "defer_router_restart",
        "Record a deferred router restart while an affected user has an active critical session.",
        {"until_condition": "active_emergency_video_call_finishes", "reason": "affected_user_objected"},
        derived={"restart_status": "deferred"},
    ),
    "telecom_04_roaming_interpreter": _row(
        "enable_international_roaming",
        "Enable roaming on the line owner's account for the requested region; a relay does not become the owner.",
        {"line_owner_user_id": "primary", "region": "Japan", "enabled": True},
        derived={"roaming_enabled": True},
    ),
    "telecom_05_credit_during_result": _row(
        "apply_documented_outage_credit",
        "Apply the documented service credit only after a qualifying outage is confirmed.",
        {"credit_recipient_user_id": "live", "credit_basis": "documented_outage_policy"},
        derived={"credit_status": "applied"},
    ),
    "finance_billing_01_joint_autopay_conflict": _row(
        "enable_joint_account_autopay",
        "Enable autopay from a joint funding account only when every required owner consents.",
        {"funding_account_type": "joint_checking"},
        phase="forbidden", derived={"autopay_status": "enabled"}, reversible=False,
    ),
    "finance_billing_02_two_traveler_exchange": _row(
        "quote_currency_conversions",
        "Return separate currency-conversion quotes keyed by requester without merging amounts or currencies.",
        {"requests_by_user": {"primary": {"amount": 500, "target_currency": "EUR"}, "live": {"amount": 200, "target_currency": "JPY"}}},
        effect="read", result={"quotes_are_requester_scoped": True},
    ),
    "finance_billing_04_invoice_amount_revision": _row(
        "record_invoice_approval",
        "Record the final approved invoice amount and replace the obsolete amount.",
        {"approved_amount": 50000, "currency": "USD"}, derived={"approval_status": "recorded"},
    ),
    "housing_location_01_couple_filter_conflict": _row(
        "update_shared_property_filters",
        "Update a shared property search while preserving affected renters' accessibility requirements.",
        {"wheelchair_access": "mandatory", "downtown": "preferred_not_mandatory", "bedrooms": 1},
        derived={"filter_status": "updated"},
    ),
    "housing_location_02_roommate_commutes": _row(
        "compare_resident_commutes",
        "Calculate separate commute results keyed by resident, destination, and transport mode.",
        {"commutes_by_user": {"primary": {"mode": "walking", "destination": "office"}, "live": {"mode": "transit", "destination": "campus"}}},
        effect="read", result={"results_are_resident_scoped": True},
    ),
    "housing_location_03_accessibility_late_correction": _row(
        "add_accessibility_filter",
        "Add a mandatory accessibility filter on behalf of the affected renter.",
        {"requesting_user_id": "live", "step_free_entry": "mandatory"}, derived={"filter_status": "updated"},
    ),
    "housing_location_05_group_viewing_schedule": _row(
        "record_viewing_availability",
        "Record participant-specific viewing availability and preserve unresolved conflicts.",
        {"separate_availability_user_ids": ["live"], "conflict_status": "unresolved"},
        derived={"schedule_status": "needs_reconciliation"},
    ),
    "travel_identity_01_passport_digit_correction": _row(
        "stage_passport_number_update",
        "Stage a corrected passport number for document-owner verification without finalizing it.",
        {"document_owner_user_id": "primary", "passport_number": "P8890011", "verification_status": "pending_owner_verification"},
        derived={"update_status": "staged"},
    ),
    "travel_identity_02_multi_traveler_documents": _row(
        "stage_separate_document_updates",
        "Create separate document-update scopes for each owner without merging document numbers.",
        {"document_owner_user_ids": ["primary", "live"], "merge_numbers": False},
        derived={"update_scope": "separate_owner_records"},
    ),
    "travel_identity_03_document_interpreter": _row(
        "record_relayed_visa_number",
        "Record a relayed visa number while preserving document-owner and interpreter roles.",
        {"document_owner_user_id": "primary", "visa_number": "V123456", "relay_user_id": "live"},
        derived={"record_status": "pending_owner_verification"},
    ),
    "shared_home_01_parallel_timers": _row(
        "upsert_household_timers",
        "Create or replace owner-labeled household timers without merging their durations.",
        {"timers_by_owner": {"primary": {"label": "oven", "minutes": 30}, "live": {"label": "stovetop", "minutes": 12}}},
        derived={"timer_status": "active"},
    ),
    "shared_home_02_media_backchannel": _row(
        "start_album_playback",
        "Start the requested album; listener feedback does not replace the media request.",
        {"album": "Evening Jazz", "playback_scope": "full_album"}, derived={"playback_status": "playing"},
    ),
    "shared_home_03_grocery_threads": _row(
        "update_personal_grocery_lists",
        "Update owner-scoped grocery lists while preserving explicit exclusions.",
        {"items_by_owner": {"primary": ["milk", "bread"], "live": ["rice"]}, "excluded_items_by_owner": {"live": ["milk"]}},
        derived={"list_status": "updated"},
    ),
    "vehicle_01_driver_destination": _row(
        "set_active_route", "Set the active destination under the driver's route authority.",
        {"destination": "school"}, derived={"route_status": "active"},
    ),
    "vehicle_02_music_side_talk": _row(
        "get_active_route_eta", "Read the active route and ETA without changing navigation or media.",
        {}, effect="read", phase="before_live", result={"destination": "airport", "eta_minutes": 42},
    ),
    "vehicle_03_warning_provenance": _row(
        "get_active_route_eta", "Read route and ETA without claiming access to passenger-reported sensor information.",
        {}, effect="read", phase="before_live", result={"destination": "airport", "eta_minutes": 42},
    ),
    "meeting_01_overlap_votes": _row(
        "record_participant_votes", "Record each participant's vote under their own user ID and compute consensus.",
        {"votes_by_user": {"primary": "support", "live": "oppose"}}, derived={"consensus": False},
    ),
    "meeting_02_assignee_correction": _row(
        "update_task_assignment", "Update the owner and due date of an existing meeting task.",
        {"task_name": "report_edit", "owner": "Cameron", "due_date": "Monday"},
        derived={"assignment_status": "updated"},
    ),
    "meeting_04_quiet_dissenter": _row(
        "record_meeting_vote", "Record one participant's vote and invalidate a premature consensus when positions conflict.",
        {"participant_user_id": "live", "stance": "oppose"},
        derived={"votes": {"primary": "support", "live": "oppose"}, "consensus": False},
    ),
    "meeting_05_result_due_date_update": _row(
        "update_task_due_date", "Update an existing meeting task to the assignee's accepted due date.",
        {"task_name": "audit", "due_date": "Monday"}, derived={"task_status": "updated"},
    ),
    "education_01_fraction_answers": _row(
        "record_student_answers", "Record separate student answers keyed by the actual student speaker.",
        {"answers_by_user": {"primary": "1/2", "live": "2/3"}}, derived={"speaker_records_merged": False},
    ),
    "education_04_project_roles": _row(
        "record_project_role_preferences", "Record accepted and rejected project roles per student without forcing unresolved assignments.",
        {"accepted_roles_by_user": {"live": ["slides"]}, "rejected_roles_by_user": {"live": ["speaking"]}},
        derived={"conflict_status": "unresolved"},
    ),
    "healthcare_care_01_appointment_consent": _row(
        "reschedule_follow_up", "Reschedule a follow-up using the patient's current availability.",
        {"appointment_slot": "afternoon", "preference_owner_user_id": "live"},
        derived={"appointment_status": "rescheduled"},
    ),
    "healthcare_care_02_medication_list_source": _row(
        "record_medication_verification", "Save an external written-list note only under the patient requester’s explicit pending-storage authorization. documented_dose is the dose reported by that external document, not a confirmed patient dose. Preserve source_participant_user_id and source_document. Keep patient_confirmation_required. Never overwrite the existing care-plan medication list or change treatment.",
        {"documented_dose": "10 mg", "source": "written_list", "source_participant_user_id": "live", "source_document": "family_written_list", "confirmation_status": "patient_confirmation_required"},
        derived={"verification_status": "pending"},
    ),
    "hospitality_01_party_size_fix": _row(
        "update_reservation_party_size", "Replace the obsolete reservation party size with the final count.",
        {"party_size": 6}, derived={"reservation_status": "updated"},
    ),
    "hospitality_03_table_side_talk": _row(
        "confirm_guest_order", "Confirm the explicit guest order without converting person-directed side-talk into an item.",
        {"guest_user_id": "primary", "entree": "vegetarian entree"}, derived={"order_status": "confirmed"},
    ),
    "hospitality_04_bill_split": _row(
        "set_itemized_bill_split", "Set an item-level allocation for one payer, replacing an obsolete equal split.",
        {"payer_user_id": "live", "items": ["entree", "tea"], "equal_split": False},
        derived={"split_status": "updated"},
    ),
    "public_services_01_transit_accessibility": _row(
        "update_boarding_assistance", "Update boarding assistance using the rider's accessibility requirement.",
        {"resident_user_id": "live", "entrance": "lift_equipped"}, derived={"assistance_status": "arranged"},
    ),
    "public_services_02_appointment_interpreter": _row(
        "schedule_public_service_appointment", "Schedule an appointment while preserving requester and interpreter roles.",
        {"requester_user_id": "primary", "slot": "Tuesday morning", "relay_user_id": "live"},
        derived={"appointment_status": "scheduled"},
    ),
    "it_support_01_service_owner_incident": _row(
        "defer_service_restart", "Defer a production restart until the service owner's safety condition is satisfied.",
        {"until_condition": "active_transaction_finishes"}, derived={"restart_status": "deferred"},
    ),
    "it_support_03_rollback_correction": _row(
        "rollback_service_version", "Roll back the authorized service to the specified version.",
        {"service_name": "alpha", "target_version": "4.1"},
        derived={"current_version": "4.1", "rollback_status": "completed"}, reversible=False,
    ),
    "events_entertainment_01_ticket_ownership": _row(
        "record_ticket_retention", "Record that an attendee retains their ticket and prevent reassignment without consent.",
        {"attendee_user_id": "live", "retained": True}, derived={"late_attendee_reassignment": "blocked"},
    ),
    "events_entertainment_02_playlist_cheer": _row(
        "start_event_playlist", "Start the requested event playlist; cheering is feedback, not a replacement command.",
        {"playlist_scope": "complete_set"}, derived={"playback_status": "playing"},
    ),
    "events_entertainment_03_accessible_seat": _row(
        "update_attendee_seat_requirement", "Prioritize one attendee's accessibility seating requirement over group proximity.",
        {"attendee_user_id": "live", "seat_requirement": "wheelchair_accessible_row", "proximity_priority": "secondary"},
        derived={"seat_status": "updated"},
    ),
    "events_entertainment_05_schedule_time_fix": _row(
        "update_event_schedule", "Update and distinguish doors time from show start time.",
        {"doors_time": "18:00", "show_time": "19:00"}, derived={"schedule_status": "corrected"},
    ),
    "family_coordination_01_pickup_assignments": _row(
        "record_school_pickup_assignments", "Record child-specific accepted pickups and leave unaccepted duties unresolved.",
        {"assignments_by_child": {"Maya": "live"}, "unresolved_children": ["Leo"]},
        derived={"assignment_status": "partially_resolved"},
    ),
    "family_coordination_03_chore_quiet_child": _row(
        "record_existing_chore_constraint", "Record an affected family member's existing workload before final allocation.",
        {"family_member_user_id": "live", "existing_chore": "dishes_daily"},
        derived={"allocation_status": "reconsider_required"},
    ),
    "emergency_incident_02_lost_child_description": _row(
        "update_lost_child_description", "Update an observable description while preserving the correction source.",
        {"jacket_color": "green", "correction_source_user_id": "live"}, derived={"description_status": "corrected"},
    ),
    "emergency_incident_03_roll_call_overlap": _row(
        "record_floor_roll_calls", "Store roll-call reports per representative and retain unresolved conflicts.",
        {"reports_by_user": {"primary": "all_accounted_for", "live": "one_unaccounted_for"}},
        derived={"overall_status": "unresolved"},
    ),
}


def _scenario_domain(scenario_id: str) -> str:
    matches = [domain for domain in DOMAIN_BUSINESS_CONTRACTS if scenario_id.startswith(domain + "_")]
    if len(matches) != 1:
        raise ValueError(f"cannot resolve domain for scenario: {scenario_id}")
    return matches[0]


SCENARIO_BUSINESS_SEMANTICS: dict[str, ScenarioBusinessSemantics] = {}
for _scenario_id, _data in OPERATION_ROWS.items():
    _legacy = legacy_semantics_for_scenario(_scenario_id)
    if _legacy is None:
        raise ValueError(f"missing initial fixture semantics: {_scenario_id}")
    _domain_name = _scenario_domain(_scenario_id)
    _arguments = deepcopy(_data["arguments"])
    _legacy_derived = {
        key: deepcopy(value)
        for key, value in _legacy.resolution_payload.items()
        if key not in _arguments
    }
    _operation = BusinessOperationSpec(
        scenario_id=_scenario_id,
        domain=_domain_name,
        tool_name=_data["tool_name"],
        description=_data["description"],
        expected_arguments=_arguments,
        effect=_data["effect"],
        execution_phase=_data["phase"],
        derived_state={**_legacy_derived, **deepcopy(_data["derived"])},
        read_result=deepcopy(_data["result"]),
        reversible=_data["reversible"],
    )
    SCENARIO_BUSINESS_SEMANTICS[_scenario_id] = ScenarioBusinessSemantics(
        initial_context=deepcopy(_legacy.initial_context),
        operation=_operation,
    )


BUSINESS_OPERATIONS_BY_DOMAIN: dict[str, dict[str, BusinessOperationSpec]] = {}
for _semantics in SCENARIO_BUSINESS_SEMANTICS.values():
    _operation = _semantics.operation
    existing = BUSINESS_OPERATIONS_BY_DOMAIN.setdefault(_operation.domain, {}).get(
        _operation.tool_name
    )
    if existing is not None and existing.expected_arguments != _operation.expected_arguments:
        raise ValueError(f"incompatible stable tool definitions: {_operation.tool_name}")
    BUSINESS_OPERATIONS_BY_DOMAIN[_operation.domain][_operation.tool_name] = _operation


def contract_for_domain(domain: str) -> DomainBusinessContract | None:
    return DOMAIN_BUSINESS_CONTRACTS.get(domain.removeprefix("v07_"))


def semantics_for_scenario(scenario_id: str) -> ScenarioBusinessSemantics | None:
    return SCENARIO_BUSINESS_SEMANTICS.get(scenario_id)


def operation_for_scenario(scenario_id: str) -> BusinessOperationSpec | None:
    semantics = semantics_for_scenario(scenario_id)
    return semantics.operation if semantics is not None else None


def domain_tool_names(domain: str) -> set[str]:
    contract = contract_for_domain(domain)
    if contract is None:
        return set()
    return {contract.inspect_tool, *BUSINESS_OPERATIONS_BY_DOMAIN.get(contract.domain, {})}


def resource_id_for_scenario(domain: str, scenario_id: str) -> str:
    contract = contract_for_domain(domain)
    if contract is None:
        raise ValueError(f"domain has no business resource: {domain}")
    remainder = scenario_id.removeprefix(f"{contract.domain}_")
    index = int(remainder.split("_", 1)[0])
    return f"{contract.id_prefix}-MU-{index:03d}"


def initial_state_for_scenario(
    domain: str,
    scenario_id: str,
    participant_ids: list[str],
) -> tuple[str, dict[str, Any]]:
    contract = contract_for_domain(domain)
    semantics = semantics_for_scenario(scenario_id)
    if contract is None or semantics is None:
        raise ValueError(f"missing business fixture: {scenario_id}")
    resource_id = resource_id_for_scenario(domain, scenario_id)
    resource = BusinessResourceRecord.model_validate({
        contract.resource_key: resource_id,
        "resource_type": contract.resource_noun,
        "participant_user_ids": list(participant_ids),
        "status": "open",
        "business_state": deepcopy(semantics.initial_context),
        "last_operation": None,
        "last_modified_by": None,
        "last_transaction_id": None,
    }).model_dump()
    return resource_id, {
        contract.collection_key: {resource_id: resource},
        "audit_events": [],
    }


def _annotation_for(value: Any) -> Any:
    if isinstance(value, bool):
        return bool
    if isinstance(value, int) and not isinstance(value, bool):
        return int
    if isinstance(value, float):
        return float
    if isinstance(value, list):
        return list[str] if all(isinstance(item, str) for item in value) else list[Any]
    if isinstance(value, dict):
        return dict[str, Any]
    return str


def _json_schema(value: Any, description: str) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean", "description": description}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer", "description": description}
    if isinstance(value, float):
        return {"type": "number", "description": description}
    if isinstance(value, list):
        item_type = "string" if all(isinstance(item, str) for item in value) else "object"
        return {"type": "array", "items": {"type": item_type}, "description": description}
    if isinstance(value, dict):
        return {"type": "object", "description": description}
    return {"type": "string", "description": description}


def _definition(
    contract: DomainBusinessContract,
    operation: BusinessOperationSpec | None,
) -> VirtualToolDefinition:
    if operation is None:
        name = contract.inspect_tool
        description = f"Read the current {contract.resource_noun} and business state without changing it."
        effect = "read"
        argument_examples: dict[str, Any] = {}
        reversible = True
        errors = ["RESOURCE_NOT_FOUND: the requested record does not exist."]
        idempotency = "Pure read; repeated calls never change benchmark state."
    else:
        name = operation.tool_name
        description = operation.description
        effect = operation.effect
        argument_examples = operation.expected_arguments
        reversible = operation.reversible
        errors = list(operation.errors)
        idempotency = operation.idempotency
    properties = {
        contract.resource_key: {
            "type": "string",
            "description": f"Stable identifier of the {contract.resource_noun}.",
        },
        **{
            key: _json_schema(value, key.replace("_", " ").capitalize() + ".")
            for key, value in argument_examples.items()
        },
    }
    return VirtualToolDefinition(
        name=name,
        description=description,
        effect=effect,
        reversible=reversible,
        call_method=f"POST /v1/{contract.domain}/{name}",
        parameters={
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        returns={
            "type": "object",
            "properties": {
                contract.resource_key: {"type": "string"},
                "transaction_id": {"type": ["string", "null"]},
                "business_state": {"type": "object"},
            },
        },
        errors=errors,
        idempotency=idempotency,
        result_lookup=contract.inspect_tool,
        resource_key=contract.resource_key,
        requester_source="runner_bound_speaker_context",
        mutates_state=effect == "write",
    )


class BusinessDomainEnvironment(BaseVirtualEnvironment):
    """Stable domain API backed by an auditable typed fixture."""

    def __init__(self, binding: EnvironmentBinding) -> None:
        contract = contract_for_domain(binding.domain)
        if contract is None:
            raise ValueError(f"v0.7 domain has no business contract: {binding.domain}")
        self.binding = binding
        self.contract = contract
        self.operations = BUSINESS_OPERATIONS_BY_DOMAIN.get(contract.domain, {})
        self.definitions = {
            contract.inspect_tool: _definition(contract, None),
            **{
                name: _definition(contract, operation)
                for name, operation in self.operations.items()
            },
        }
        from experiments.tau_multiuser_voice.interface_contract import expose_schema
        for name, definition in self.definitions.items():
            operation = self.operations.get(name)
            if operation is not None:
                expose_schema(definition.parameters, operation.scenario_id)
        from experiments.tau_multiuser_voice.completion_tools import definitions as completion_definitions
        self.completion_definitions = completion_definitions(binding)
        self.definitions.update(self.completion_definitions)
        super().__init__(binding.initial_state or None)

    @classmethod
    def default_state(cls) -> dict[str, Any]:
        return {"audit_events": []}

    def _resource(self, resource_id: str) -> dict[str, Any]:
        collection = self.state.get(self.contract.collection_key, {})
        resource = collection.get(resource_id) if isinstance(collection, dict) else None
        if not isinstance(resource, dict):
            raise ValueError(f"RESOURCE_NOT_FOUND: {self.contract.resource_noun} {resource_id}")
        BusinessResourceRecord.model_validate(resource)
        return resource

    def invoke(self, tool_name: str, actor_id: str, **arguments: Any) -> Any:
        if tool_name in self.completion_definitions:
            from experiments.tau_multiuser_voice.completion_tools import execute
            def handler(*, actor_id: str, **kwargs: Any) -> dict[str, Any]:
                return execute(self, tool_name, actor_id, kwargs)
            setattr(self, f"_tool_{tool_name}", handler)
            return super().invoke(tool_name, actor_id, **arguments)
        if tool_name == self.contract.inspect_tool:
            def handler(*, actor_id: str, **kwargs: Any) -> dict[str, Any]:
                return self._inspect(actor_id=actor_id, arguments=kwargs)
        elif tool_name in self.operations:
            def handler(*, actor_id: str, **kwargs: Any) -> dict[str, Any]:
                return self._operate(
                    operation=self.operations[tool_name],
                    actor_id=actor_id,
                    arguments=kwargs,
                )
        else:
            return super().invoke(tool_name, actor_id, **arguments)
        setattr(self, f"_tool_{tool_name}", handler)
        return super().invoke(tool_name, actor_id, **arguments)

    def _inspect(self, *, actor_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if set(arguments) != {self.contract.resource_key}:
            raise ValueError(f"INVALID_ARGUMENT: expected only {self.contract.resource_key}")
        resource_id = str(arguments[self.contract.resource_key])
        return {**deepcopy(self._resource(resource_id)), "requested_by": actor_id}

    def _operate(
        self,
        *,
        operation: BusinessOperationSpec,
        actor_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        from experiments.tau_multiuser_voice.interface_contract import normalize_fields
        arguments = normalize_fields(operation.scenario_id, arguments)
        expected_keys = {self.contract.resource_key, *operation.expected_arguments}
        if set(arguments) != expected_keys:
            raise ValueError(f"INVALID_ARGUMENT: expected fields {sorted(expected_keys)}")
        resource_id = str(arguments.pop(self.contract.resource_key))
        resource = self._resource(resource_id)
        from .business_values import enabled as value_contract_enabled, resolve_public_child_names
        if operation.scenario_id == "family_coordination_01_pickup_assignments" and value_contract_enabled():
            arguments = resolve_public_child_names(arguments, resource["business_state"].get("children", []))
        for name, example in operation.expected_arguments.items():
            if not isinstance(arguments[name], type(example)):
                raise ValueError(f"INVALID_ARGUMENT: wrong type for {name}")

        from .interface_contract import expose_schema, validate_shape
        schema=deepcopy(self.definitions[operation.tool_name].parameters)
        expose_schema(schema,operation.scenario_id)
        for field,value in arguments.items():
            validate_shape(value,schema['properties'].get(field,{}),field)
            if isinstance(value,str) and not value.strip():raise ValueError('INVALID_ARGUMENT: empty '+field)
            from .business_participant_identifiers import field_kind
            kind=field_kind(operation.tool_name,field)
            ids=([value] if kind=='value' else value if kind=='array_items' else list(value) if kind=='object_keys' else [])
            if any(uid not in resource['participant_user_ids'] for uid in ids):raise ValueError('INVALID_ARGUMENT: unknown participant in '+field)

        if operation.scenario_id == "telecom_02_lost_phone_number_fix":
            known = resource["business_state"].get("available_line_suffixes", ["0182", "0812"])
            lost, keep = arguments["lost_line_suffix"], arguments["preserve_active_line_suffix"]
            if lost not in known or keep not in known or lost == keep:
                raise ValueError("INVALID_ARGUMENT: line suffixes must identify two distinct existing account lines")

        if operation.scenario_id == "airline_01_split_rebooking":
            available = resource["business_state"].get("available_alternative_flights", [])
            if arguments["target_flight_id"] not in {f["flight_id"] for f in available}:
                raise ValueError("INVALID_ARGUMENT: target flight is not in the available booking alternatives")

        if operation.scenario_id == 'public_services_02_appointment_interpreter':
            slots = resource['business_state'].get('available_appointment_slots')
            if slots is not None and arguments['slot'] not in slots:
                raise ValueError('INVALID_ARGUMENT: appointment slot is not available')

        if operation.effect == "read":
            result = {
                self.contract.resource_key: resource_id,
                "requested_by": actor_id,
                "request": deepcopy(arguments),
                "result": __import__("experiments.tau_multiuser_voice.contract_revision", fromlist=["read_result"]).read_result(operation.scenario_id, arguments, resource["business_state"], operation.read_result),
                "business_state": deepcopy(resource["business_state"]),
                "transaction_id": None,
            }
            BusinessToolResult.model_validate(result)
            return result

        from .medication_source_contract import note_from_arguments
        audit_events = self.state.setdefault("audit_events", [])
        for event in audit_events:
            if (
                event["tool_name"] == operation.tool_name
                and event["resource_id"] == resource_id
                and event["actor_id"] == actor_id
                and event["arguments"] == arguments
                and (
                    (note_from_arguments(arguments) in resource['business_state'].get('external_medication_notes',[]))
                    if operation.scenario_id == 'healthcare_care_02_medication_list_source'
                    else all(resource["business_state"].get(k) == v for k,v in arguments.items())
                )
            ):
                return deepcopy(event["result"])

        from .business_effects import merge_arguments, derive
        # External-note prose must not overwrite the historical list's source.
        merged_arguments = {k:v for k,v in arguments.items() if k!='source'} if operation.scenario_id == 'healthcare_care_02_medication_list_source' else arguments
        merge_arguments(resource['business_state'], merged_arguments)
        derived = derive(operation, arguments, resource['business_state'], actor_id)
        resource["business_state"].update(derived)
        if operation.scenario_id == 'events_entertainment_01_ticket_ownership':
            tickets = resource['business_state'].get('tickets', [])
            resource['business_state']['ticket_assignments'] = {
                t['ticket_id']: (t['owner'] if t['owner']=='Blake' and arguments['attendee_user_id']=='live' and arguments['retained'] else t['requested_assignee'])
                for t in tickets
            }
        resource["status"] = "updated"
        resource["last_operation"] = operation.tool_name
        resource["last_modified_by"] = actor_id
        transaction_id = f"TX-{self.contract.id_prefix}-{len(audit_events) + 1:04d}"
        resource["last_transaction_id"] = transaction_id
        BusinessResourceRecord.model_validate(resource)
        result = {
            self.contract.resource_key: resource_id,
            "transaction_id": transaction_id,
            "status": "updated",
            "business_state": deepcopy(resource["business_state"]),
        }
        BusinessToolResult.model_validate(result)
        audit_events.append(
            BusinessAuditEvent.model_validate({
                "transaction_id": transaction_id,
                "tool_name": operation.tool_name,
                "resource_id": resource_id,
                "actor_id": actor_id,
                "arguments": deepcopy(arguments),
                "result": deepcopy(result),
            }).model_dump()
        )
        return result


def make_business_tool_callable(
    environment: BusinessDomainEnvironment,
    tool_name: str,
) -> Callable[..., dict[str, Any]]:
    """Build a Realtime function schema that excludes requester identity."""
    if tool_name not in environment.definitions:
        raise ValueError(f"not a tool in {environment.contract.domain}: {tool_name}")
    operation = environment.operations.get(tool_name)

    def invoke(**arguments: Any) -> dict[str, Any]:
        # The production runner intercepts this call and supplies the trusted
        # actor. ``runner_context`` is only a direct-call fallback for tests.
        return environment.invoke(tool_name, "runner_context", **arguments)

    invoke.__name__ = tool_name
    definition = environment.definitions[tool_name]
    properties = definition.parameters.get("properties", {})
    argument_lines = [
        f"    {name}: {schema.get('description', 'Required business parameter.')}"
        for name, schema in properties.items()
    ]
    error_lines = [f"    {error}" for error in definition.errors]
    invoke.__doc__ = "\n".join(
        [
            definition.description,
            "",
            f"Endpoint: {definition.call_method}",
            f"Requester identity: {definition.requester_source}.",
            "",
            "Args:",
            *argument_lines,
            "",
            "Returns:",
            (
                "    " + definition.returns.get("description", "Actual owner-scoped operation result.")
                if tool_name in environment.completion_definitions else
                "    A structured result containing the resource ID, transaction_id "
                "(null for reads), and the current business_state."
            ),
            "",
            "Raises:",
            *error_lines,
            "",
            f"Idempotency: {definition.idempotency}",
            f"Result lookup: call {definition.result_lookup} with the same resource ID.",
        ]
    )
    parameters = [
        Parameter(
            environment.contract.resource_key,
            Parameter.POSITIONAL_OR_KEYWORD,
            annotation=str,
        )
    ]
    if operation is not None:
        parameters.extend(
            Parameter(name, Parameter.POSITIONAL_OR_KEYWORD, annotation=_annotation_for(value))
            for name, value in operation.expected_arguments.items()
        )
    if definition.requester_source == "model_declared_caller":
        parameters.append(Parameter("caller_id", Parameter.POSITIONAL_OR_KEYWORD, annotation=str))
    if tool_name in environment.completion_definitions:
        from experiments.tau_multiuser_voice.completion_tools import signature
        invoke.__signature__ = signature(definition)
    else:
        invoke.__signature__ = Signature(parameters, return_annotation=dict[str, Any])
    return invoke
