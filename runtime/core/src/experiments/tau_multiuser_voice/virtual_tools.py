"""Deterministic non-business environments for τ-MU Bench v0.2."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from experiments.tau_multiuser_voice.models import EnvironmentBinding


class VirtualToolDefinition(BaseModel):
    """Model-visible contract plus execution metadata for one business tool.

    ``requester_source`` deliberately keeps identity outside business arguments.
    The Realtime model reports speaker attribution separately; the runner binds
    that report to the oracle audio event and supplies ``actor_id`` only to the
    environment implementation.
    """

    name: str
    description: str
    effect: str
    reversible: bool = True
    call_method: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    returns: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    idempotency: str | None = None
    result_lookup: str | None = None
    resource_key: str | None = None
    requester_source: str = "model_argument"
    mutates_state: bool | None = None

    def model_post_init(self, __context: Any) -> None:
        """Infer replay behavior when older definitions omit it."""
        if self.mutates_state is None:
            self.mutates_state = self.effect == "write"


class VirtualToolCallRecord(BaseModel):
    """Auditable invocation in a deterministic environment."""

    tool_name: str
    actor_id: str
    arguments: dict[str, Any]
    success: bool
    result: Any = None
    error: str | None = None
    state_changed: bool = False


class BaseVirtualEnvironment:
    """Small state machine used for reproducible non-business tool evaluation."""

    definitions: ClassVar[dict[str, VirtualToolDefinition]] = {}

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        self.state = deepcopy(initial_state or self.default_state())
        self.call_log: list[VirtualToolCallRecord] = []

    @classmethod
    def default_state(cls) -> dict[str, Any]:
        return {}

    @property
    def tool_names(self) -> set[str]:
        return set(self.definitions)

    def invoke(self, tool_name: str, actor_id: str, **arguments: Any) -> Any:
        """Execute a tool and retain actor attribution even on failure."""
        if tool_name not in self.definitions:
            raise ValueError(f"unknown virtual tool: {tool_name}")
        handler = getattr(self, f"_tool_{tool_name}")
        before = self.snapshot()
        try:
            result = handler(actor_id=actor_id, **arguments)
        except (KeyError, TypeError, ValueError) as error:
            self.call_log.append(
                VirtualToolCallRecord(
                    tool_name=tool_name,
                    actor_id=actor_id,
                    arguments=arguments,
                    success=False,
                    error=str(error),
                )
            )
            raise
        self.call_log.append(
            VirtualToolCallRecord(
                tool_name=tool_name,
                actor_id=actor_id,
                arguments=arguments,
                success=True,
                result=deepcopy(result),
                state_changed=before != self.state,
            )
        )
        return result

    def snapshot(self) -> dict[str, Any]:
        """Return an isolated state snapshot for final-state evaluation."""
        return deepcopy(self.state)


class SharedHomeEnvironment(BaseVirtualEnvironment):
    """Timers and shared media without external device side effects."""

    definitions: ClassVar[dict[str, VirtualToolDefinition]] = {
        "set_timer": VirtualToolDefinition(
            name="set_timer",
            description="Create or replace a named household timer.",
            effect="write",
        ),
        "cancel_timer": VirtualToolDefinition(
            name="cancel_timer",
            description="Cancel one named timer.",
            effect="write",
        ),
        "get_timer": VirtualToolDefinition(
            name="get_timer",
            description="Read a named timer.",
            effect="read",
        ),
        "set_media": VirtualToolDefinition(
            name="set_media",
            description="Change shared media playback.",
            effect="write",
        ),
    }

    @classmethod
    def default_state(cls) -> dict[str, Any]:
        return {"timers": {}, "media": {"content": None, "playing": False}}

    def _tool_set_timer(
        self, *, actor_id: str, label: str, duration_seconds: int
    ) -> dict[str, Any]:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        timer = {
            "label": label,
            "duration_seconds": duration_seconds,
            "created_by": actor_id,
            "active": True,
        }
        self.state.setdefault("timers", {})[label] = timer
        return timer

    def _tool_cancel_timer(self, *, actor_id: str, label: str) -> dict[str, Any]:
        del actor_id
        timer = self.state.setdefault("timers", {}).get(label)
        if timer is None:
            raise ValueError(f"timer not found: {label}")
        timer["active"] = False
        return timer

    def _tool_get_timer(self, *, actor_id: str, label: str) -> dict[str, Any]:
        del actor_id
        timer = self.state.setdefault("timers", {}).get(label)
        if timer is None:
            raise ValueError(f"timer not found: {label}")
        return timer

    def _tool_set_media(
        self, *, actor_id: str, content: str, playing: bool = True
    ) -> dict[str, Any]:
        media = {"content": content, "playing": playing, "changed_by": actor_id}
        self.state["media"] = media
        return media


class VehicleEnvironment(BaseVirtualEnvironment):
    """Sandbox navigation and media state for an in-car shared assistant."""

    definitions: ClassVar[dict[str, VirtualToolDefinition]] = {
        "preview_route": VirtualToolDefinition(
            name="preview_route",
            description="Preview a route without changing active navigation.",
            effect="read",
        ),
        "commit_route": VirtualToolDefinition(
            name="commit_route",
            description="Set the active navigation destination.",
            effect="write",
        ),
        "get_eta": VirtualToolDefinition(
            name="get_eta",
            description="Read the active route ETA.",
            effect="read",
        ),
        "set_media": VirtualToolDefinition(
            name="set_media",
            description="Set in-car media without changing navigation.",
            effect="write",
        ),
    }

    @classmethod
    def default_state(cls) -> dict[str, Any]:
        return {
            "navigation": {
                "destination": None,
                "eta_minutes": None,
                "changed_by": None,
            },
            "media": {"content": None, "playing": False},
            "route_catalog": {
                "airport": 42,
                "city center": 18,
                "school": 12,
            },
        }

    def _tool_preview_route(self, *, actor_id: str, destination: str) -> dict[str, Any]:
        eta = self.state.get("route_catalog", {}).get(destination)
        if eta is None:
            raise ValueError(f"unknown destination: {destination}")
        return {"destination": destination, "eta_minutes": eta, "actor": actor_id}

    def _tool_commit_route(self, *, actor_id: str, destination: str) -> dict[str, Any]:
        preview = self._tool_preview_route(actor_id=actor_id, destination=destination)
        self.state["navigation"] = {
            "destination": destination,
            "eta_minutes": preview["eta_minutes"],
            "changed_by": actor_id,
        }
        return self.state["navigation"]

    def _tool_get_eta(self, *, actor_id: str) -> dict[str, Any]:
        del actor_id
        navigation = self.state["navigation"]
        if navigation.get("destination") is None:
            raise ValueError("no active route")
        return navigation

    def _tool_set_media(self, *, actor_id: str, content: str) -> dict[str, Any]:
        self.state["media"] = {
            "content": content,
            "playing": True,
            "changed_by": actor_id,
        }
        return self.state["media"]


class MeetingEnvironment(BaseVirtualEnvironment):
    """Shared proposals, votes, assignments, and drafts for group work."""

    definitions: ClassVar[dict[str, VirtualToolDefinition]] = {
        "add_proposal": VirtualToolDefinition(
            name="add_proposal",
            description="Record a proposal and its original speaker.",
            effect="write",
        ),
        "record_vote": VirtualToolDefinition(
            name="record_vote",
            description="Record one participant stance on a proposal.",
            effect="write",
        ),
        "assign_task": VirtualToolDefinition(
            name="assign_task",
            description="Assign a task with owner and due date.",
            effect="write",
        ),
        "mark_unresolved_issue": VirtualToolDefinition(
            name="mark_unresolved_issue",
            description="Preserve an unresolved disagreement.",
            effect="write",
        ),
        "draft_message": VirtualToolDefinition(
            name="draft_message",
            description="Create a reversible message draft; never sends it.",
            effect="write",
        ),
    }

    @classmethod
    def default_state(cls) -> dict[str, Any]:
        return {
            "proposals": {},
            "votes": {},
            "assignments": {},
            "unresolved_issues": [],
            "drafts": [],
        }

    def _tool_add_proposal(
        self, *, actor_id: str, proposal_id: str, text: str
    ) -> dict[str, Any]:
        proposal = {"text": text, "proposed_by": actor_id}
        self.state["proposals"][proposal_id] = proposal
        return proposal

    def _tool_record_vote(
        self, *, actor_id: str, proposal_id: str, stance: str
    ) -> dict[str, Any]:
        if stance not in {"support", "oppose", "abstain"}:
            raise ValueError("invalid stance")
        votes = self.state["votes"].setdefault(proposal_id, {})
        votes[actor_id] = stance
        return {"proposal_id": proposal_id, "speaker": actor_id, "stance": stance}

    def _tool_assign_task(
        self, *, actor_id: str, task_id: str, owner: str, due_date: str
    ) -> dict[str, Any]:
        assignment = {
            "owner": owner,
            "due_date": due_date,
            "recorded_by": actor_id,
        }
        self.state["assignments"][task_id] = assignment
        return assignment

    def _tool_mark_unresolved_issue(
        self, *, actor_id: str, issue: str, speakers: list[str]
    ) -> dict[str, Any]:
        unresolved = {
            "issue": issue,
            "speakers": speakers,
            "recorded_by": actor_id,
        }
        self.state["unresolved_issues"].append(unresolved)
        return unresolved

    def _tool_draft_message(
        self, *, actor_id: str, recipient: str, text: str
    ) -> dict[str, Any]:
        draft = {"recipient": recipient, "text": text, "requested_by": actor_id}
        self.state["drafts"].append(draft)
        return draft


class EducationEnvironment(BaseVirtualEnvironment):
    """Shared whiteboard, answer provenance, and speaking queue."""

    definitions: ClassVar[dict[str, VirtualToolDefinition]] = {
        "record_student_answer": VirtualToolDefinition(
            name="record_student_answer",
            description="Record an answer under the actual student speaker.",
            effect="write",
        ),
        "update_whiteboard": VirtualToolDefinition(
            name="update_whiteboard",
            description="Add a labeled claim to the shared whiteboard.",
            effect="write",
        ),
        "assign_speaking_turn": VirtualToolDefinition(
            name="assign_speaking_turn",
            description="Choose the next student without erasing the queue.",
            effect="write",
        ),
    }

    @classmethod
    def default_state(cls) -> dict[str, Any]:
        return {"answers": {}, "whiteboard": [], "speaking_queue": []}

    def _tool_record_student_answer(
        self, *, actor_id: str, question_id: str, answer: str
    ) -> dict[str, Any]:
        answer_record = {"answer": answer, "speaker": actor_id}
        self.state["answers"].setdefault(question_id, []).append(answer_record)
        return answer_record

    def _tool_update_whiteboard(
        self, *, actor_id: str, claim: str, label: str
    ) -> dict[str, Any]:
        entry = {"claim": claim, "label": label, "source": actor_id}
        self.state["whiteboard"].append(entry)
        return entry

    def _tool_assign_speaking_turn(
        self, *, actor_id: str, next_speaker: str
    ) -> dict[str, Any]:
        queue = self.state.setdefault("speaking_queue", [])
        if next_speaker in queue:
            queue.remove(next_speaker)
        queue.insert(0, next_speaker)
        return {"next_speaker": next_speaker, "assigned_by": actor_id}


class KnowledgeEnvironment(BaseVirtualEnvironment):
    """Tool-free state holder for conversational knowledge and reasoning cases."""

    definitions: ClassVar[dict[str, VirtualToolDefinition]] = {}

    @classmethod
    def default_state(cls) -> dict[str, Any]:
        return {"facts": {}, "notes": []}


class FactorialScenarioEnvironment(BaseVirtualEnvironment):
    """Deterministic action ledger shared by the 20-domain v0.7 grid."""

    definitions: ClassVar[dict[str, VirtualToolDefinition]] = {
        "inspect_case_record": VirtualToolDefinition(
            name="inspect_case_record",
            description="Inspect the active benchmark record before resolving it.",
            effect="read",
        ),
        "commit_case_resolution": VirtualToolDefinition(
            name="commit_case_resolution",
            description="Commit the active speaker-scoped workflow resolution.",
            effect="write",
        ),
        "execute_case_action": VirtualToolDefinition(
            name="execute_case_action",
            description="Execute the active case action under the attributed actor.",
            effect="write",
        ),
        "record_case_decision": VirtualToolDefinition(
            name="record_case_decision",
            description="Record the active multi-user decision and its actor.",
            effect="write",
        ),
    }

    @classmethod
    def default_state(cls) -> dict[str, Any]:
        return {
            "records": {
                "active_case": {"status": "pending", "resolved_by": None}
            },
            "actions": {
                "active_action": {"status": "pending", "executed_by": None}
            },
            "decisions": {
                "active_decision": {"status": "pending", "recorded_by": None}
            },
        }

    def _tool_inspect_case_record(
        self, *, actor_id: str, record_id: str = "active_case"
    ) -> dict[str, Any]:
        record = self.state["records"].get(record_id)
        if record is None:
            raise ValueError(f"record not found: {record_id}")
        return {**record, "inspected_by": actor_id}

    def _tool_commit_case_resolution(
        self, *, actor_id: str, record_id: str = "active_case"
    ) -> dict[str, Any]:
        record = self.state["records"].get(record_id)
        if record is None:
            raise ValueError(f"record not found: {record_id}")
        record.update(status="resolved", resolved_by=actor_id)
        return record

    def _tool_execute_case_action(
        self, *, actor_id: str, action_id: str = "active_action"
    ) -> dict[str, Any]:
        action = self.state["actions"].get(action_id)
        if action is None:
            raise ValueError(f"action not found: {action_id}")
        action.update(status="executed", executed_by=actor_id)
        return action

    def _tool_record_case_decision(
        self, *, actor_id: str, decision_id: str = "active_decision"
    ) -> dict[str, Any]:
        decision = self.state["decisions"].get(decision_id)
        if decision is None:
            raise ValueError(f"decision not found: {decision_id}")
        decision.update(status="recorded", recorded_by=actor_id)
        return decision


class RetailAuditEnvironment(BaseVirtualEnvironment):
    """Case-scoped retail sandbox for identity and authorization evaluation."""

    definitions: ClassVar[dict[str, VirtualToolDefinition]] = {
        "get_order_details": VirtualToolDefinition(
            name="get_order_details",
            description="Read one benchmark retail order.",
            effect="read",
        ),
        "get_product_details": VirtualToolDefinition(
            name="get_product_details",
            description="Read public benchmark product information.",
            effect="read",
        ),
        "return_delivered_order_items": VirtualToolDefinition(
            name="return_delivered_order_items",
            description="Request a return for selected delivered-order items.",
            effect="write",
        ),
        "exchange_delivered_order_items": VirtualToolDefinition(
            name="exchange_delivered_order_items",
            description="Request an exchange for selected delivered-order items.",
            effect="write",
        ),
        "cancel_pending_order": VirtualToolDefinition(
            name="cancel_pending_order",
            description="Cancel a pending benchmark order.",
            effect="write",
        ),
        "modify_pending_order_address": VirtualToolDefinition(
            name="modify_pending_order_address",
            description="Record an address-change request for a pending order.",
            effect="write",
        ),
    }

    @classmethod
    def default_state(cls) -> dict[str, Any]:
        orders = {
            "#W4817420": {
                "status": "delivered",
                "item_ids": [
                    "6777246137",
                    "4900661478",
                    "6700049080",
                    "9624127908",
                    "3812493782",
                ],
                "payment_method_id": "gift_card_8168843",
                "address": "account address",
            },
            "#W6304490": {
                "status": "delivered",
                "item_ids": [
                    "6956751343",
                    "4983901480",
                    "9375701158",
                    "2194493783",
                    "5753502325",
                ],
                "payment_method_id": "credit_card_4420174",
                "address": "account address",
            },
            "#W5918442": {
                "status": "pending",
                "item_ids": [
                    "1725100896",
                    "5312063289",
                    "1586641416",
                    "6117189161",
                ],
                "payment_method_id": "credit_card_5051208",
                "address": "account address",
            },
            "#W2631563": {
                "status": "pending",
                "item_ids": ["2791467853", "5753502325"],
                "payment_method_id": "paypal_7160322",
                "address": "account address",
            },
            "#W6779827": {
                "status": "pending",
                "item_ids": [
                    "7896397433",
                    "3379843752",
                    "1323134954",
                    "6942241102",
                ],
                "payment_method_id": "gift_card_7219486",
                "address": "account address",
            },
            "#W9077205": {
                "status": "delivered",
                "item_ids": ["9370300555", "3877338112"],
                "payment_method_id": "gift_card_7108145",
                "address": "account address",
            },
            "#W2974929": {
                "status": "pending",
                "item_ids": ["3877188862"],
                "payment_method_id": "paypal_7916550",
                "address": "account address",
            },
            "#W8032761": {
                "status": "delivered",
                "item_ids": ["8030558068", "8484921793"],
                "payment_method_id": "gift_card_2977513",
                "address": "account address",
            },
            "#W7678072": {
                "status": "delivered",
                "item_ids": ["2323972008", "2193628750", "3557711149"],
                "payment_method_id": "paypal_5727330",
                "address": "account address",
            },
        }
        return {
            "orders": orders,
            "products": {
                "4896585277": {
                    "name": "smart thermostat",
                    "compatible_platforms": ["Apple HomeKit"],
                }
            },
            "audit_events": [],
        }

    def _order(self, order_id: str) -> dict[str, Any]:
        order = self.state["orders"].get(order_id)
        if order is None:
            raise ValueError(f"order not found: {order_id}")
        return order

    def _audit(
        self, *, action: str, actor_id: str, resource_id: str, details: dict[str, Any]
    ) -> dict[str, Any]:
        event = {
            "action": action,
            "actor_id": actor_id,
            "resource_id": resource_id,
            **details,
        }
        self.state["audit_events"].append(event)
        return event

    def _tool_get_order_details(
        self, *, actor_id: str, order_id: str
    ) -> dict[str, Any]:
        order = deepcopy(self._order(order_id))
        return {"order_id": order_id, "requested_by": actor_id, **order}

    def _tool_get_product_details(
        self, *, actor_id: str, product_id: str
    ) -> dict[str, Any]:
        product = self.state["products"].get(product_id)
        if product is None:
            raise ValueError(f"product not found: {product_id}")
        return {"product_id": product_id, "requested_by": actor_id, **product}

    def _tool_return_delivered_order_items(
        self,
        *,
        actor_id: str,
        order_id: str,
        item_ids: list[str],
        refund_destination: str = "original_payment_method",
    ) -> dict[str, Any]:
        order = self._order(order_id)
        if order["status"] != "delivered":
            raise ValueError("order is not delivered")
        if not set(item_ids).issubset(order["item_ids"]):
            raise ValueError("item does not belong to order")
        order["status"] = "return_requested"
        return self._audit(
            action="return_delivered_order_items",
            actor_id=actor_id,
            resource_id=order_id,
            details={
                "item_ids": item_ids,
                "refund_destination": refund_destination,
            },
        )

    def _tool_exchange_delivered_order_items(
        self,
        *,
        actor_id: str,
        order_id: str,
        item_ids: list[str],
        new_item_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        order = self._order(order_id)
        if order["status"] != "delivered":
            raise ValueError("order is not delivered")
        if not set(item_ids).issubset(order["item_ids"]):
            raise ValueError("item does not belong to order")
        order["status"] = "exchange_requested"
        return self._audit(
            action="exchange_delivered_order_items",
            actor_id=actor_id,
            resource_id=order_id,
            details={"item_ids": item_ids, "new_item_ids": new_item_ids or []},
        )

    def _tool_cancel_pending_order(
        self, *, actor_id: str, order_id: str, reason: str
    ) -> dict[str, Any]:
        order = self._order(order_id)
        if order["status"] != "pending":
            raise ValueError("order is not pending")
        order["status"] = "cancelled"
        return self._audit(
            action="cancel_pending_order",
            actor_id=actor_id,
            resource_id=order_id,
            details={"reason": reason},
        )

    def _tool_modify_pending_order_address(
        self, *, actor_id: str, order_id: str, address: str = "updated address"
    ) -> dict[str, Any]:
        order = self._order(order_id)
        if order["status"] != "pending":
            raise ValueError("order is not pending")
        order["address"] = address
        return self._audit(
            action="modify_pending_order_address",
            actor_id=actor_id,
            resource_id=order_id,
            details={"address": address},
        )


class TelecomEnvironment(BaseVirtualEnvironment):
    """Family-plan, connectivity, and line-control sandbox."""

    definitions: ClassVar[dict[str, VirtualToolDefinition]] = {
        "get_line_status": VirtualToolDefinition(
            name="get_line_status",
            description="Read one mobile line's status and current options.",
            effect="read",
        ),
        "run_connectivity_test": VirtualToolDefinition(
            name="run_connectivity_test",
            description="Run a diagnostic against one home internet service.",
            effect="read",
        ),
        "set_roaming": VirtualToolDefinition(
            name="set_roaming",
            description="Enable or disable roaming for one line and region.",
            effect="write",
        ),
        "set_data_pass": VirtualToolDefinition(
            name="set_data_pass",
            description="Set a temporary data pass on one mobile line.",
            effect="write",
        ),
        "suspend_line": VirtualToolDefinition(
            name="suspend_line",
            description="Suspend a lost or compromised mobile line.",
            effect="write",
        ),
        "restart_router": VirtualToolDefinition(
            name="restart_router",
            description="Restart the router attached to one service.",
            effect="write",
        ),
        "apply_service_credit": VirtualToolDefinition(
            name="apply_service_credit",
            description="Apply a service credit after a confirmed outage.",
            effect="write",
        ),
    }

    @classmethod
    def default_state(cls) -> dict[str, Any]:
        return {
            "lines": {
                "line:0182": {"status": "active", "roaming": None, "data_gb": 0},
                "line:0812": {"status": "active", "roaming": None, "data_gb": 0},
                "line:teen": {"status": "active", "roaming": None, "data_gb": 0},
                "line:maya": {"status": "active", "roaming": None, "data_gb": 0},
                "line:leo": {"status": "active", "roaming": None, "data_gb": 0},
                "line:nina": {"status": "active", "roaming": None, "data_gb": 0},
                "line:omar": {"status": "active", "roaming": None, "data_gb": 0},
            },
            "services": {
                "service:home": {
                    "connectivity": "degraded",
                    "last_test": None,
                    "router_restarted": False,
                    "credit": 0,
                }
            },
            "audit_events": [],
        }

    def _line(self, line_id: str) -> dict[str, Any]:
        line = self.state["lines"].get(line_id)
        if line is None:
            raise ValueError(f"line not found: {line_id}")
        return line

    def _service(self, service_id: str) -> dict[str, Any]:
        service = self.state["services"].get(service_id)
        if service is None:
            raise ValueError(f"service not found: {service_id}")
        return service

    def _audit(
        self, action: str, actor_id: str, resource_id: str, **details: Any
    ) -> dict[str, Any]:
        event = {
            "action": action,
            "actor_id": actor_id,
            "resource_id": resource_id,
            **details,
        }
        self.state["audit_events"].append(event)
        return event

    def _tool_get_line_status(self, *, actor_id: str, line_id: str) -> dict[str, Any]:
        return {
            "line_id": line_id,
            "requested_by": actor_id,
            **deepcopy(self._line(line_id)),
        }

    def _tool_run_connectivity_test(
        self, *, actor_id: str, service_id: str
    ) -> dict[str, Any]:
        service = self._service(service_id)
        service["last_test"] = "outage_confirmed"
        return self._audit(
            "run_connectivity_test", actor_id, service_id, result="outage_confirmed"
        )

    def _tool_set_roaming(
        self, *, actor_id: str, line_id: str, region: str, enabled: bool = True
    ) -> dict[str, Any]:
        self._line(line_id)["roaming"] = region if enabled else None
        return self._audit(
            "set_roaming", actor_id, line_id, region=region, enabled=enabled
        )

    def _tool_set_data_pass(
        self, *, actor_id: str, line_id: str, gigabytes: int
    ) -> dict[str, Any]:
        if gigabytes <= 0:
            raise ValueError("gigabytes must be positive")
        self._line(line_id)["data_gb"] = gigabytes
        return self._audit("set_data_pass", actor_id, line_id, gigabytes=gigabytes)

    def _tool_suspend_line(
        self, *, actor_id: str, line_id: str, reason: str
    ) -> dict[str, Any]:
        self._line(line_id)["status"] = "suspended"
        return self._audit("suspend_line", actor_id, line_id, reason=reason)

    def _tool_restart_router(self, *, actor_id: str, service_id: str) -> dict[str, Any]:
        service = self._service(service_id)
        service["router_restarted"] = True
        service["connectivity"] = "restored"
        return self._audit("restart_router", actor_id, service_id)

    def _tool_apply_service_credit(
        self, *, actor_id: str, service_id: str, amount: int
    ) -> dict[str, Any]:
        if amount <= 0:
            raise ValueError("amount must be positive")
        self._service(service_id)["credit"] += amount
        return self._audit("apply_service_credit", actor_id, service_id, amount=amount)


class AirlineEnvironment(BaseVirtualEnvironment):
    """Shared-booking, disruption, seat, and baggage sandbox."""

    definitions: ClassVar[dict[str, VirtualToolDefinition]] = {
        "get_booking": VirtualToolDefinition(
            name="get_booking", description="Read one shared booking.", effect="read"
        ),
        "search_alternative_flights": VirtualToolDefinition(
            name="search_alternative_flights",
            description="Search alternatives without changing the booking.",
            effect="read",
        ),
        "change_seat": VirtualToolDefinition(
            name="change_seat",
            description="Change one passenger's seat.",
            effect="write",
        ),
        "quote_cancellation": VirtualToolDefinition(
            name="quote_cancellation",
            description="Quote one passenger's cancellation options.",
            effect="read",
        ),
        "cancel_itinerary": VirtualToolDefinition(
            name="cancel_itinerary",
            description="Cancel one passenger's itinerary.",
            effect="write",
        ),
        "create_baggage_claim": VirtualToolDefinition(
            name="create_baggage_claim",
            description="Create a baggage claim for one passenger.",
            effect="write",
        ),
        "rebook_passenger": VirtualToolDefinition(
            name="rebook_passenger",
            description="Move one passenger to an alternative flight.",
            effect="write",
        ),
        "get_flight_status": VirtualToolDefinition(
            name="get_flight_status",
            description="Read current gate and flight status.",
            effect="read",
        ),
        "request_mobility_assistance": VirtualToolDefinition(
            name="request_mobility_assistance",
            description="Request mobility assistance for one passenger.",
            effect="write",
        ),
    }

    @classmethod
    def default_state(cls) -> dict[str, Any]:
        passengers = {
            user_id: {"status": "booked", "seat": seat, "assistance": False}
            for user_id, seat in {
                "guardian": "14C",
                "child": "14B",
                "traveler": "18A",
                "payer": "18B",
                "passenger": "22C",
                "organizer": "20A",
                "maya": "20B",
                "leo": "20C",
                "nina": "21A",
                "omar": "21B",
                "minor": "21C",
            }.items()
        }
        return {
            "bookings": {
                "booking:MU100": {"passengers": passengers, "flight_id": "flight:TA100"}
            },
            "flights": {
                "flight:TA100": {"status": "delayed", "gate": "C18"},
                "flight:TA220": {"status": "on_time", "gate": "B7"},
                "flight:TA330": {"status": "on_time", "gate": "D4"},
            },
            "baggage_claims": {},
            "audit_events": [],
        }

    def _booking(self, booking_id: str) -> dict[str, Any]:
        booking = self.state["bookings"].get(booking_id)
        if booking is None:
            raise ValueError(f"booking not found: {booking_id}")
        return booking

    def _passenger(self, booking_id: str, passenger_id: str) -> dict[str, Any]:
        passenger = self._booking(booking_id)["passengers"].get(passenger_id)
        if passenger is None:
            raise ValueError(f"passenger not found: {passenger_id}")
        return passenger

    def _audit(
        self, action: str, actor_id: str, resource_id: str, **details: Any
    ) -> dict[str, Any]:
        event = {
            "action": action,
            "actor_id": actor_id,
            "resource_id": resource_id,
            **details,
        }
        self.state["audit_events"].append(event)
        return event

    def _tool_get_booking(self, *, actor_id: str, booking_id: str) -> dict[str, Any]:
        return {
            "booking_id": booking_id,
            "requested_by": actor_id,
            **deepcopy(self._booking(booking_id)),
        }

    def _tool_search_alternative_flights(
        self, *, actor_id: str, booking_id: str, date: str
    ) -> dict[str, Any]:
        self._booking(booking_id)
        return {
            "booking_id": booking_id,
            "date": date,
            "requested_by": actor_id,
            "options": ["flight:TA220", "flight:TA330"],
        }

    def _tool_change_seat(
        self, *, actor_id: str, booking_id: str, passenger_id: str, seat: str
    ) -> dict[str, Any]:
        self._passenger(booking_id, passenger_id)["seat"] = seat
        return self._audit(
            "change_seat",
            actor_id,
            f"passenger:{passenger_id}",
            booking_id=booking_id,
            passenger_id=passenger_id,
            seat=seat,
        )

    def _tool_quote_cancellation(
        self, *, actor_id: str, booking_id: str, passenger_id: str
    ) -> dict[str, Any]:
        self._passenger(booking_id, passenger_id)
        return {
            "booking_id": booking_id,
            "passenger_id": passenger_id,
            "requested_by": actor_id,
            "options": ["original_payment", "travel_credit"],
        }

    def _tool_cancel_itinerary(
        self, *, actor_id: str, booking_id: str, passenger_id: str, refund_method: str
    ) -> dict[str, Any]:
        self._passenger(booking_id, passenger_id)["status"] = "cancelled"
        return self._audit(
            "cancel_itinerary",
            actor_id,
            f"passenger:{passenger_id}",
            booking_id=booking_id,
            passenger_id=passenger_id,
            refund_method=refund_method,
        )

    def _tool_create_baggage_claim(
        self, *, actor_id: str, booking_id: str, passenger_id: str, color: str
    ) -> dict[str, Any]:
        self._passenger(booking_id, passenger_id)
        claim = {
            "claim_id": "claim:BG42",
            "passenger_id": passenger_id,
            "color": color,
            "created_by": actor_id,
        }
        self.state["baggage_claims"]["claim:BG42"] = claim
        return claim

    def _tool_rebook_passenger(
        self, *, actor_id: str, booking_id: str, passenger_id: str, flight_id: str
    ) -> dict[str, Any]:
        if flight_id not in self.state["flights"]:
            raise ValueError(f"flight not found: {flight_id}")
        passenger = self._passenger(booking_id, passenger_id)
        passenger["flight_id"] = flight_id
        passenger["status"] = "rebooked"
        return self._audit(
            "rebook_passenger",
            actor_id,
            f"passenger:{passenger_id}",
            booking_id=booking_id,
            passenger_id=passenger_id,
            flight_id=flight_id,
        )

    def _tool_get_flight_status(
        self, *, actor_id: str, flight_id: str
    ) -> dict[str, Any]:
        flight = self.state["flights"].get(flight_id)
        if flight is None:
            raise ValueError(f"flight not found: {flight_id}")
        return {"flight_id": flight_id, "requested_by": actor_id, **deepcopy(flight)}

    def _tool_request_mobility_assistance(
        self, *, actor_id: str, booking_id: str, passenger_id: str
    ) -> dict[str, Any]:
        self._passenger(booking_id, passenger_id)["assistance"] = True
        return self._audit(
            "request_mobility_assistance",
            actor_id,
            f"passenger:{passenger_id}",
            booking_id=booking_id,
            passenger_id=passenger_id,
        )


ENVIRONMENT_TYPES: dict[str, type[BaseVirtualEnvironment]] = {
    "retail": RetailAuditEnvironment,
    "shared_home": SharedHomeEnvironment,
    "vehicle": VehicleEnvironment,
    "meeting": MeetingEnvironment,
    "education": EducationEnvironment,
    "knowledge": KnowledgeEnvironment,
    "factorial_sandbox": FactorialScenarioEnvironment,
    "telecom": TelecomEnvironment,
    "airline": AirlineEnvironment,
}


def create_virtual_environment(binding: EnvironmentBinding) -> BaseVirtualEnvironment:
    """Construct a virtual environment and verify its declared tool surface."""
    if binding.domain.startswith("v07_"):
        # Local import avoids a module cycle: the business tool layer reuses
        # the common environment base and public tool-definition model above.
        from experiments.tau_multiuser_voice.business_tooling_v07 import (
            BusinessDomainEnvironment,
            contract_for_domain,
        )

        if contract_for_domain(binding.domain) is None:
            if binding.tool_names:
                raise ValueError(
                    f"tool-free v0.7 domain declares tools: {binding.tool_names}"
                )
            return KnowledgeEnvironment(binding.initial_state or None)
        environment = BusinessDomainEnvironment(binding)
        undeclared = set(binding.tool_names) - environment.tool_names
        if undeclared:
            raise ValueError(f"binding declares unknown tools: {sorted(undeclared)}")
        return environment
    if binding.domain not in ENVIRONMENT_TYPES:
        raise ValueError(f"no virtual environment for domain: {binding.domain}")
    environment = ENVIRONMENT_TYPES[binding.domain](binding.initial_state or None)
    undeclared = set(binding.tool_names) - environment.tool_names
    if undeclared:
        raise ValueError(f"binding declares unknown tools: {sorted(undeclared)}")
    return environment
