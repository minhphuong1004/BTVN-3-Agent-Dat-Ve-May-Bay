"""Harness = deterministic control layer around the model.

Agent = Model + Harness. The model only *proposes* tool calls; this layer
decides whether they run, whether the task is done, and when to hand over to a
human. Nothing here imports LangChain at module level, so it is unit-testable.

Layers:
    1. Constraints are data        -> domain.Constraints
    2. Data validation             -> Harness.validate
    3. Permission check            -> Harness.check_permission
    4. Completion check (by code)  -> Harness.check_completion
    5. Handoff / boundary          -> Harness.handoff
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .domain import Constraints

# Tools the agent may call at all. Anything else (e.g. cancel_booking) is denied.
DEFAULT_ALLOWED_TOOLS = frozenset({"search_flights", "book_seat", "pay", "get_booking"})

# Required string arguments per tool (schema-level validation).
REQUIRED_STR_ARGS: dict[str, tuple[str, ...]] = {
    "search_flights": ("origin", "destination", "date"),
    "book_seat": ("flight",),
    "pay": ("code",),
    "get_booking": ("code",),
}


class Harness:
    def __init__(
        self,
        constraints: Constraints,
        get_flights: Callable[[], list[dict]],
        bookings: dict[str, dict],
        *,
        allowed_tools: frozenset[str] = DEFAULT_ALLOWED_TOOLS,
        max_tool_calls: int = 12,
        enforce: bool = True,
    ) -> None:
        self.constraints = constraints
        self.get_flights = get_flights      # ground-truth flight list (backend)
        self.bookings = bookings            # ground-truth booking DB (backend)
        self.allowed_tools = allowed_tools
        self.max_tool_calls = max_tool_calls
        self.enforce = enforce              # False only for the ablation experiment
        self.log: list[tuple[str, dict, dict]] = []

    # ------------------------------------------------------------------
    # 2. DATA VALIDATION
    # ------------------------------------------------------------------
    def validate(self, tool_name: str, args: Any) -> tuple[bool, str | None]:
        if not isinstance(args, dict):
            return False, "Tool arguments must be a dictionary."
        for key in REQUIRED_STR_ARGS.get(tool_name, ()):
            value = args.get(key)
            if not isinstance(value, str):
                return False, f"'{key}' must be a string."
            if not value.strip():
                return False, f"'{key}' cannot be empty."
        return True, None

    # ------------------------------------------------------------------
    # 3. PERMISSION CHECK  (returns a rule name + reason, or None if allowed)
    # ------------------------------------------------------------------
    def check_permission(self, tool_name: str, args: dict) -> tuple[str, str] | None:
        if tool_name not in self.allowed_tools:
            return "tool_not_allowed", f"Tool '{tool_name}' is not permitted for this agent."

        if len(self.log) >= self.max_tool_calls:
            return "budget_exceeded", f"Tool-call budget ({self.max_tool_calls}) exhausted."

        paid = [b for b in self.bookings.values() if b.get("paid")]

        if tool_name == "search_flights":
            c = self.constraints
            if not c.route_ok(args["origin"], args["destination"], args["date"]):
                return "invalid_route", (
                    f"Search must be {c.origin}->{c.destination} on {c.date}, "
                    f"got {args['origin']}->{args['destination']} on {args['date']}."
                )

        elif tool_name == "book_seat":
            flight_id = args["flight"]
            flight = next((f for f in self.get_flights() if f["flight"] == flight_id), None)
            if flight is None:
                return "unknown_flight", f"Unknown flight: {flight_id}"
            if not self.constraints.is_ok(flight):
                return "constraint_violation", (
                    f"{flight_id} violates the booking constraints: "
                    f"{self.constraints.to_prompt()}"
                )
            if paid:
                return "already_ticketed", "A ticket is already paid; only one is allowed."
            if self.bookings:
                return "duplicate_hold", (
                    f"Already holding {next(iter(self.bookings))}; pay it instead of booking again."
                )

        elif tool_name == "pay":
            code = args["code"]
            booking = self.bookings.get(code)
            if booking is None:
                return "unknown_booking", f"Cannot pay unknown booking: {code}"
            if booking.get("paid"):
                return "already_paid", f"Booking {code} is already paid."
            if paid:
                return "already_ticketed", "A ticket is already paid; only one is allowed."
            if not self.constraints.is_ok(booking) or booking.get("amount", 0) > self.constraints.max_price:
                return "constraint_violation", f"Booking {code} violates the constraints; refusing to pay."

        return None

    def pre_check(self, tool_name: str, args: Any) -> dict | None:
        """Single entry point used by every design. Returns a rejection or None."""
        if not self.enforce:
            return None
        ok, reason = self.validate(tool_name, args)
        if not ok:
            return {"status": "invalid_data", "reason": reason}
        denied = self.check_permission(tool_name, args)
        if denied:
            rule, reason = denied
            return {"status": "denied", "rule": rule, "reason": reason}
        return None

    # ------------------------------------------------------------------
    # 4. COMPLETION CHECK  (decided by code, never by the model's words)
    # ------------------------------------------------------------------
    def check_completion(self) -> bool:
        paid = [b for b in self.bookings.values() if b.get("paid")]
        return len(paid) == 1 and self.constraints.is_ok(paid[0])

    # ------------------------------------------------------------------
    # 5. HANDOFF / BOUNDARY
    # ------------------------------------------------------------------
    def handoff(self, reason: str) -> dict:
        """Package enough state for a human to continue. The agent never
        relaxes constraints on its own - it can only ask."""
        return {
            "reason": reason,
            "constraints": self.constraints.to_prompt(),
            "done_so_far": [
                f"{code}: paid={b.get('paid')}" for code, b in self.bookings.items()
            ] or ["Nothing booked, nothing paid"],
            "tried": [f"{t}({a}) -> {r.get('status', 'unknown')}" for t, a, r in self.log],
            "question": (
                "Do you want to retry later, change the date/time, "
                "or raise the price limit?"
            ),
        }

    # ------------------------------------------------------------------
    # Execution paths
    # ------------------------------------------------------------------
    def _record(self, tool: str, args: Any, result: dict) -> dict:
        self.log.append((tool, dict(args) if isinstance(args, dict) else {"_raw": args}, result))
        return result

    def execute_step(self, tool_name: str, args: dict, tools: dict[str, Any]) -> dict:
        """Run one tool call outside LangChain (Plan-then-Execute / Hybrid)."""
        rejected = self.pre_check(tool_name, args)
        if rejected:
            return self._record(tool_name, args, rejected)
        tool = tools.get(tool_name)
        if tool is None:
            return self._record(tool_name, args, {"status": "error", "error": f"Unknown tool: {tool_name}"})
        try:
            result = tool.invoke(args) if hasattr(tool, "invoke") else tool(**args)
        except Exception as exc:  # tool crashed -> contain it
            result = {"status": "error", "error": str(exc)}
        return self._record(tool_name, args, result)

    def middleware(self):
        """LangChain middleware that guards every ReAct tool call."""
        from langchain.agents.middleware import wrap_tool_call
        from langchain_core.messages import ToolMessage

        harness = self

        @wrap_tool_call
        def guard(request, handler):
            call = request.tool_call
            name, args = call["name"], call["args"]

            rejected = harness.pre_check(name, args)
            if rejected:
                harness._record(name, args, rejected)
                return ToolMessage(content=json.dumps(rejected), tool_call_id=call["id"])

            try:
                message = handler(request)
                try:
                    parsed = json.loads(message.content)
                    result = parsed if isinstance(parsed, dict) else {"status": "ok"}
                except (TypeError, json.JSONDecodeError):
                    result = {"status": "ok", "content": message.content}
                harness._record(name, args, result)
                return message
            except Exception as exc:
                result = {"status": "error", "error": str(exc)}
                harness._record(name, args, result)
                return ToolMessage(content=json.dumps(result), tool_call_id=call["id"])

        return guard
