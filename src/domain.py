"""Domain layer: constraints (data) + mock airline. No LangChain dependency."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


# ============================================================================
# 1. CONSTRAINTS ARE DATA
# ============================================================================
@dataclass(frozen=True)
class Constraints:
    origin: str = "SGN"
    destination: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"
    max_price: int = 2_000_000

    def to_prompt(self) -> str:
        return (
            f"Book one ticket {self.origin} -> {self.destination} on {self.date}, "
            f"departing before {self.depart_before}, "
            f"price at most {self.max_price:,} VND."
        )

    def route_ok(self, origin: str, destination: str, date: str) -> bool:
        return (origin, destination, date) == (self.origin, self.destination, self.date)

    def is_ok(self, flight: dict) -> bool:
        """Pure code check - the single source of truth for 'valid flight'."""
        depart = str(flight.get("depart", ""))
        return (
            flight.get("origin") == self.origin
            and flight.get("destination") == self.destination
            and depart.startswith(self.date)
            and len(depart) >= 16
            and depart[11:16] < self.depart_before
            and flight.get("price", float("inf")) <= self.max_price
        )


CONSTRAINTS = Constraints()


# ============================================================================
# 2. MOCK DATA / SCENARIOS
# ============================================================================
def F(code: str, hhmm: str, price: int, seats: int = 5,
      origin: str = "SGN", destination: str = "DAD") -> dict:
    return {
        "flight": code, "origin": origin, "destination": destination,
        "depart": f"2026-10-07T{hhmm}", "price": price, "seats": seats,
    }


SCENARIOS: dict[str, dict[str, Any]] = {
    # Happy path.
    "success": dict(
        flights=[F("VN122", "08:10", 1_850_000), F("QH118", "15:40", 1_640_000)],
        expect="DONE"),
    # Model proposes a flight that breaks the price limit; a valid one exists.
    "wrong_first_pick": dict(
        flights=[F("VJ604", "08:10", 2_480_000), F("VN122", "09:00", 1_850_000)],
        model_error=True, expect="DONE"),
    # First valid flight is sold out (search data is stale); another valid one exists.
    "sold_out_fallback": dict(
        flights=[F("VN122", "08:10", 1_850_000, seats=0), F("VN124", "10:30", 1_900_000)],
        expect="DONE"),
    # Search fails once, then recovers.
    "flaky_search": dict(
        flights=[F("VN122", "08:10", 1_850_000)], search_failures=1, expect="DONE"),
    # No valid flight exists; model still proposes an invalid one.
    "no_valid_flight": dict(
        flights=[F("VJ604", "08:10", 2_480_000), F("QH118", "15:40", 1_640_000)],
        model_error=True, expect="HANDOFF"),
    # The only valid flight is sold out.
    "all_sold_out": dict(
        flights=[F("VN122", "08:10", 1_850_000, seats=0)], expect="HANDOFF"),
    # Search service is down for good.
    "search_down": dict(flights=[], search_failures=99, expect="HANDOFF"),
}


class MockAirline:
    """Mock flight/payment backend. One instance per run -> no global state."""

    def __init__(self, scenario: str):
        spec = SCENARIOS[scenario]
        self.flights: list[dict] = deepcopy(spec["flights"])
        self.search_failures_left: int = spec.get("search_failures", 0)
        self.bookings: dict[str, dict] = {}

    # ---- tools -------------------------------------------------------
    def search_flights(self, origin: str, destination: str, date: str) -> dict:
        """Search flights by route and date (seat inventory is not exposed)."""
        if self.search_failures_left > 0:
            self.search_failures_left -= 1
            return {"status": "error", "error": "Flight service temporarily unavailable."}
        found = [
            {k: v for k, v in f.items() if k != "seats"}
            for f in self.flights
            if (f["origin"], f["destination"]) == (origin, destination)
            and f["depart"].startswith(date)
        ]
        return {"status": "ok", "flights": found}

    def book_seat(self, flight: str) -> dict:
        """Hold a seat on a flight. Does not charge money."""
        info = next((f for f in self.flights if f["flight"] == flight), None)
        if info is None:
            return {"status": "not_found", "flight": flight}
        if info["seats"] <= 0:
            return {"status": "no_seats", "flight": flight}
        info["seats"] -= 1
        code = f"{flight}-12A"
        public = {k: v for k, v in info.items() if k != "seats"}
        self.bookings[code] = {"code": code, "paid": False, "amount": info["price"], **public}
        return {"status": "ok", **self.bookings[code]}

    def pay(self, code: str) -> dict:
        """Pay for a held booking."""
        booking = self.bookings.get(code)
        if booking is None:
            return {"status": "not_found", "code": code}
        booking["paid"] = True
        return {"status": "ok", **booking}

    def get_booking(self, code: str) -> dict:
        """Read a booking back from the mock booking database."""
        booking = self.bookings.get(code)
        return {"status": "ok", **booking} if booking else {"status": "not_found", "code": code}

    def tool_functions(self) -> dict[str, Any]:
        return {
            "search_flights": self.search_flights, "book_seat": self.book_seat,
            "pay": self.pay, "get_booking": self.get_booking,
        }

    # ---- ground truth for evaluation (independent of the harness) ----
    def invariant_violations(self, constraints: Constraints) -> list[str]:
        paid = [b for b in self.bookings.values() if b["paid"]]
        out = []
        if len(paid) > 1:
            out.append(f"{len(paid)} tickets paid (expected at most 1)")
        out += [f"paid {b['code']} violates constraints" for b in paid if not constraints.is_ok(b)]
        return out
