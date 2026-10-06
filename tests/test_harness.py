"""Unit tests for the harness layers. Mapped to the 5 cases in the original skeleton."""
import pytest

from src.domain import CONSTRAINTS, MockAirline
from src.harness import Harness


def make(scenario="success"):
    airline = MockAirline(scenario)
    h = Harness(CONSTRAINTS, lambda: airline.flights, airline.bookings)
    return airline, h, airline.tool_functions()


# --- 1. valid booking ------------------------------------------------------
def test_valid_booking_completes():
    _, h, tools = make("success")
    assert h.execute_step("book_seat", {"flight": "VN122"}, tools)["status"] == "ok"
    assert h.execute_step("pay", {"code": "VN122-12A"}, tools)["status"] == "ok"
    assert h.check_completion() is True


def test_hold_without_payment_is_not_complete():
    _, h, tools = make("success")
    h.execute_step("book_seat", {"flight": "VN122"}, tools)
    assert h.check_completion() is False


# --- 2. invalid route ------------------------------------------------------
def test_invalid_route_is_denied():
    _, h, tools = make("success")
    r = h.execute_step("search_flights",
                       {"origin": "SGN", "destination": "HAN", "date": "2026-10-07"}, tools)
    assert r["status"] == "denied" and r["rule"] == "invalid_route"


# --- 3. flight not found ---------------------------------------------------
def test_unknown_flight_is_denied():
    _, h, tools = make("success")
    r = h.execute_step("book_seat", {"flight": "XX999"}, tools)
    assert r["status"] == "denied" and r["rule"] == "unknown_flight"


# --- 4. no permission ------------------------------------------------------
def test_tool_outside_allow_list_is_denied():
    _, h, tools = make("success")
    r = h.execute_step("cancel_booking", {"code": "X"}, tools)
    assert r["rule"] == "tool_not_allowed"


def test_flight_violating_constraints_cannot_be_booked():
    airline, h, tools = make("no_valid_flight")
    r = h.execute_step("book_seat", {"flight": "VJ604"}, tools)      # 2.48M > 2M
    assert r["rule"] == "constraint_violation" and airline.bookings == {}
    r = h.execute_step("book_seat", {"flight": "QH118"}, tools)      # 15:40 >= 12:00
    assert r["rule"] == "constraint_violation"


def test_cannot_pay_unknown_or_twice_or_buy_second_ticket():
    _, h, tools = make("success")
    assert h.execute_step("pay", {"code": "NOPE"}, tools)["rule"] == "unknown_booking"
    h.execute_step("book_seat", {"flight": "VN122"}, tools)
    assert h.execute_step("book_seat", {"flight": "VN122"}, tools)["rule"] == "duplicate_hold"
    h.execute_step("pay", {"code": "VN122-12A"}, tools)
    assert h.execute_step("pay", {"code": "VN122-12A"}, tools)["rule"] == "already_paid"
    assert h.execute_step("book_seat", {"flight": "VN122"}, tools)["rule"] == "already_ticketed"


# --- 5. no available seats -------------------------------------------------
def test_no_seats_is_reported_and_leads_to_handoff():
    _, h, tools = make("all_sold_out")
    r = h.execute_step("book_seat", {"flight": "VN122"}, tools)
    assert r["status"] == "no_seats"
    assert h.check_completion() is False
    ho = h.handoff("sold out")
    assert ho["done_so_far"] == ["Nothing booked, nothing paid"] and ho["tried"]


# --- data validation & budget ---------------------------------------------
@pytest.mark.parametrize("tool,args", [
    ("book_seat", {}), ("book_seat", {"flight": "  "}), ("book_seat", {"flight": 5}),
    ("pay", {"code": None}), ("search_flights", {"origin": "SGN"}),
])
def test_invalid_data_rejected(tool, args):
    _, h, tools = make("success")
    assert h.execute_step(tool, args, tools)["status"] == "invalid_data"


def test_tool_call_budget():
    _, h, tools = make("success")
    h.max_tool_calls = 2
    for _ in range(2):
        h.execute_step("get_booking", {"code": "x"}, tools)
    assert h.execute_step("get_booking", {"code": "x"}, tools)["rule"] == "budget_exceeded"


def test_tool_crash_is_contained():
    _, h, _ = make("success")
    def boom(**kw): raise RuntimeError("db down")
    r = h.execute_step("get_booking", {"code": "x"}, {"get_booking": boom})
    assert r == {"status": "error", "error": "db down"}
