"""Behavioural tests for the three designs (mock mode, no LangChain needed)."""
import pytest

from src.domain import CONSTRAINTS, SCENARIOS
from src.flight_booking_agent import (react_policy, run_hybrid, run_plan_then_execute,
                                      run_react)


def test_policy_is_reactive_to_search_failure_and_sold_out():
    c = CONSTRAINTS
    assert react_policy([], c)[1] == "search_flights"
    h = [("search_flights", {}, {"status": "error"})]
    assert react_policy(h, c)[1] == "search_flights"                    # retries
    flights = [{"flight": "A", "origin": "SGN", "destination": "DAD", "depart": "2026-10-07T08:00", "price": 1}]
    h = [("search_flights", {}, {"status": "ok", "flights": flights}),
         ("book_seat", {"flight": "A"}, {"status": "no_seats"})]
    assert react_policy(h, c)[0] == "final"                              # nothing else to try


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_hybrid_and_react_always_correct_and_safe(scenario):
    for run in (run_hybrid, lambda s: run_react(s, engine="python")):
        row = run(scenario)
        assert row["correct"], row
        assert row["violations"] == []
        assert (row["handoff"] is None) == (row["outcome"] == "DONE")


@pytest.mark.parametrize("scenario", ["wrong_first_pick", "sold_out_fallback", "flaky_search"])
def test_plan_then_execute_is_rigid(scenario):
    row = run_plan_then_execute(scenario)
    assert row["outcome"] == "HANDOFF" and row["violations"] == []       # fails safely


@pytest.mark.parametrize("run", [run_react, run_plan_then_execute, run_hybrid])
def test_harness_off_lets_bad_booking_through(run):
    row = run("wrong_first_pick", enforce=False)
    assert row["violations"], "ablation should expose the violation the harness prevents"


def test_react_with_langchain_if_installed():
    pytest.importorskip("langchain.agents")
    row = run_react("sold_out_fallback", engine="langchain")
    assert row["outcome"] == "DONE" and row["violations"] == []
