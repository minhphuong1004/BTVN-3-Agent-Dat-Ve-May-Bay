"""BTVN#3 - Flight booking agent with three designs (ReAct / Plan-then-Execute / Hybrid).

Run (from the project root, NOT from inside src/):
    python -m src.flight_booking_agent --design all            # evaluation table
    python -m src.flight_booking_agent --design react --scenario sold_out_fallback
    python -m src.flight_booking_agent --ablation              # harness ON vs OFF

Agent = Model + Harness. The model only proposes; src/harness.py decides.
In MOCK mode the "model" is a deterministic rule-based stand-in (so results are
reproducible and cost nothing). Set MOCK_MODE=false + LLM_* in .env for a real LLM.

NOTE: this file deliberately has no `from __future__ import annotations`, because
pydantic models are declared inside functions.
"""
import argparse
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from dotenv import load_dotenv

from .domain import CONSTRAINTS, SCENARIOS, Constraints, MockAirline
from .harness import Harness

load_dotenv()

MAX_REACT_STEPS = 10
MAX_REPLANS = 2
SEARCH_ATTEMPTS = 3
Action = tuple  # ("call", tool_name, args) | ("final", text)


# ============================================================================
# 1. MODEL SETUP
# ============================================================================
def build_real_model():
    """OpenAI-compatible chat model (OpenAI, Gemini-compat, local servers...)."""
    from langchain_openai import ChatOpenAI

    api_key, model_name = os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL")
    if not api_key or not model_name:
        raise RuntimeError("Set LLM_API_KEY and LLM_MODEL in .env when MOCK_MODE=false.")
    kwargs: dict[str, Any] = {"model": model_name, "api_key": api_key, "temperature": 0}
    if os.getenv("LLM_BASE_URL"):
        kwargs["base_url"] = os.getenv("LLM_BASE_URL")
    return ChatOpenAI(**kwargs)


# ============================================================================
# 2. SHARED RUN SETUP / HELPERS
# ============================================================================
def setup(scenario: str, enforce: bool = True) -> tuple[MockAirline, Harness]:
    airline = MockAirline(scenario)
    harness = Harness(CONSTRAINTS, lambda: airline.flights, airline.bookings, enforce=enforce)
    return airline, harness


def finish(design: str, scenario: str, airline: MockAirline, harness: Harness,
           answer: str, llm_calls: int, reason: str) -> dict:
    """Build the result row. Completion is decided by CODE, not by `answer`."""
    done = harness.check_completion()
    log = harness.log
    violations = airline.invariant_violations(harness.constraints)
    return {
        "design": design,
        "scenario": scenario,
        "outcome": "DONE" if done else "HANDOFF",
        "expected": SCENARIOS[scenario]["expect"],
        # correct = right outcome AND no real-world rule broken (ground truth from the backend)
        "correct": ("DONE" if done else "HANDOFF") == SCENARIOS[scenario]["expect"] and not violations,
        "tool_calls": len(log),
        "llm_calls": llm_calls,
        "blocked": sum(1 for _, _, r in log if r.get("status") in ("denied", "invalid_data")),
        "violations": violations,
        "answer": answer,
        "handoff": None if done else harness.handoff(reason or "Task not completed."),
    }


def search(harness: Harness, tools: dict, attempts: int = 1) -> dict:
    c = harness.constraints
    result: dict = {}
    for _ in range(attempts):
        result = harness.execute_step(
            "search_flights",
            {"origin": c.origin, "destination": c.destination, "date": c.date}, tools)
        if result.get("status") == "ok":
            break
    return result


# ============================================================================
# 3. DESIGN 1 - ReAct  (think -> act -> observe, one step at a time)
# ============================================================================
SYSTEM_PROMPT = """
You are a flight booking agent. Use the tools to search, hold a seat, pay, and read a booking.
Only book a flight that satisfies ALL constraints.
React to every tool result. If a tool is denied or fails, adapt; if no valid option remains,
stop and explain. Never claim success unless the system confirms it.
""".strip()


def react_policy(history: list, c: Constraints, model_error: bool = False) -> Action:
    """Rule-based stand-in for the LLM in MOCK mode. It READS the tool results in
    `history` = [(tool, args, result), ...] and chooses the next action, so it is
    genuinely reactive (retries search, falls back to another flight)."""
    if any(n == "pay" and r.get("status") == "ok" for n, _, r in history):
        return "final", "Booked and paid."

    held = [r for n, _, r in history if n == "book_seat" and r.get("status") == "ok"]
    if held:
        if sum(1 for n, _, _ in history if n == "pay") >= 2:
            return "final", "Payment keeps failing; stopping."
        return "call", "pay", {"code": held[-1]["code"]}

    flights = next((r["flights"] for n, _, r in reversed(history)
                    if n == "search_flights" and r.get("status") == "ok"), None)
    if flights is None:
        errors = sum(1 for n, _, r in history if n == "search_flights")
        if errors >= SEARCH_ATTEMPTS:
            return "final", "Flight search is unavailable; stopping."
        return "call", "search_flights", {
            "origin": c.origin, "destination": c.destination, "date": c.date}

    tried = {a.get("flight") for n, a, _ in history if n == "book_seat"}
    if model_error and not tried:      # simulate a careless first pick (ignores price)
        pool = sorted(flights, key=lambda f: f["depart"])
    else:
        pool = [f for f in flights if c.is_ok(f)]
    pool = [f for f in pool if f["flight"] not in tried]
    if not pool:
        return "final", "No remaining valid flight."
    return "call", "book_seat", {"flight": pool[0]["flight"]}


def history_from_messages(messages: list) -> list:
    from langchain_core.messages import AIMessage, ToolMessage

    pending: dict[str, tuple[str, dict]] = {}
    history = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                pending[tc["id"]] = (tc["name"], tc["args"])
        elif isinstance(m, ToolMessage):
            name, args = pending.get(m.tool_call_id, ("?", {}))
            try:
                result = json.loads(m.content)
            except (TypeError, json.JSONDecodeError):
                result = {"status": "ok", "content": m.content}
            history.append((name, args, result if isinstance(result, dict) else {"status": "ok"}))
    return history


def make_reactive_mock_model(c: Constraints, model_error: bool):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ReactiveMock(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "reactive-mock"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            act = react_policy(history_from_messages(messages), c, model_error)
            if act[0] == "final":
                msg = AIMessage(content=act[1])
            else:
                msg = AIMessage(content="", tool_calls=[{
                    "name": act[1], "args": act[2],
                    "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}])
            return ChatResult(generations=[ChatGeneration(message=msg)])

    return ReactiveMock()


def langchain_tools(airline: MockAirline) -> list:
    from langchain_core.tools import tool

    @tool
    def search_flights(origin: str, destination: str, date: str) -> dict:
        """Search flights by route and date."""
        return airline.search_flights(origin, destination, date)

    @tool
    def book_seat(flight: str) -> dict:
        """Hold a seat on a flight. This does not charge money."""
        return airline.book_seat(flight)

    @tool
    def pay(code: str) -> dict:
        """Pay for a held booking."""
        return airline.pay(code)

    @tool
    def get_booking(code: str) -> dict:
        """Read a booking back from the booking database."""
        return airline.get_booking(code)

    return [search_flights, book_seat, pay, get_booking]


def _langchain_available() -> bool:
    try:
        import langchain.agents  # noqa: F401
        return True
    except ImportError:
        return False


def run_react(scenario: str, use_real_model: bool = False, enforce: bool = True,
              engine: str = "auto") -> dict:
    """engine: 'langchain' (create_agent + middleware) | 'python' (same policy,
    plain loop; only for machines without LangChain) | 'auto'."""
    airline, harness = setup(scenario, enforce)
    model_error = SCENARIOS[scenario].get("model_error", False)
    if engine == "auto":
        engine = "langchain" if (use_real_model or _langchain_available()) else "python"
    design = "ReAct" if engine == "langchain" else "ReAct (py-loop)"

    if engine == "langchain":
        from langchain.agents import create_agent
        from langchain.agents.middleware import ModelCallLimitMiddleware

        model = build_real_model() if use_real_model else \
            make_reactive_mock_model(harness.constraints, model_error)
        agent = create_agent(
            model=model, tools=langchain_tools(airline), system_prompt=SYSTEM_PROMPT,
            middleware=[harness.middleware(),
                        ModelCallLimitMiddleware(run_limit=MAX_REACT_STEPS, exit_behavior="end")])
        try:
            out = agent.invoke({"messages": [{"role": "user", "content": CONSTRAINTS.to_prompt()}]})
            answer = out["messages"][-1].content
            llm_calls = sum(1 for m in out["messages"] if m.type == "ai")
        except Exception as exc:
            answer, llm_calls = f"Agent error: {exc}", 0
    else:
        tools, history, llm_calls, answer = airline.tool_functions(), [], 0, "Step limit reached."
        for _ in range(MAX_REACT_STEPS):
            llm_calls += 1
            act = react_policy(history, harness.constraints, model_error)
            if act[0] == "final":
                answer = act[1]
                break
            result = harness.execute_step(act[1], act[2], tools)
            history.append((act[1], act[2], result))

    return finish(design, scenario, airline, harness, answer, llm_calls,
                  "ReAct run ended without a valid paid ticket.")


# ============================================================================
# 4. DESIGN 2 - Plan-then-Execute  (plan once, run the fixed plan, no replanning)
# ============================================================================
@dataclass
class Step:
    tool: str
    args: dict


@dataclass
class Plan:
    steps: list = field(default_factory=list)

    def dump(self) -> list:
        return [{"tool": s.tool, "args": s.args} for s in self.steps]


Planner = Callable[[Constraints, list, set], Plan]


def _standard_plan(flight: str) -> Plan:
    return Plan([Step("book_seat", {"flight": flight}),
                 Step("pay", {"code": "$booking_code"}),
                 Step("get_booking", {"code": "$booking_code"})])


def make_mock_planner(model_error: bool) -> Planner:
    """Rule-based stand-in for the planning LLM. With model_error, its FIRST plan
    ignores the price limit (a careless model); later plans use the feedback."""
    calls = {"n": 0}

    def planner(c: Constraints, flights: list, excluded: set) -> Plan:
        calls["n"] += 1
        pool = [f for f in flights if f["flight"] not in excluded]
        if model_error and calls["n"] == 1:
            pool.sort(key=lambda f: f["depart"])
        else:
            pool = [f for f in pool if c.is_ok(f)]
        return _standard_plan(pool[0]["flight"]) if pool else Plan([])

    return planner


def make_llm_planner(model) -> Planner:
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel

    class StepSchema(BaseModel):
        tool: Literal["book_seat", "pay", "get_booking"]
        flight: Optional[str] = None
        code: Optional[str] = None      # use "$booking_code" for the code from book_seat

    class PlanSchema(BaseModel):
        steps: list[StepSchema]

    chain = ChatPromptTemplate.from_messages([
        ("system",
         "You are a flight booking planner. Produce the whole plan before execution. "
         "Tools: book_seat(flight), pay(code), get_booking(code). Use code='$booking_code' "
         "for the code created by book_seat. Choose a flight satisfying ALL constraints and "
         "not in the excluded list. If none qualifies, return an empty plan."),
        ("human", "Goal: {goal}\nAvailable flights: {flights}\nExcluded: {excluded}"),
    ]) | model.with_structured_output(PlanSchema)

    def planner(c: Constraints, flights: list, excluded: set) -> Plan:
        out = chain.invoke({"goal": c.to_prompt(), "flights": json.dumps(flights),
                            "excluded": ", ".join(sorted(excluded)) or "none"})
        return Plan([Step(s.tool, {"flight": s.flight} if s.tool == "book_seat"
                                  else {"code": s.code or ""}) for s in out.steps])

    return planner


def get_planner(scenario: str, use_real_model: bool) -> Planner:
    if use_real_model:
        return make_llm_planner(build_real_model())
    return make_mock_planner(SCENARIOS[scenario].get("model_error", False))


def execute_plan(plan: Plan, harness: Harness, tools: dict) -> tuple[bool, Optional[str], dict]:
    """Run steps in order. Returns (ok, failed_flight_if_booking_step_failed, last_result)."""
    code, flight, result = "", None, {}
    for step in plan.steps:
        args = {k: (code if v == "$booking_code" else v) for k, v in step.args.items()}
        if step.tool == "book_seat":
            flight = args.get("flight")
        result = harness.execute_step(step.tool, args, tools)
        if result.get("status") != "ok":
            return False, (flight if step.tool == "book_seat" else None), result
        if step.tool == "book_seat":
            code = result["code"]
    return True, None, result


def run_plan_then_execute(scenario: str, use_real_model: bool = False,
                          enforce: bool = True) -> dict:
    airline, harness = setup(scenario, enforce)
    tools = airline.tool_functions()
    found = search(harness, tools, attempts=1)               # no retry: rigid by design
    plan = get_planner(scenario, use_real_model)(harness.constraints, found.get("flights", []), set())
    reason = "Search failed." if found.get("status") != "ok" else "Planner found no valid flight."
    if plan.steps:
        ok, _, last = execute_plan(plan, harness, tools)
        reason = "" if ok else f"Fixed plan failed at a step: {last}"
    return finish("Plan-then-Execute", scenario, airline, harness,
                  json.dumps(plan.dump()), 1, reason)


# ============================================================================
# 5. DESIGN 3 - Lai / Hybrid  (plan first, execute deterministically, REPLAN on failure)
# ============================================================================
def run_hybrid(scenario: str, use_real_model: bool = False, enforce: bool = True) -> dict:
    """Plan-then-Execute skeleton + ReAct-style feedback loop:
       search (with retry) -> plan -> execute -> on a failed BOOKING step, exclude that
       flight, re-search (inventory may have changed) and replan, up to MAX_REPLANS.
       Money steps (pay) are never blindly retried: failure there goes to a human."""
    airline, harness = setup(scenario, enforce)
    tools = airline.tool_functions()
    planner = get_planner(scenario, use_real_model)
    excluded: set = set()
    llm_calls, reason, plan = 0, "", Plan([])

    for _ in range(MAX_REPLANS + 1):
        found = search(harness, tools, attempts=SEARCH_ATTEMPTS)
        if found.get("status") != "ok":
            reason = "Flight search unavailable after retries."
            break
        plan = planner(harness.constraints, found["flights"], excluded)
        llm_calls += 1
        if not plan.steps:
            reason = "No valid candidate flight remains."
            break
        ok, bad_flight, last = execute_plan(plan, harness, tools)
        if ok:
            reason = ""
            break
        if bad_flight is None:
            reason = f"Non-booking step failed (not retried automatically): {last}"
            break
        excluded.add(bad_flight)
        reason = f"{bad_flight} failed ({last.get('status')}); replan limit reached."

    return finish("Hybrid", scenario, airline, harness, json.dumps(plan.dump()),
                  llm_calls, reason)


# ============================================================================
# 6. EVALUATION
# ============================================================================
DESIGNS: dict[str, Callable] = {
    "react": run_react, "plan": run_plan_then_execute, "hybrid": run_hybrid}


def evaluate(use_real_model: bool = False, enforce: bool = True,
             scenarios: Optional[list] = None) -> list:
    return [DESIGNS[d](s, use_real_model=use_real_model, enforce=enforce)
            for s in (scenarios or list(SCENARIOS)) for d in DESIGNS]


def print_evaluation(rows: list, title: str = "EVALUATION") -> None:
    print(f"\n=== {title} ===")
    print(f"{'Scenario':<18}{'Design':<20}{'Outcome':<9}{'OK?':<5}{'Tools':>6}{'LLM':>5}"
          f"{'Blocked':>8}  Violations")
    print("-" * 86)
    for r in rows:
        print(f"{r['scenario']:<18}{r['design']:<20}{r['outcome']:<9}"
              f"{'yes' if r['correct'] else 'NO':<5}{r['tool_calls']:>6}{r['llm_calls']:>5}"
              f"{r['blocked']:>8}  {'; '.join(r['violations']) or '-'}")
    print("\nSummary per design")
    print(f"{'Design':<20}{'Correct':>9}{'Avg tools':>11}{'Avg LLM':>9}{'Violations':>12}")
    for d in dict.fromkeys(r["design"] for r in rows):
        sub = [r for r in rows if r["design"] == d]
        print(f"{d:<20}{sum(r['correct'] for r in sub):>5}/{len(sub):<3}"
              f"{sum(r['tool_calls'] for r in sub) / len(sub):>11.1f}"
              f"{sum(r['llm_calls'] for r in sub) / len(sub):>9.1f}"
              f"{sum(len(r['violations']) for r in sub):>12}")


# ============================================================================
# 7. CLI
# ============================================================================
def main() -> None:
    p = argparse.ArgumentParser(description="BTVN#3 flight booking agent")
    p.add_argument("--design", choices=[*DESIGNS, "all"], default="all")
    p.add_argument("--scenario", choices=list(SCENARIOS))
    p.add_argument("--real", action="store_true", help="use the real LLM from .env")
    p.add_argument("--ablation", action="store_true", help="compare harness ON vs OFF")
    p.add_argument("--json", action="store_true", help="print full result rows as JSON")
    args = p.parse_args()

    use_real = args.real or os.getenv("MOCK_MODE", "true").lower() == "false"

    if args.ablation:
        scen = ["wrong_first_pick", "no_valid_flight"]
        print_evaluation(evaluate(use_real, True, scen), "HARNESS ON")
        print_evaluation(evaluate(use_real, False, scen), "HARNESS OFF (ablation)")
        return

    designs = list(DESIGNS) if args.design == "all" else [args.design]
    scenarios = [args.scenario] if args.scenario else list(SCENARIOS)
    rows = [DESIGNS[d](s, use_real_model=use_real) for s in scenarios for d in designs]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print_evaluation(rows)


if __name__ == "__main__":
    main()
