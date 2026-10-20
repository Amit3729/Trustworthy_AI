"""Ask the guarded clinic assistant a question.

    python demo.py "What time do you open on Saturday?"
    python demo.py --verbose "My SSN is 123-45-6789, do you accept Medicare?"
    python demo.py --llm groq "How do I book an appointment?"     # needs `pip install groq` + GROQ_API_KEY
    python demo.py --llm claude "How do I book an appointment?"   # needs `pip install anthropic` + API key
    python demo.py                                                  # interactive mode
"""
from __future__ import annotations

import argparse
import json

from guardrails import ClaudeLLM, GroqLLM, GuardedAssistant, MockLLM


def show(result, verbose: bool) -> None:
    fired = [k for k, v in result.triggered.items() if v]
    print(f"\ndecision : {result.decision.value}")
    print(f"triggered: {', '.join(fired) or 'none'}")
    if result.modifications:
        print(f"modified : {', '.join(result.modifications)}")
    print(f"\n{result.response['answer']}")
    if result.response.get("citations"):
        print(f"\nsources  : {', '.join(result.response['citations'])}")
    for notice in result.response.get("notices", []):
        print(f"notice   : {notice}")
    if verbose:
        print("\n--- guardrail report ---")
        print(json.dumps({"input": result.input_report, "retrieval": result.retrieval_report,
                          "attempts": result.attempts}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--llm", choices=["mock", "groq", "claude"], default="mock")
    parser.add_argument("--verbose", "-v", action="store_true", help="print the full guardrail report")
    parser.add_argument("--audit-log", default="audit_log.jsonl", help="append-only audit log (redacted)")
    args = parser.parse_args()

    llm = {"mock": MockLLM, "groq": GroqLLM, "claude": ClaudeLLM}[args.llm]()
    assistant = GuardedAssistant(llm=llm, audit_path=args.audit_log)
    if args.question:
        show(assistant.run(args.question), args.verbose)
        return
    print("Riverside Community Clinic assistant (guarded). Empty line to quit.")
    while question := input("\n> ").strip():
        show(assistant.run(question), args.verbose)


if __name__ == "__main__":
    main()
