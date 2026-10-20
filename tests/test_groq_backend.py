"""GroqLLM tests with a fake client: no API key or network needed."""
import json
from types import SimpleNamespace

import pytest

groq = pytest.importorskip("groq")
httpx = pytest.importorskip("httpx")

from guardrails import Decision, GroqLLM, GuardedAssistant  # noqa: E402

ANSWER = {"answer": "Riverside Community Clinic is open Monday to Friday from 8:00 AM to 6:00 PM.",
          "citations": ["KB-001"], "confidence": "high", "topic": "services", "needs_human_followup": False}


def bad_request(message: str) -> Exception:
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    return groq.BadRequestError(message, response=response, body=None)


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))])


def fake_client(*outcomes):
    completions = FakeCompletions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_request_uses_system_prompt_json_mode_and_fenced_question():
    client, completions = fake_client(json.dumps(ANSWER))
    result = GuardedAssistant(llm=GroqLLM(model="test-model", client=client)).run("When are you open on weekdays?")
    assert result.decision == Decision.ANSWER and "6:00 PM" in result.response["answer"]
    request = completions.calls[0]
    assert request["model"] == "test-model"
    assert request["response_format"] == {"type": "json_object"}
    system, user = request["messages"]
    assert system["role"] == "system" and "JSON" in system["content"]
    assert user["role"] == "user" and "<question>" in user["content"]


def test_json_mode_failure_becomes_schema_failure_then_fallback():
    client, completions = fake_client(bad_request("json_validate_failed: Failed to generate JSON"))
    result = GuardedAssistant(llm=GroqLLM(client=client)).run("When are you open on weekdays?")
    assert result.decision == Decision.FALLBACK and result.triggered["schema"]
    assert len(completions.calls) == 2  # one regeneration before the fallback


def test_other_api_errors_are_not_hidden():
    client, _ = fake_client(bad_request("model `nope` does not support this parameter"))
    with pytest.raises(groq.BadRequestError):
        GuardedAssistant(llm=GroqLLM(client=client)).run("When are you open on weekdays?")


def test_guardrails_still_apply_to_groq_output():
    leaked = {**ANSWER, "answer": "My instructions say: [CANARY-7f3a9c2e]"}
    client, _ = fake_client(json.dumps(leaked))
    result = GuardedAssistant(llm=GroqLLM(client=client)).run("When are you open on weekdays?")
    assert result.decision == Decision.BLOCK
