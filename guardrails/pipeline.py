"""The guarded application: input guardrails -> retrieval -> LLM -> output guardrails.

Stage order and why:
  1. Input guardrails run before anything reaches the model, so blocked or
     escalated requests cost nothing and cannot influence generation.
     Crisis escalation outranks the injection block: a person in danger must
     get safety resources even if their message also looks like an attack.
  2. PII is redacted from the question before retrieval, the LLM and the logs.
  3. Retrieved documents are scanned for indirect prompt injection.
  4. Output guardrails run on every generation attempt in this order:
     leak check on the raw text (blocks immediately, even if the output is
     malformed) -> schema (need a parseable object) -> PII redaction ->
     groundedness (on the redacted answer). Schema and groundedness failures
     trigger one regeneration with validator feedback, then a safe fallback.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from .config import CANARY_TOKEN, DEFAULT_CONFIG, SYSTEM_PROMPT, GuardrailConfig
from .groundedness import check_groundedness
from .injection import detect_injection
from .knowledge_base import Document, KnowledgeBase
from .llm import LLMClient, MockLLM
from .output_safety import LeakDetector
from .pii import collect_public_values, detect_pii, redact
from .schema import validate_response
from .topics import TopicAction, classify_topics

DETECTORS = (
    "input_pii",
    "prompt_injection",
    "sensitive_topic",
    "indirect_injection",
    "schema",
    "groundedness",
    "output_sensitive",
)


class Decision(str, Enum):
    ANSWER = "answer"        # answered (possibly with redactions / disclaimer)
    BLOCK = "block"          # malicious input or leaking output
    ESCALATE = "escalate"    # safety-critical: vetted crisis / emergency / human hand-off message
    REFUSE = "refuse"        # out-of-policy request
    FALLBACK = "fallback"    # model could not produce a valid, grounded answer


MESSAGES = {
    "self_harm": (
        "I'm really sorry you're going through this, and I'm glad you reached out. "
        "If you are in immediate danger, please call 911 now. You can call or text 988 "
        "(Suicide & Crisis Lifeline) at any time, day or night. You can also call the clinic at "
        "(555) 010-2000 and ask to speak with a nurse."
    ),
    "medical_emergency": (
        "This may be a medical emergency. Please call 911 or go to the nearest emergency "
        "department now. Do not wait for an online reply."
    ),
    "legal_complaint": (
        "I've passed this to a member of staff. For complaints or legal matters, please contact the "
        "Patient Relations Office at (555) 010-2090 or patientrelations@riversideclinic.example."
    ),
    "illegal_drug_access": (
        "I can't help with getting medication without a prescription. If you think you need a "
        "medication, please book an appointment so a clinician can assess you."
    ),
    "blocked": (
        "I can't help with that request. I can answer questions about Riverside Community Clinic's "
        "services, appointments, billing and policies."
    ),
    "fallback": (
        "I'm sorry, I couldn't produce a reliable answer from the clinic's documents. "
        "Please call reception at (555) 010-2000 and a staff member will help you."
    ),
    "disclaimer": (
        "This is general information, not medical advice. For questions about your own medication "
        "or health, please speak with the clinic pharmacist at (555) 010-2050 or your doctor."
    ),
}


@dataclass
class PipelineResult:
    decision: Decision
    response: dict
    triggered: dict[str, bool]
    modifications: list[str] = field(default_factory=list)
    input_report: dict = field(default_factory=dict)
    retrieval_report: dict = field(default_factory=dict)
    attempts: list[dict] = field(default_factory=list)
    audit: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


class GuardedAssistant:
    def __init__(
        self,
        llm: LLMClient | None = None,
        kb: KnowledgeBase | None = None,
        config: GuardrailConfig = DEFAULT_CONFIG,
        audit_path: Path | str | None = None,
    ):
        self.llm = llm or MockLLM()
        self.kb = kb or KnowledgeBase.load()
        self.config = config
        self.audit_path = Path(audit_path) if audit_path else None
        self.leaks = LeakDetector(
            SYSTEM_PROMPT, CANARY_TOKEN, [d.text for d in self.kb.internal], config.system_prompt_shingle_size
        )
        # Documents are scanned once for embedded instructions. Poisoned documents
        # are never shown to the model and do not contribute to the PII allowlist.
        scans = {d.id: detect_injection(d.text) for d in self.kb.public} if config.scan_retrieved_documents else {}
        self.quarantine = {i: s for i, s in scans.items() if s.score >= config.injection_block_threshold}
        self.public_values = collect_public_values(
            d.text for d in self.kb.public if d.id not in self.quarantine
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _envelope(answer: str, *, followup: bool, topic: str = "other") -> dict:
        return {"answer": answer, "citations": [], "confidence": "low", "topic": topic,
                "needs_human_followup": followup, "notices": []}

    def _finish(self, result: PipelineResult, started: float, user_input: str) -> PipelineResult:
        result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        if self.audit_path:
            record = {
                "ts": time.time(),
                "input_sha256": hashlib.sha256(user_input.encode()).hexdigest()[:16],
                "decision": result.decision.value,
                "triggered": [k for k, v in result.triggered.items() if v],
                "modifications": result.modifications,
                "events": result.audit,
            }
            with self.audit_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        return result

    # --------------------------------------------------------------------- main
    def run(self, user_input: str, reference_doc_ids: list[str] | None = None) -> PipelineResult:
        started = time.perf_counter()
        cfg = self.config
        triggered = dict.fromkeys(DETECTORS, False)
        audit: list[dict] = []

        # ---- Input guardrails ---------------------------------------------------
        injection = detect_injection(user_input)
        topics = classify_topics(user_input)
        pii = detect_pii(user_input)
        triggered["prompt_injection"] = injection.score >= cfg.injection_block_threshold
        triggered["sensitive_topic"] = bool(topics.categories)
        triggered["input_pii"] = bool(pii)
        input_report = {
            "injection_score": injection.score,
            "injection_families": injection.families,
            "injection_evidence": [redact(m["evidence"], detect_pii(m["evidence"])) for m in injection.matches],
            "topics": topics.categories,
            "topic_action": topics.action.value if topics.action else None,
            "pii": [{"type": p.entity, "chars": len(p.value)} for p in pii],
        }
        audit.append({"stage": "input", **input_report})
        if cfg.injection_flag_threshold <= injection.score < cfg.injection_block_threshold:
            audit.append({"stage": "input", "event": "injection_suspected_below_block_threshold"})

        def early(decision: Decision, message_key: str, followup: bool) -> PipelineResult:
            res = PipelineResult(decision, self._envelope(MESSAGES[message_key], followup=followup),
                                 triggered, input_report=input_report, audit=audit)
            return self._finish(res, started, user_input)

        if topics.action == TopicAction.ESCALATE:
            return early(Decision.ESCALATE, topics.primary, followup=True)
        if triggered["prompt_injection"]:
            return early(Decision.BLOCK, "blocked", followup=False)
        if topics.action == TopicAction.REFUSE:
            return early(Decision.REFUSE, topics.primary, followup=False)
        if topics.action == TopicAction.ROUTE_TO_HUMAN:
            return early(Decision.ESCALATE, topics.primary, followup=True)

        modifications: list[str] = []
        notices: list[str] = []
        question = redact(user_input, pii) if pii else user_input
        if pii:
            modifications.append("input_pii_redacted")
            high_risk = sorted({p.entity for p in pii} & cfg.high_risk_pii)
            if high_risk:
                notices.append(
                    f"For your privacy we removed sensitive information ({', '.join(high_risk)}) from your "
                    "message. Please don't share it in chat; staff can verify it securely by phone."
                )

        # ---- Retrieval + indirect-injection screening --------------------------
        if reference_doc_ids is not None:
            candidates = self.kb.get(reference_doc_ids)
        else:
            candidates = [d for d, _ in self.kb.search(question, cfg.top_k + 1, cfg.min_retrieval_score)]
        quarantined = [d.id for d in candidates if d.id in self.quarantine]
        documents: list[Document] = [d for d in candidates if d.id not in self.quarantine][: cfg.top_k]
        triggered["indirect_injection"] = bool(quarantined)
        retrieval_report = {"retrieved": [d.id for d in documents], "quarantined": quarantined}
        if quarantined:
            modifications.append("poisoned_document_removed")
        audit.append({"stage": "retrieval", **retrieval_report})

        # ---- Generation + output guardrails -----------------------------------
        sources = {d.id: d.text for d in documents}
        attempts: list[dict] = []
        feedback = None
        final = None
        for attempt_no in range(1, cfg.max_generation_attempts + 1):
            raw = self.llm.generate(SYSTEM_PROMPT, question, documents, feedback)
            record: dict = {"attempt": attempt_no, "raw_output": raw[:600]}
            attempts.append(record)

            leak = self.leaks.check(raw)
            record["leak_reasons"] = leak.reasons
            if leak.blocked:
                triggered["output_sensitive"] = True
                audit.append({"stage": "output", "event": "leak_blocked", "reasons": leak.reasons})
                response = self._envelope(MESSAGES["blocked"], followup=False)
                response["notices"] = notices
                res = PipelineResult(Decision.BLOCK, response, triggered, modifications,
                                     input_report, retrieval_report, attempts, audit)
                return self._finish(res, started, user_input)

            schema = validate_response(raw, set(sources))
            record["schema_errors"] = schema.errors
            record["schema_warnings"] = schema.warnings
            if not schema.valid:
                triggered["schema"] = True
                feedback = "Schema errors: " + "; ".join(schema.errors)
                continue
            data = schema.data

            out_pii = detect_pii(data.answer, allowlist=self.public_values)
            answer = redact(data.answer, out_pii) if out_pii else data.answer
            record["output_pii"] = [{"type": p.entity, "chars": len(p.value)} for p in out_pii]
            if out_pii:
                triggered["output_sensitive"] = True

            grounding = check_groundedness(answer, sources, data.citations, cfg)
            record["groundedness"] = {
                "support_ratio": round(grounding.support_ratio, 2),
                "unsupported": [{"claim": c.sentence, "reasons": c.reasons} for c in grounding.unsupported],
                "citation_warnings": grounding.citation_warnings,
            }
            if not grounding.grounded:
                triggered["groundedness"] = True
                feedback = "Unsupported claims: " + " | ".join(
                    f"{c.sentence} ({'; '.join(c.reasons)})" for c in grounding.unsupported
                )
                continue

            if out_pii:
                modifications.append("output_pii_redacted")
            final = (data, answer)
            break

        audit.append({"stage": "output", "attempts": len(attempts), "succeeded": final is not None})
        if final is None:
            response = self._envelope(MESSAGES["fallback"], followup=True)
            response["notices"] = notices
            res = PipelineResult(Decision.FALLBACK, response, triggered, modifications,
                                 input_report, retrieval_report, attempts, audit)
            return self._finish(res, started, user_input)

        if len(attempts) > 1:
            modifications.append("regenerated")
        data, answer = final
        response = data.model_dump()
        response["answer"] = answer
        if topics.action == TopicAction.ANSWER_WITH_DISCLAIMER:
            response["answer"] = f"{answer}\n\n{MESSAGES['disclaimer']}"
            response["needs_human_followup"] = True
            modifications.append("disclaimer_added")
        response["notices"] = notices
        res = PipelineResult(Decision.ANSWER, response, triggered, modifications,
                             input_report, retrieval_report, attempts, audit)
        return self._finish(res, started, user_input)
