# Architecture

![Guardrail architecture](architecture.svg)

The same flow as a Mermaid diagram (renders on GitHub):

```mermaid
flowchart TD
    U([User message]) --> IG

    subgraph IG[Input guardrails]
        direction LR
        INJ[Prompt-injection detector<br/>weighted rules, noisy-OR score]
        TOP[Sensitive-topic classifier<br/>5 categories → action]
        PII[PII detector<br/>regex + validators]
    end

    IG --> R{Policy router<br/>escalate › block › refuse › route › continue}
    R -- crisis / emergency / legal --> T[Vetted response templates]
    R -- injection score ≥ 0.70 --> T
    R -- drug access --> T
    R -- continue: redacted question --> RET[Retriever<br/>TF-IDF, public docs only]
    RET --> SCAN[Indirect-injection scan<br/>quarantine poisoned docs]
    SCAN --> LLM[LLM, untrusted<br/>Mock / Scripted / Claude]

    LLM --> OG
    subgraph OG[Output guardrails, per attempt]
        direction LR
        S1[1. Leak detector<br/>raw text: canary, prompt n-grams, internal docs] --> S2[2. Schema<br/>strict Pydantic + citation check]
        S2 --> S3[3. Output PII<br/>allowlist public contacts, redact rest]
        S3 --> S4[4. Groundedness<br/>overlap, numbers, negation]
    end

    S2 -. invalid .-> RETRY[Regenerate once<br/>with validator feedback]
    S4 -. unsupported .-> RETRY
    RETRY -.-> LLM
    S1 -- leak --> BLOCK[Block<br/>security event, no retry]
    S4 -- pass --> ANS[Answer<br/>+ citations, disclaimer, notices]
    RETRY -- still failing --> FB[Safe fallback<br/>call reception]

    IG -.-> AUD[(Audit log<br/>hash + events, PII as type only)]
    SCAN -.-> AUD
    OG -.-> AUD
```

## Life of a request

1. **Input checks run together on the raw message.** The injection detector scores it, the topic classifier labels it, and the PII detector finds identifiers. Nothing has reached the model yet.
2. **The router applies one priority order.** Self-harm and medical emergencies are escalated first, ahead of the injection block, so a person in crisis always receives safety resources. Next come the injection block, refusal of drug-access requests, and hand-off of legal complaints. Everything else continues.
3. **PII is redacted** (`sita@example.com` → `[EMAIL]`) before the question is used for retrieval, sent to the model, or logged. High-risk types (SSN, card number) also produce a privacy notice for the user.
4. **Retrieval** returns public documents only. Each document was scanned for embedded instructions when the assistant started, and poisoned documents are dropped from the context. They are also excluded from the PII allowlist, so an attacker's email address planted in a document is not treated as "public".
5. **The model** gets a system prompt containing a canary token, the documents, and the question fenced off as untrusted text. It must return one JSON object.
6. **Output checks run on every attempt:** leakage on the raw text first (block, even if the output is malformed), then schema, then PII redaction, then groundedness on the redacted text. A schema or groundedness failure triggers one regeneration with the validator's feedback. If the second attempt also fails, the user gets a safe fallback.
7. **Post-processing:** medication or diagnosis questions get a disclaimer and `needs_human_followup = true`. Every stage appends to the audit trail.
