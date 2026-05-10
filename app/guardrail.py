import re


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def check_guardrails(query: str):
    query_norm = normalize(query)

    # ------------------------------------------------------------
    # STEP 1 — ALLOW SAFE / VICTIM / LEGAL CONTEXT
    # ------------------------------------------------------------
    safe_context_keywords = [
        "after", "what to do", "legal", "law", "punishment",
        "rights", "penalty", "complaint", "report", "fir",
        "police complaint", "legal action", "procedure"
    ]

    crime_words = [
        "rob", "robbery", "steal", "theft",
        "hack", "fraud", "scam",
        "kill", "murder", "attack"
    ]

    # If query contains crime word BUT also safe/legal context → ALLOW
    if any(c in query_norm for c in crime_words):
        if any(s in query_norm for s in safe_context_keywords):
            return True, None  # ✅ Allow

    # ------------------------------------------------------------
    # STEP 2 — BLOCK CLEAR MALICIOUS INTENT
    # ------------------------------------------------------------
    harmful_patterns = [
        "how to rob", "how to steal", "how to hack",
        "how to commit", "illegal way", "break law",
        "get away with", "without getting caught",
        "without getting arrested", "escape police",
        "evade police", "avoid arrest"
    ]

    for pattern in harmful_patterns:
        if pattern in query_norm:
            return False, (
                "I can’t assist with illegal or harmful activities. "
                "I can explain legal consequences or your rights under the law."
            )

    # ------------------------------------------------------------
    # STEP 3 — REGEX (STRONG INTENT DETECTION)
    # ------------------------------------------------------------
    harmful_regex = [
        r"how to .*rob",
        r"how to .*steal",
        r"how to .*hack",
        r"how to .*kill",
        r"how to .*get away",
        r"without getting (caught|arrested)"
    ]

    for pattern in harmful_regex:
        if re.search(pattern, query_norm):
            return False, (
                "I can’t assist with illegal or harmful activities. "
                "I can explain legal consequences or your rights instead."
            )

    # ------------------------------------------------------------
    # STEP 4 — DEFAULT ALLOW
    # ------------------------------------------------------------
    return True, None