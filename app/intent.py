def classify_intent(query: str) -> str:
    query = query.lower()

    # SCENARIO / LEGAL REASONING (MOST IMPORTANT)
    if any(word in query for word in [
        "arrest", "police", "illegal", "lawful", "rights",
        "detained", "custody", "case", "crime"
    ]):
        return "legal_reasoning"

    elif any(word in query for word in ["punishment", "penalty", "fine", "sentence"]):
        return "legal_reasoning"

    elif any(word in query for word in ["summarize", "summary", "brief"]):
        return "summarization"

    elif any(word in query for word in ["article", "section"]):
        return "legal_lookup"

    elif any(word in query for word in ["explain", "what is", "define", "meaning"]):
        return "explanation"

    else:
        return "general"