# ============================================================
# SECTION  — FEW-SHOT IN-CONTEXT LEARNING (MISTRAL 7B)
# ============================================================

from llama_cpp import Llama
from app.config import LLM_MODEL_PATH

print("Loading Mistral 7B...")

llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=2048,
    n_threads=8
)

print("Mistral loaded successfully.")


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(query, context_df, intent):

    print("Running Reasoning Module...")

    # -------------------------
    # Guard Clauses
    # -------------------------
    if context_df.empty:
        return "No relevant legal information found."

    if "text" not in context_df.columns:
        return "Invalid context format."

    # -------------------------
    # Step 1 — Prepare context
    # -------------------------
    texts = context_df["text"].tolist()[:5]
    context = "\n\n".join(texts)

    # -------------------------
    # Step 2 — Intent-based instruction
    # -------------------------
    if intent == "summarization":
        instruction = "Summarize the legal content clearly and concisely."

    elif intent == "legal_reasoning":
        instruction = """Identify the legal issue, cite relevant constitutional articles, and explain clearly in a structured manner:

1. Relevant Articles (mention article numbers)
2. Legal Rights (explain clearly)
3. Conclusion (state if action is lawful or not)

Always explicitly mention article numbers in the answer.
"""

    elif intent == "explanation":
        instruction = "Explain in simple terms."

    elif intent == "legal_lookup":
        instruction = "Provide accurate legal information based only on the context."

    else:
        instruction = "Answer based on the provided context."

    # -------------------------
    # Step 3 — Prompt (ICL)
    # -------------------------
    prompt = f"""
You are an AI legal assistant trained on the Indian Constitution.

TASK:
{instruction}

STRICT RULES:
- Use ONLY the provided context
- Answer ONLY the question asked
- DO NOT generate additional questions or examples
- STOP after giving the answer
- Base your answer strictly on constitutional principles in the context
- Do NOT make assumptions
- Do NOT give personal legal advice
- If answer is not in context, say: "Not found in provided legal documents"
- Do NOT continue conversation
- Follow this exact structure: Relevant Articles, Legal Rights, Conclusion

----------------------------------

### EXAMPLE 1 — FACTUAL
Context:
Article 21 guarantees right to life and personal liberty.

Question:
What is Article 21?

Answer:
Article 21 guarantees the right to life and personal liberty.

----------------------------------

### EXAMPLE 2 — EXPLANATION
Context:
Article 19 ensures freedom of speech.

Question:
Explain Article 19 in simple terms.

Answer:
Article 19 gives people the right to freely express their thoughts and opinions.

----------------------------------

### EXAMPLE 3 — SCENARIO
Context:
Article 21 guarantees right to life and personal liberty.
Article 22 provides protection against arbitrary arrest.

Question:
Police arrested a person without a warrant and denied communication. Is this legal?

Answer:
Relevant Articles:
- Article 21: Right to life and personal liberty
- Article 22(1): Right to be informed of grounds of arrest and consult a lawyer
- Article 22(2): Right to be produced before a magistrate within 24 hours

Legal Rights:
A person who is arrested must be informed of the reason for arrest and has the right to consult a legal practitioner. They must also be produced before a magistrate within 24 hours.

Conclusion:
If these conditions are not met, the arrest may be unlawful and violate constitutional rights.

----------------------------------

### EXAMPLE 4 — SAFE ADVICE
Context:
Article 21 guarantees right to life.

Question:
What should I do if my rights are violated?

Answer:
If your rights are violated, you may consider seeking legal assistance or consulting a qualified lawyer. This system provides informational guidance only.

----------------------------------

### FINAL ANSWER (ONLY ONE ANSWER, DO NOT CONTINUE):

Context:
{context}

Question:
{query}

Answer:
"""

    # -------------------------
    # Step 4 — LLM CALL
    # -------------------------
    response = llm(
        prompt,
        max_tokens=256,
        temperature=0.1,
        stop=["Question:", "###"]
    )

    answer = response["choices"][0]["text"].strip()

    # -------------------------
    # Step 5 — Clean output
    # -------------------------
    if answer.lower().startswith("answer:"):
        answer = answer[len("answer:"):].strip()

    # Remove unwanted continuation
    for marker in ["Question:", "###"]:
        if marker in answer:
            answer = answer.split(marker)[0].strip()

    if not answer:
        return "No answer generated."

    return answer