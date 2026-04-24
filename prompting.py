import logging

logger = logging.getLogger(__name__)


def generate_prompt_template_v1(query, context):
    """Basic prompt template: inject context and ask for answer."""
    prompt = f"""
Use the following context to answer the query. If the context does not contain relevant information, say 'I don't know'.

Context:
{context}

Query: {query}

Answer:
"""
    return prompt


def generate_prompt_template_v2(query, context):
    """Improved prompt: add hallucination control and structure."""
    prompt = f"""
You are a careful, domain-aware assistant. Use ONLY the provided context to answer the query. Do not invent facts or use outside knowledge. If the context does not answer the question, respond exactly with 'Insufficient information in context'.

Context:
{context}

Query: {query}

Provide a concise, factual answer and cite the context if possible:
"""
    return prompt


def generate_prompt_template_v4(query, context):
    """High-accuracy prompt: emphasize grounding, evidence, and refusal for unknowns."""
    prompt = f"""
You are an AI assistant returning factual, evidence-based answers from the supplied context only.
1. Read the context carefully.
2. Answer the query using only the context.
3. Quote or cite the context if relevant.
4. If the context does not contain enough information, respond exactly: 'Insufficient information in context'.
5. Do not guess or fabricate any details.

Context:
{context}

Query: {query}

Answer:
"""
    return prompt


def generate_prompt_template_v3(query, context):
    """Advanced prompt: include reasoning steps and citation."""
    prompt = f"""
Task: Answer the query using ONLY the provided context. Follow these steps:
1. Identify relevant information from the context.
2. Reason step-by-step based on that information.
3. Provide a final answer with citations (e.g., quote from context).
4. If no relevant info, say 'No relevant information found'.

Context:
{context}

Query: {query}

Response:
"""
    return prompt


def simulate_llm_response(prompt, max_words=100):
    """Simulate an LLM response for demonstration purposes."""
    text = prompt.lower()
    if 'economic growth' in text:
        return "Based on context, GDP growth is projected to moderate to 4.0% in 2025."
    elif 'budget allocation' in text:
        return "Government is committed to aligning expenditures with fiscal realities."
    elif 'fiscal policy' in text:
        return "The fiscal framework reflects amendments to the Fiscal Responsibility Act."
    elif 'football' in text or 'sports' in text:
        return "No relevant information found in context."
    elif 'current budget' in text:
        return "Insufficient information in context."
    elif 'insufficient information in context' in prompt:
        return "Insufficient information in context."
    elif "i don't know" in prompt:
        return "I don't know."
    else:
        return "No relevant information found in context."
