"""LangChain prompt templates for RAG, comparison, routing, and self-correction."""

from langchain_core.prompts import ChatPromptTemplate

# ── Standard RAG Prompt ──────────────────────────────────────────────

STANDARD_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an academic research assistant. "
            "Answer questions ONLY using the provided context. "
            "Always cite the source document and page for each claim. "
            'If the answer is not in the context, say exactly: '
            '"This information is not available in the provided documents."',
        ),
        (
            "human",
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Provide a detailed, well-structured answer with citations.",
        ),
    ]
)

# ── Cross-Document Comparison Prompt ─────────────────────────────────

CROSS_DOC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at comparing and contrasting research papers. "
            "Analyze the provided excerpts from multiple documents. "
            "Structure your response with these sections:\n"
            "1. Overview of each document's position\n"
            "2. Key Agreements\n"
            "3. Key Contradictions or Differences\n"
            "4. Your synthesis and conclusion\n\n"
            "Always cite specific document names for every claim. "
            "Also return two JSON arrays at the end of your response:\n"
            'AGREEMENTS_JSON: ["agreement1", "agreement2", ...]\n'
            'CONTRADICTIONS_JSON: ["contradiction1", "contradiction2", ...]',
        ),
        (
            "human",
            "{formatted_doc_contexts}\n\n"
            "Question: {question}\n"
            "Aspect to focus on: {aspect}",
        ),
    ]
)

# ── Query Classification Prompt ──────────────────────────────────────

QUERY_CLASSIFICATION_PROMPT_TEMPLATE = (
    "You are a query classifier for an academic research RAG system.\n\n"
    "Classify the following user question for routing.\n\n"
    "Available documents:\n{available_documents}\n\n"
    "Question: {question}\n\n"
    "Return valid JSON only with this exact structure:\n"
    '{{\n'
    '  "requires_comparison": true/false,\n'
    '  "mode": "standard" or "reasoning",\n'
    '  "mentioned_docs": [],\n'
    '  "reasoning": "brief explanation"\n'
    '}}\n\n'
    "Rules:\n"
    '- If the question uses words like "compare", "contrast", '
    '"difference between", "versus", "vs" → requires_comparison=true, mode="reasoning"\n'
    '- If the question needs deep analysis, multi-step reasoning, or synthesis → mode="reasoning"\n'
    '- If the question is a simple factual lookup → mode="standard"\n'
    "- If specific document names are mentioned, list their doc_ids in mentioned_docs\n\n"
    "Return ONLY the JSON, no other text."
)

# Create a simple string template for format() usage
QUERY_CLASSIFICATION_PROMPT = QUERY_CLASSIFICATION_PROMPT_TEMPLATE

# ── Relevance Check Prompt ───────────────────────────────────────────

RELEVANCE_CHECK_PROMPT_TEMPLATE = (
    "You are a relevance evaluator for a RAG system.\n\n"
    "Score how relevant the following retrieved chunks are to answering "
    "the user's question.\n\n"
    "Question: {question}\n\n"
    "Retrieved chunks:\n{context}\n\n"
    "Return valid JSON only:\n"
    '{{\n'
    '  "relevance_score": 0.0 to 1.0,\n'
    '  "reason": "brief explanation of the score"\n'
    '}}\n\n'
    "Scoring guide:\n"
    "- 0.0-0.3: Chunks are completely unrelated to the question\n"
    "- 0.3-0.5: Chunks are tangentially related but don't answer the question\n"
    "- 0.5-0.7: Chunks contain some relevant information\n"
    "- 0.7-0.9: Chunks are highly relevant and can mostly answer the question\n"
    "- 0.9-1.0: Chunks directly and completely address the question\n\n"
    "Return ONLY the JSON, no other text."
)

RELEVANCE_CHECK_PROMPT = RELEVANCE_CHECK_PROMPT_TEMPLATE

# ── Query Rewrite Prompt ─────────────────────────────────────────────

QUERY_REWRITE_PROMPT_TEMPLATE = (
    "You are a search query optimizer for an academic research database.\n\n"
    "The following search query returned poor results.\n\n"
    "Original query: {question}\n"
    "Reason for poor results: {reason}\n\n"
    "Rewrite this search query to improve document retrieval. "
    "Make it more specific and academically precise. "
    "Use relevant technical terminology.\n\n"
    "Return ONLY the rewritten query, nothing else. "
    "Do not add explanations or quotes around it."
)

QUERY_REWRITE_PROMPT = QUERY_REWRITE_PROMPT_TEMPLATE
