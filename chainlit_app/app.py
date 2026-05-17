"""Chainlit UI entry point — connects to FastAPI backend via httpx."""

from __future__ import annotations

import os

import chainlit as cl
import httpx

# Get FastAPI base URL from environment
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")


@cl.on_chat_start
async def on_chat_start() -> None:
    """Display welcome message with available commands on chat start."""
    welcome = (
        "# 📚 Multi-Document RAG System\n\n"
        "Welcome! I can help you analyze and compare academic research papers.\n\n"
        "## Available Commands\n"
        "| Command | Description |\n"
        "|---------|-------------|\n"
        "| **/docs** | List all ingested documents |\n"
        "| **/compare** | Compare two or more documents |\n"
        "| **/eval** | Run RAGAS evaluation on sample questions |\n"
        "| **/help** | Show this menu again |\n\n"
        "To ingest a new document, simply attach it to a message and send it!\n"
        "Or just type a question to query your documents!\n"
    )
    await cl.Message(content=welcome).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming chat messages and route to appropriate actions.

    Dispatches based on command prefix or treats as a query.

    Args:
        message: The incoming Chainlit message.
    """
    text = message.content.strip()

    # Handle file uploads
    if message.elements:
        await _handle_file_upload(message.elements)
        return

    # Route commands
    if text.lower().startswith("/help"):
        await on_chat_start()
        return

    if text.lower().startswith("/docs"):
        await _handle_list_docs()
        return

    if text.lower().startswith("/compare"):
        await _handle_compare(text)
        return

    if text.lower().startswith("/eval"):
        await _handle_eval(text)
        return

    # Default: treat as a query
    if text:
        await _handle_query(text)


async def _handle_file_upload(elements: list) -> None:
    """Process uploaded files by sending them to the ingest endpoint.

    Args:
        elements: List of Chainlit file elements.
    """
    for element in elements:
        if not hasattr(element, "path") or not element.path:
            continue

        filename = getattr(element, "name", "unknown_file")
        await cl.Message(content=f"⏳ Ingesting **{filename}**...").send()

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                with open(element.path, "rb") as f:
                    response = await client.post(
                        f"{FASTAPI_BASE_URL}/api/v1/ingest",
                        files={"file": (filename, f)},
                        data={"source": "chainlit_upload"},
                    )

            if response.status_code == 200:
                result = response.json()
                msg = (
                    f"✅ **Ingested {result['filename']}**\n\n"
                    f"- **Doc ID:** `{result['doc_id']}`\n"
                    f"- **Chunks Created:** {result['chunks_created']}\n"
                    f"- **Doc Type:** {result['metadata'].get('doc_type', 'unknown')}\n"
                    f"- **Status:** {result['status']}"
                )
            else:
                error = response.json().get("detail", "Unknown error")
                msg = f"❌ Failed to ingest **{filename}**: {error}"

            await cl.Message(content=msg).send()

        except Exception as e:
            await cl.Message(
                content=f"❌ Error uploading **{filename}**: {str(e)}"
            ).send()


async def _handle_list_docs() -> None:
    """Fetch and display all ingested documents."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{FASTAPI_BASE_URL}/api/v1/documents")

        if response.status_code == 200:
            data = response.json()
            docs = data.get("documents", [])
            total = data.get("total_count", 0)

            if not docs:
                await cl.Message(content="📭 No documents ingested yet.").send()
                return

            table = "# 📄 Ingested Documents\n\n"
            table += "| # | Filename | Doc ID | Type | Source | Chunks |\n"
            table += "|---|----------|--------|------|--------|--------|\n"

            for i, doc in enumerate(docs, 1):
                doc_id_short = doc["doc_id"][:8] + "..."
                table += (
                    f"| {i} | {doc['filename']} | `{doc_id_short}` | "
                    f"{doc['doc_type']} | {doc['source']} | "
                    f"{doc['chunk_count']} |\n"
                )

            table += f"\n**Total:** {total} documents"
            await cl.Message(content=table).send()
        else:
            await cl.Message(content="❌ Failed to fetch documents.").send()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()


async def _handle_query(question: str) -> None:
    """Send a query to the RAG endpoint and display the result.

    Args:
        question: The user's question text.
    """
    thinking_msg = cl.Message(content="🤔 Searching documents and generating answer...")
    await thinking_msg.send()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{FASTAPI_BASE_URL}/api/v1/query",
                json={
                    "question": question,
                    "mode": "standard",
                    "top_k": 5,
                },
            )

        if response.status_code == 200:
            result = response.json()
            answer = result["answer"]
            sources = result.get("sources", [])
            model = result.get("model_used", "unknown")
            latency = result.get("latency_ms", 0)
            relevance = result.get("relevance_score", 0.0)

            # Build response
            msg = f"## Answer\n\n{answer}\n\n"

            # Add sources as expandable elements
            if sources:
                msg += "---\n\n<details>\n<summary>📖 Sources (click to expand)</summary>\n\n"
                for i, src in enumerate(sources, 1):
                    score_pct = f"{src['score']:.1%}" if src.get("score") else "N/A"
                    msg += (
                        f"**Source {i}:** {src['filename']} "
                        f"(chunk {src['chunk_index']}, score: {score_pct})\n"
                        f"> {src['text'][:200]}...\n\n"
                    )
                msg += "</details>\n\n"

            msg += (
                f"---\n"
                f"🤖 *Model: {model} | "
                f"⏱️ {latency}ms | "
                f"📊 Relevance: {relevance:.1%}*"
            )

            await cl.Message(content=msg).send()
        else:
            error = response.json().get("detail", "Unknown error")
            await cl.Message(content=f"❌ Query failed: {error}").send()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()


async def _handle_compare(text: str) -> None:
    """Handle cross-document comparison command.

    Prompts user for doc IDs and aspect if not provided inline.

    Args:
        text: The full command text.
    """
    parts = text.split()

    if len(parts) < 3:
        await cl.Message(
            content=(
                "📋 **Compare Documents**\n\n"
                "Usage: `/compare <doc_id_1> <doc_id_2> [aspect]`\n\n"
                "Example:\n"
                "```\n"
                "/compare abc123 def456 methodology\n"
                "```\n\n"
                "Use `/docs` to see available document IDs."
            )
        ).send()
        return

    doc_ids = [parts[1], parts[2]]
    aspect = " ".join(parts[3:]) if len(parts) > 3 else "general"

    await cl.Message(content="🔄 Comparing documents...").send()

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{FASTAPI_BASE_URL}/api/v1/query/compare",
                json={
                    "question": f"Compare these documents focusing on {aspect}",
                    "doc_ids": doc_ids,
                    "aspect": aspect,
                },
            )

        if response.status_code == 200:
            result = response.json()
            comparison = result.get("comparison", "No comparison generated.")
            agreements = result.get("agreements", [])
            contradictions = result.get("contradictions", [])
            model = result.get("model_used", "unknown")
            latency = result.get("latency_ms", 0)

            msg = f"## 🔍 Document Comparison\n\n{comparison}\n\n"

            if agreements:
                msg += "### ✅ Agreements\n"
                for a in agreements:
                    msg += f"- {a}\n"
                msg += "\n"

            if contradictions:
                msg += "### ⚡ Contradictions / Differences\n"
                for c in contradictions:
                    msg += f"- {c}\n"
                msg += "\n"

            msg += f"---\n🤖 *Model: {model} | ⏱️ {latency}ms*"

            await cl.Message(content=msg).send()
        else:
            error = response.json().get("detail", "Unknown error")
            await cl.Message(content=f"❌ Comparison failed: {error}").send()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()


async def _handle_eval(text: str) -> None:
    """Handle RAGAS evaluation command.

    Args:
        text: The full command text.
    """
    await cl.Message(content="📊 Running RAGAS evaluation... This may take a few minutes.").send()

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{FASTAPI_BASE_URL}/api/v1/evaluate",
                json={"sample_size": 5},
            )

        if response.status_code == 200:
            result = response.json()
            status = result.get("status", "unknown")
            metrics = result.get("metrics")

            if status == "completed" and metrics:
                table = (
                    "## 📊 RAGAS Evaluation Results\n\n"
                    "| Metric | Score |\n"
                    "|--------|-------|\n"
                    f"| Faithfulness | {metrics['faithfulness']:.4f} |\n"
                    f"| Answer Relevancy | {metrics['answer_relevancy']:.4f} |\n"
                    f"| Context Precision | {metrics['context_precision']:.4f} |\n"
                    f"| Context Recall | {metrics['context_recall']:.4f} |\n"
                    f"| Answer Correctness | {metrics['answer_correctness']:.4f} |\n\n"
                    f"**Eval ID:** `{metrics.get('eval_id', 'N/A')}`\n"
                    f"**Questions Evaluated:** {metrics.get('question_count', 0)}"
                )
                await cl.Message(content=table).send()
            else:
                await cl.Message(
                    content=(
                        f"📊 Evaluation started (ID: `{result.get('eval_id', 'N/A')}`).\n"
                        f"Status: **{status}**\n\n"
                        f"Use the API to check: `GET /api/v1/evaluate/{result.get('eval_id', '')}`"
                    )
                ).send()
        else:
            error = response.json().get("detail", "Unknown error")
            await cl.Message(content=f"❌ Evaluation failed: {error}").send()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()
