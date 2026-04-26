"""
ui/app.py
---------
Gradio chat interface for the RAG chatbot.

Run:
    python -m ui.app
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

CSS = """
.source-box {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px;
    font-size: 0.9em;
}
.refusal { color: #e53e3e; font-style: italic; }
footer { display: none !important; }
"""

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from generation.generator import RAGGenerator

# Singleton generator (loaded once)
_generator: RAGGenerator | None = None


def get_generator() -> RAGGenerator:
    global _generator
    if _generator is None:
        _generator = RAGGenerator()
    return _generator


def chat(message: str, history: list):
    if not message.strip():
        yield history, "", ""
        return

    gen = get_generator()
    hist_tuples = [(h["role"] == "user" and h["content"], h["role"] == "assistant" and h["content"]) 
                   for h in history if h.get("role") in ("user", "assistant")]
    hist_tuples = [(u, a) for u, a in zip(
        [h["content"] for h in history if h["role"] == "user"],
        [h["content"] for h in history if h["role"] == "assistant"],
    )] if history else []

    result = gen.answer(message, history=hist_tuples or None)
    answer = result["answer"]
    sources = result["sources"]

    if sources and not result["is_refusal"]:
        source_lines = ["**Retrieved Sources:**\n"]
        seen = {}
        for i, chunk in enumerate(sources, start=1):
            meta = chunk.get("metadata", {})
            title = meta.get("title") or meta.get("source", f"Document {i}")
            score = chunk.get("score", 0)
            if title not in seen:
                seen[title] = score
                source_lines.append(f"**[{len(seen)}]** {title} (similarity: {score:.3f})")
        source_text = "\n".join(source_lines)
    else:
        source_text = "No sources retrieved."

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    partial = ""
    for word in answer.split(" "):
        partial += word + " "
        new_history[-1]["content"] = partial.strip()
        yield new_history, source_text, ""

    yield new_history, source_text, ""

def build_ui() -> gr.Blocks:

    with gr.Blocks(title="RAG Chatbot") as demo:

        gr.Markdown(
            """
# 📚 RAG Chatbot
**Grounded answers with source citations** — powered by Retrieval-Augmented Generation.
All answers are derived exclusively from the indexed knowledge base.
"""
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=520,
                    show_label=False,
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask a question about your documents…",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

            with gr.Column(scale=1):
                sources_box = gr.Markdown(
                    label="Sources",
                    value="*Sources will appear here after each answer.*",
                    elem_classes=["source-box"],
                )

        gr.Markdown(
            """
---
**How to use:** Type your question and press Send. The chatbot retrieves relevant 
passages from the knowledge base and generates a grounded answer with citations.
If the answer cannot be found in the documents, it will say so explicitly.
"""
        )

        # Example queries
        gr.Examples(
            examples=[
                ["What is Retrieval-Augmented Generation?"],
                ["How does BERT differ from GPT-2?"],
                ["What are the limitations of vector similarity search?"],
                ["What is the capital of Mars?"],  # out-of-context trigger
            ],
            inputs=msg_box,
            label="Example queries",
        )

        # State
        status = gr.Textbox(visible=False)

        # Event wiring
        send_btn.click(
            fn=chat,
            inputs=[msg_box, chatbot],
            outputs=[chatbot, sources_box, status],
        )
        msg_box.submit(
            fn=chat,
            inputs=[msg_box, chatbot],
            outputs=[chatbot, sources_box, status],
        )
        clear_btn.click(
            fn=lambda: ([], "*Sources will appear here after each answer.*"),
            outputs=[chatbot, sources_box],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CSS,
    )
