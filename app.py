"""
Gradio UI for Article Q&A system.

Two tabs: Setup (API key) and Article QA (query interface).
"""

import os

import gradio as gr
from dotenv import load_dotenv

from src.database import MultimodalDB
from src.LLM_usage import ArticleQABot
from src.retriever import MultimodalRetriever, build_filter

# Load .env
load_dotenv()

# Global components
bot = None
retriever = None

# Available categories (from The Batch navigation)
CATEGORIES = [
    "Weekly Issues", "Andrew's Letters", "Data Points", "ML Research",
    "Business", "Science", "Culture", "Hardware", "AI Careers"
]


def initialize_bot(api_key):
    """Initialize the Q&A bot with API key."""
    global bot, retriever

    if not api_key:
        return "Error: Please provide a Groq API key"

    os.environ["GROQ_API_KEY"] = api_key

    # Initialize components (only if not already initialized)
    if retriever is None:
        db = MultimodalDB(persist_directory="data/qdrant_db")
        retriever = MultimodalRetriever(
            db=db,
            bm25_path="data/cache/bm25_index.pkl",
            tokens_path="data/cache/tokenized_docs.pkl"
        )

    if bot is None:
        bot = ArticleQABot(retriever=retriever)

    return "✅ Article Q&A Bot initialized successfully!"


def answer_question(query, categories, date_start, date_end, n_results, use_hybrid):
    """Answer user question with optional filters."""
    global bot

    if bot is None:
        return "Error: Bot not initialized. Set API key in Setup tab.", "", []

    if not query.strip():
        return "Please enter a question.", "", []

    # Build filter
    filter_obj = None
    if categories or date_start or date_end:
        filter_obj = build_filter(
            categories=categories if categories else None,
            date_start=date_start if date_start else None,
            date_end=date_end if date_end else None
        )

    # Generate answer
    try:
        result = bot.answer_question(
            query=query,
            n_results=n_results,
            where=filter_obj,
            use_hybrid=use_hybrid
        )

        # Format sources
        sources_text = "\n\n".join([
            f"📄 **{src['title']}**\n"
            f"   📅 {src['date']} | 🏷️ {', '.join(src['categories'])}\n"
            f"   🔗 {src['url']}"
            for src in result["sources"]
        ])

        # Format images (return full paths for Gallery)
        image_paths = [
            os.path.join("data/images", img["image_path"])
            for img in result["images"]
        ] if result["images"] else []

        return result["answer"], sources_text, image_paths

    except Exception as e:
        return f"Error: {str(e)}", "", []


# Setup tab
setup_tab = gr.Interface(
    fn=initialize_bot,
    inputs=[gr.Textbox(label="Groq API Key", type="password", placeholder="gsk_...")],
    outputs=[gr.Textbox(label="Status")],
    title="Setup Article Q&A Bot",
    description="Enter your Groq API key. Get one at: https://console.groq.com/keys"
)

# Article QA tab
with gr.Blocks() as qa_tab:
    gr.Markdown("# 📰 Article Q&A System")
    gr.Markdown("Ask questions about AI news from The Batch articles")

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="What are the latest developments in transformers?",
                lines=2
            )

            with gr.Row():
                categories_input = gr.CheckboxGroup(
                    choices=CATEGORIES,
                    label="Filter by Categories (optional)",
                    value=[]
                )

            with gr.Row():
                date_start_input = gr.Textbox(
                    label="Start Date (optional)",
                    placeholder="2025-10-01"
                )
                date_end_input = gr.Textbox(
                    label="End Date (optional)",
                    placeholder="2025-11-01"
                )

            with gr.Row():
                n_results_input = gr.Slider(
                    minimum=1, maximum=10, step=1, value=5,
                    label="Number of Results"
                )
                hybrid_input = gr.Checkbox(
                    label="Use Hybrid Search (BM25 + Semantic)",
                    value=True
                )

            submit_btn = gr.Button("🔍 Ask Question", variant="primary")

        with gr.Column(scale=3):
            answer_output = gr.Textbox(label="Answer", lines=8)
            sources_output = gr.Markdown(label="Sources")

    images_output = gr.Gallery(
        label="Related Images",
        columns=3
    )

    submit_btn.click(
        fn=answer_question,
        inputs=[query_input, categories_input, date_start_input,
                date_end_input, n_results_input, hybrid_input],
        outputs=[answer_output, sources_output, images_output]
    )

# Combine tabs
demo = gr.TabbedInterface(
    [setup_tab, qa_tab],
    ["Setup", "Article QA"]
)

if __name__ == "__main__":
    # Use share=True for WSL2 or remote environments
    demo.launch(share=True, debug=True) # debug=True for detailed error messages
