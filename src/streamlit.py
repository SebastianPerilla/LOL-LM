import streamlit as st
from typing import Any, Dict

from services import ArticleGenerationService

# Page configuration
st.set_page_config(
    page_title="LOL-LM Satirical News",
    page_icon="📰",
    layout="wide",
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize article generation service
if "article_service" not in st.session_state:
    st.session_state.article_service = ArticleGenerationService()


def format_pipeline_markdown(result: Dict[str, Any]) -> str:
    """Build a markdown view of the whole pipeline for the chat bubble."""
    lines: list[str] = []

    # Retrieved articles
    lines.append("### Retrieved articles")
    if result["articles_preview"]:
        for art in result["articles_preview"]:
            title = art.get("title") or "Untitled"
            source = art.get("source") or "unknown"
            url = art.get("url") or ""
            snippet = art.get("snippet") or ""
            if url:
                lines.append(f"- **{title}** (*{source}*) - [{url}]({url})  ")
            else:
                lines.append(f"- **{title}** (*{source}*)  ")
            if snippet:
                lines.append(f"  {snippet}")
    else:
        lines.append("- _No retrieved articles (RAG disabled or no matches)._")

    # Editor output
    lines.append("\n### Editor output")
    lines.append(f"`{result['editor_output']}`")

    # Final article
    lines.append("\n### Final article")
    headline = result.get("headline")
    article = result.get("article")
    raw_text = result.get("raw_text", "")

    if headline or article:
        if headline:
            lines.append(f"## {headline}")
        if article:
            lines.append("")
            lines.append(article)
    else:
        # Fallback: show raw text
        lines.append(raw_text)

    return "\n".join(lines)


def display_message(message: Dict[str, Any]) -> None:
    """Display a message in the chat interface."""
    with st.chat_message(message["role"]):
        st.markdown(message["text"])


# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Generation Settings")

    use_rag_editor = st.checkbox(
        "Use RAG & Editor (Llama)",
        value=True,
        help="If disabled, your query is passed directly to the fine-tuned model.",
    )

    st.subheader("Fine-tuned writer parameters")
    max_length = st.slider("Max Length", 128, 1024, 512, 32)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)

    st.divider()

    # Clear chat button
    if st.button("🗑️ New chat / Clear"):
        st.session_state.messages = []
        st.rerun()


# Main chat interface
st.title("📰 LOL-LM Satirical News Generator")
st.caption("Ask for a topic, see RAG + Llama editor + fine-tuned writer work together.")

# Display chat history
for message in st.session_state.messages:
    display_message(message)

# Chat input
if prompt := st.chat_input("Ask for a satirical article about..."):
    # Add user message to chat history
    user_message = {
        "role": "user",
        "text": prompt,
    }
    st.session_state.messages.append(user_message)
    display_message(user_message)

    # Generate response using article service
    service: ArticleGenerationService = st.session_state.article_service

    with st.chat_message("assistant"):
        with st.spinner("Running RAG, editor, and writer pipeline..."):
            result = service.generate(
                prompt,
                use_rag=use_rag_editor,
                temperature=temperature,
                top_p=top_p,
                max_length=max_length,
            )
            assistant_text = format_pipeline_markdown(result)
            assistant_message = {
                "role": "assistant",
                "text": assistant_text,
            }
            st.markdown(assistant_text)

    # Add assistant message to chat history
    st.session_state.messages.append(assistant_message)

