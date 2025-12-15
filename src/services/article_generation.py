"""Service that orchestrates RAG + Llama editor + fine-tuned writer generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from logger import log
from prompts.editor import get_editor_prompt
from prompts.writer import SYSTEM_PROMPT
from services.rag import RAGService
from utils.device import get_device


class ArticleGenerationService:
    """Pipeline: RAG retrieval -> Llama (editor) -> fine-tuned writer."""

    def __init__(
        self,
        editor_model: str = "gpt-oss:20b",
        writer_model_path: Optional[Path | str] = None,
        rag_collection: str = "articles",
        rag_top_k: int = 5,
        max_preview_chars: int = 240,
    ):
        self.rag_top_k = rag_top_k
        self.max_preview_chars = max_preview_chars

        # RAG
        self.rag = RAGService(collection_name=rag_collection)

        # Editor (Llama/Ollama)
        self.editor_llm = OllamaLLM(model=editor_model, temperature=0.3)
        self.editor_prompt = PromptTemplate(
            template=get_editor_prompt(),
            input_variables=["articles_preview", "raw_news"],
        )
        self.editor_chain = self.editor_prompt | self.editor_llm

        # Writer (fine-tuned GPT-2)
        project_root = Path(__file__).parent.parent.parent
        default_writer_path = project_root / "data" / "model" / "gpt2-satirical-news" / "final"
        self.writer_model_path = Path(writer_model_path) if writer_model_path else default_writer_path

        device_str = get_device()
        self.device = torch.device(
            "cuda" if device_str == "cuda" else "mps" if device_str == "mps" else "cpu"
        )
        log.info(f"Using device for writer: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.writer_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.writer_model_path).to(self.device)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.writer_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def _preview_from_docs(self, docs: List[Any]) -> tuple[str, List[Dict[str, str]]]:
        previews: List[str] = []
        structured: List[Dict[str, str]] = []
        for idx, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            title = meta.get("title") or "Untitled"
            source = meta.get("source") or meta.get("hostname") or "unknown"
            url = meta.get("url") or ""
            snippet = (doc.page_content or "").replace("\n", " ").strip()
            snippet = snippet[: self.max_preview_chars]
            previews.append(f"{idx}. {title} [{source}] {url}\n   {snippet}")
            structured.append(
                {
                    "title": title,
                    "source": source,
                    "url": url,
                    "snippet": snippet,
                }
            )
        return "\n".join(previews) if previews else "No retrieved articles.", structured

    def _run_editor(self, articles_preview: str, raw_news: str) -> str:
        response = self.editor_chain.invoke(
            {
                "articles_preview": articles_preview,
                "raw_news": raw_news,
            }
        )
        editor_text = response.content if hasattr(response, "content") else str(response)
        editor_text = editor_text.strip()
        # Ensure exact prefix
        if not editor_text.lower().startswith("generate an article on:"):
            editor_text = f"Generate an article on: {editor_text}"
        return editor_text

    def _run_writer(
        self,
        user_prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_length: int,
    ) -> str:
        """Run the fine-tuned writer model, always passing current date context."""
        current_date = datetime.now().strftime("%B %d, %Y")
        user_with_date = f"{user_prompt} (Today is {current_date}.)"
        writer_input = f"<system>{SYSTEM_PROMPT}<user>{user_with_date}<assistant>"
        outputs = self.writer_pipe(
            writer_input,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return outputs[0]["generated_text"]

    @staticmethod
    def _parse_headline_article(text: str) -> tuple[Optional[str], Optional[str]]:
        headline = None
        article = None
        if "Headline:" in text:
            after_head = text.split("Headline:", 1)[1]
            if "Article:" in after_head:
                headline_part, article_part = after_head.split("Article:", 1)
                headline = headline_part.strip()
                article = article_part.strip()
            else:
                headline = after_head.strip()
        return headline, article

    def generate(
        self,
        user_query: str,
        *,
        use_rag: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_length: int = 512,
    ) -> Dict[str, Any]:
        """End-to-end generation.

        If use_rag is False, bypasses RAG + editor and passes the user query
        directly (wrapped as a 'Generate an article on: ...' prompt) to the writer.
        """
        if use_rag:
            docs = self.rag.search(user_query, k=self.rag_top_k)
            articles_preview, structured = self._preview_from_docs(docs)
            raw_news = "\n\n".join([doc.page_content or "" for doc in docs])[:2000]
            editor_output = self._run_editor(articles_preview, raw_news)
        else:
            docs = []
            structured = []
            articles_preview = "RAG and editor disabled. Using user query directly."
            raw_news = ""
            editor_output = f"Generate an article on: {user_query}"

        writer_raw = self._run_writer(
            editor_output,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
        )
        headline, article = self._parse_headline_article(writer_raw)

        return {
            "articles_preview": structured,
            "editor_output": editor_output,
            "raw_text": writer_raw,
            "headline": headline,
            "article": article,
        }

