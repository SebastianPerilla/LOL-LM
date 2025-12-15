## LOL-LM – Satirical News Generator

LOL-LM is a small end-to-end system that writes **satirical news articles about real, up‑to‑date events**.  
Think: a tiny Babylon Bee / Onion intern powered by GPT‑2 + RAG + an Ollama “editor” model.

The core idea:
- **Writer**: a fine‑tuned GPT‑2 model that knows how to write satirical news in a consistent format.  
- **Editor**: a stronger general‑purpose model (via Ollama) that reads fresh news (RAG) and decides *what* the satire should be about.  
- **RAG**: a ChromaDB-based retriever that keeps the system grounded on current articles instead of hallucinating 2018 headlines.

## Workflow Overview

High‑level flow:

1. **Ingestion**: fetch real news via NewsAPI, scrape full bodies with Trafilatura, and store them in ChromaDB using sentence embeddings.
2. **Fine‑tuning**: train GPT‑2 on Babylon Bee articles to learn the satirical “house style”.
3. **Generation time**:
   - User asks for a topic (e.g. “write about Trump and AI regulation”).
   - RAG retrieves relevant, recent news articles.
   - Editor model (via Ollama) turns those snippets into a compact user prompt:  
     `Generate an article on: <topics + short up‑to‑date description>`
   - Fine‑tuned GPT‑2 writes:  
     `Headline: ...`  
     `Article: ...`


![LOL-LM workflow](assets/workflow_diagram.png)

---

## Features

- **News ingestion + scraping**
  - Fetches latest articles from NewsAPI across categories: business, entertainment, general, health, science, sports, technology.
  - Uses **Trafilatura** to bypass paywalls where possible and extract the **headline + article body + metadata**.
  - Stores everything as `Article` objects and ingests them into **ChromaDB** with a HuggingFace embedding model (`all-MiniLM-L6-v2`).

- **Satire fine‑tuning**
  - Uses the 2024 dataset *“A Dataset of Multi‑modal Satire – humorous and non‑humorous”* (Babylon Bee articles, ~10k+).
  - Cleans everything into `Headline` and `Article` columns.
  - Runs spaCy to extract **headline‑first topics** (noun chunks/entities) and builds:
    - `system`: `"You are a satirical news generator. When given a topic, generate a funny headline followed by the article."`
    - `user`: `"Generate an article on: {topics}"`
    - `result`: `"Headline: {headline}\nArticle: {article}"`
    - `training_text`: `<system>...<user>...<assistant>Headline: ...\nArticle: ...<|endoftext|>`
  - Fine‑tunes **GPT‑2** on this training_text.

- **RAG‑augmented generation**
  - At generation time, doesn’t just hallucinate: it queries ChromaDB for real articles and shows you the **first snippets** in the UI so you can sanity‑check them.
  - Editor model compresses those snippets + today’s date into a short `Generate an article on: ...` prompt.
  - Fine‑tuned GPT‑2 writes the satire; result is parsed into `headline` + `article` and rendered nicely in Streamlit.

- **Streamlit app**
  - Chat-style interface with:
    - Full pipeline trace: retrieved articles, editor output, final article.
    - Left sidebar to tweak:
      - Writer **temperature**, **top‑p**, **max length**.
      - Toggle **RAG + editor** on/off. When off, the user query is fed **directly** to the fine‑tuned GPT‑2 writer.
  - Final output is rendered as markdown:
    - Headline in big heading.
    - Article underneath as normal text.

## Installation

From the project root:

```bash
uv venv .venv
source .venv/bin/activate        # on macOS / Linux
# or on Windows:
# .venv\Scripts\activate
uv sync
```

You’ll also want:

- **Ollama** installed and running, with at least `gpt-oss:20b` pulled:

```bash
ollama pull gpt-oss:20b
```

- A `.env` file with your NewsAPI key (see `src/utils/env.py` / `src/services/extract_news.py` for exact variable names).


## RAG Pipeline

In this project the purpose of RAG is to **search for real breaking news**, use that as context, and then write satire about it.

**Without RAG**

- GPT‑2 would write about:
  - Events from years ago, depending on its pretraining cut‑off, or
  - Completely fabricated topics that don’t resemble current headlines.

**With RAG**

- We improve **contextual understanding** and reduce hallucination by feeding in:
  - Recent article snippets (with source + URL).
  - Today’s date.

### Managing the RAG index

The `Makefile` exposes a few handy commands:

```bash
# Refresh RAG with default number of articles (NUM_ARTICLES=100, COUNTRY=us)
make refresh-rag

# Clear collection and fully refresh from all categories
make refresh-rag-all NUM_ARTICLES=200

# Show collection info (document count, etc.)
make rag-info

# Remove duplicate documents (by URL)
make rag-remove-duplicates

# Clear the collection completely
make rag-clear
```

Under the hood, these call the scripts in `src/pipelines/refresh_rag.py` and `src/pipelines/rag_manage.py`, which use `RAGService` (`src/services/rag.py`) to manage ChromaDB.

---

## Fine‑Tuning GPT‑2

The full fine‑tuning process lives in the notebook:

- `src/notebooks/finetuning.ipynb`

Key steps:

1. **Load dataset**
   - Babylon Bee articles from the satire dataset → `babylonbee_processed.csv`.
2. **Topic extraction**
   - spaCy noun‑chunks/entities, headline‑first.
   - Result stored in `topics` column.
3. **Training CSV cache**
   - Notebook creates `babylonbee_finetune.csv` with:
     - `Headline`, `Article`, `topics`, `system`, `user`, `result`, `training_text`.
   - On reruns, it loads this CSV instead of recomputing.
4. **Formatting & tokenization**
   - Adds special tokens (`<system>`, `<user>`, `<assistant>`).
   - Truncates to 512 tokens and splits into train/val.
5. **Training**
   - Detects device (`cuda` → `mps` → `cpu`).
   - Uses `TrainingArguments` with fp16 only on CUDA, and reasonable defaults for batch size / epochs.

Trained model gets saved under:

```text
data/model/gpt2-satirical-news/final
```

This is what the Streamlit app loads as the writer.

## Article Generation Service

The glue logic for generation lives in:

- `src/services/article_generation.py`

High‑level API:

```python
from services import ArticleGenerationService

service = ArticleGenerationService(
    editor_model="gpt-oss:20b",  # Ollama model used as the editor
    # writer_model_path defaults to data/model/gpt2-satirical-news/final
)

result = service.generate(
    "Write about US election debates",
    use_rag=True,          # toggle RAG + editor
    temperature=0.8,
    top_p=0.95,
    max_length=512,
)

print(result["headline"])
print()
print(result["article"])
```

Returned JSON shape:

```python
{
    "articles_preview": [  # snippets for UI
        {"title": "...", "source": "...", "url": "...", "snippet": "..."},
        # ...
    ],
    "editor_output": "Generate an article on: ...",
    "raw_text": "... full GPT-2 output ...",
    "headline": "Parsed headline or None",
    "article": "Parsed article body or None",
}
```

Internally:

- If `use_rag=True`:
  - Calls `RAGService.search` to get top‑k documents.
  - Editor model (Ollama) runs over:
    - `articles_preview` (short list for the user),
    - `raw_news` (joined content chunks),
    - and outputs **exactly one line**:  
      `Generate an article on: ...`
- If `use_rag=False`:
  - It just builds that line directly from the user query.
- Writer model:
  - Always gets a prompt of the form:  
    `<system>{SYSTEM_PROMPT}<user>Generate an article on: ... (Today is {DATE}.)<assistant>`
  - Ensures the model is **always aware of the current date**.

## Streamlit App

The interactive UI is defined in:

- `src/streamlit.py`

Launch it with:

```bash
make run-app
```

That runs:

```bash
uv run streamlit run src/streamlit.py
```

### What you see in the app

- **Chat interface**:
  - Type: “write about Trump and AI regulation” or “give me a satire about tech layoffs”.
  - Each assistant reply shows:
    - **Retrieved articles** (titles, sources, URLs, and the first part of each snippet).
    - **Editor output** (`Generate an article on: ...`).
    - **Final article**:
      - Headline as a `##` markdown heading.
      - Article body below it.

- **Sidebar controls**:
  - `Use RAG & Editor (Llama)`:
    - On: full editor + RAG pipeline.
    - Off: your prompt is fed directly (in the same `Generate an article on: ...` format) to the fine‑tuned GPT‑2 model.
  - Writer parameters:
    - `Max Length`
    - `Temperature`
    - `Top‑p`
  - `🗑️ New chat / Clear` resets the conversation state.

## Dependencies (non‑exhaustive)

- **Core**
  - Python 3.11
  - `uv` (for env + dependency management)
  - `torch` (with CUDA or MPS where available)
  - `transformers`, `datasets`

- **RAG / LLM orchestration**
  - `langchain`, `langchain_ollama`, `langchain_chroma`, `langchain_huggingface`
  - `ChromaDB`

- **NLP / scraping**
  - `spaCy` + `en_core_web_sm`
  - `trafilatura`

- **Serving / UI**
  - `streamlit`
  - `ollama` (installed separately on the host machine)

Check `pyproject.toml` and `uv.lock` for exact versions.

## Notes / Limitations

- GPT‑2 is **small** – that’s by design for speed and simplicity. It learns style, not deep world knowledge.
- The satire quality is very dependent on:
  - The quality of retrieved articles.
  - The editor model’s ability to compress them into a good `Generate an article on: ...` topic line.
- RAG index must be refreshed regularly (`make refresh-rag`) if you care about staying current.  

Have fun breaking the news. Literally.  
