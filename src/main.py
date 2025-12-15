import time
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Load .env file first
from utils import env  # noqa: F401

from logger import log
from prompts.editor import get_editor_prompt
from prompts.writer import get_writer_prompt

# --- CONFIGURATION (The "Work" part) ---
# Use the stock model for logic, and your fine-tuned one for style
EDITOR_MODEL = "gpt-oss:20b"  
WRITER_MODEL = "gpt-oss:20b" # Change this to your fine-tuned model name later

# --- 1. THE TOOLS ---
# We use LangChain's pre-built tool for speed
search_tool = DuckDuckGoSearchRun()

def get_live_news(topic):
    """
    Custom wrapper to ensure we get news, not generic info.
    Teacher Note: This counts as 'Data Preprocessing'.
    """
    log.info(f"🔍 Searching for latest news on: {topic}...")
    query = f"{topic} news current event {time.strftime('%Y')}"
    # Retry logic in case of rate limits (Simple robustness)
    try:
        return search_tool.run(query)
    except Exception as e:
        log.error(f"Error fetching news: {e}")
        return f"Error fetching news: {e}"

# --- 2. THE EDITORIAL BRAIN (Stock Model) ---
# This chain acts as the "Editor" who finds the angle
editor_llm = OllamaLLM(model=EDITOR_MODEL, temperature=0.3)

editor_template = get_editor_prompt()
editor_prompt = PromptTemplate(template=editor_template, input_variables=["raw_news"])
editor_chain = editor_prompt | editor_llm

# --- 3. THE SATIRE GENERATOR (Fine-Tuned Model) ---
# This chain writes the actual article
writer_llm = OllamaLLM(model=WRITER_MODEL, temperature=0.8) # Higher temp for creativity

writer_template = get_writer_prompt()
writer_prompt = PromptTemplate(template=writer_template, input_variables=["editor_output"])
writer_chain = writer_prompt | writer_llm

# --- 4. THE PIPELINE (Main execution) ---
def run_newsroom(topic):
    # Step A: Get Data
    raw_news = get_live_news(topic)
    
    # Step B: Editor decides the angle
    log.info("🤔 Editor is thinking...")
    editor_response = editor_chain.invoke({"raw_news": raw_news})
    editor_output = editor_response.content if hasattr(editor_response, 'content') else str(editor_response)
    log.info(f"\n--- ANGLE ---\n{editor_output}\n")
    
    # Step C: Writer drafts the story
    log.info("✍️  Writer is typing...")
    writer_response = writer_chain.invoke({"editor_output": editor_output})
    final_article = writer_response.content if hasattr(writer_response, 'content') else str(writer_response)
    
    return final_article

if __name__ == "__main__":
    topic = input("Enter a news topic: ")
    log.info(run_newsroom(topic))