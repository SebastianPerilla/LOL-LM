"""Pipeline to refresh RAG by fetching and ingesting new articles."""

import os
import sys
from pathlib import Path
from typing import Optional, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file first
from utils import env  # noqa: F401

from services.extract_news import ExtractorService
from services.rag import RAGService
from logger import log

# Categories available in News API top-headlines endpoint
NEWS_API_CATEGORIES = [
    "business",
    "entertainment",
    "general",
    "health",
    "science",
    "sports",
    "technology",
]

# Default categories to fetch (can be customized)
DEFAULT_CATEGORIES = NEWS_API_CATEGORIES  # Fetch from all categories


def refresh_rag(
    num_articles: int = 100,
    country: str = "us",
    categories: Optional[List[str]] = None,
    clear_existing: bool = False,
    articles_per_category: Optional[int] = None,
) -> None:
    """
    Refresh RAG by fetching new articles and ingesting them.
    
    Args:
        num_articles: Total number of articles to fetch and ingest (default: 100).
                     If articles_per_category is set, this is ignored.
        country: Country code for headlines (default: "us")
        categories: List of categories to fetch from (default: all categories)
        clear_existing: Whether to clear existing collection before ingesting
        articles_per_category: Number of articles to fetch per category.
                              If set, fetches this many from each category (ignores num_articles).
    """
    log.info("=" * 60)
    log.info("Starting RAG refresh pipeline")
    if articles_per_category:
        log.info(f"Target: {articles_per_category} articles per category")
    else:
        log.info(f"Target: {num_articles} articles total")
    if categories:
        log.info(f"Categories: {', '.join(categories)}")
    else:
        log.info(f"Categories: {', '.join(DEFAULT_CATEGORIES)} (all)")
    log.info("=" * 60)
    
    # Initialize services
    log.info("Initializing services...")
    extractor = ExtractorService()
    rag = RAGService()
    
    # Clear existing collection if requested
    if clear_existing:
        log.warning("Clearing existing RAG collection...")
        rag.clear_collection()
    
    # Use default categories if none specified
    if categories is None:
        categories = DEFAULT_CATEGORIES
    
    # Validate categories
    invalid_categories = [c for c in categories if c not in NEWS_API_CATEGORIES]
    if invalid_categories:
        log.warning(f"Invalid categories: {invalid_categories}. Using valid ones only.")
        categories = [c for c in categories if c in NEWS_API_CATEGORIES]
    
    if not categories:
        log.error("No valid categories specified. Aborting.")
        return
    
    # Step 1: Fetch headlines from News API across all categories
    if articles_per_category:
        total_target = articles_per_category * len(categories)
        log.info(f"Step 1: Fetching {articles_per_category} articles per category ({total_target} total) from News API...")
    else:
        log.info(f"Step 1: Fetching {num_articles} headlines from News API...")
    articles_fetched = []
    
    try:
        # Calculate articles per category
        if articles_per_category is None:
            # Distribute total articles across categories
            target_per_category = max(1, num_articles // len(categories))
        else:
            # Use specified articles per category
            target_per_category = articles_per_category
        
        for category in categories:
            log.info(f"Fetching from category: {category}")
            category_articles = []
            page = 1
            target_count = target_per_category
            
            # Fetch articles for this category with proper pagination
            while len(category_articles) < target_count:
                remaining = target_count - len(category_articles)
                page_size = min(remaining, 100)  # Max is 100 per request
                
                headlines = extractor.fetch_top_headlines(
                    country=country,
                    category=category,
                    page_size=page_size,
                    page=page,
                )
                
                if not headlines:
                    log.info(f"No more articles in category '{category}'. Got {len(category_articles)} articles.")
                    break
                
                category_articles.extend(headlines)
                log.info(f"Fetched {len(headlines)} headlines from {category} (total: {len(category_articles)})")
                
                # Check if we have enough or if we need to paginate
                if len(category_articles) >= target_count:
                    break
                
                # If we got exactly page_size results, there might be more pages
                if len(headlines) == page_size:
                    page += 1
                    # Safety check
                    if page > 10:  # Max 10 pages per category
                        log.warning(f"Reached max pages for category {category}")
                        break
                else:
                    # Got fewer results than requested, no more pages
                    break
            
            # Trim to target count for this category
            category_articles = category_articles[:target_count]
            articles_fetched.extend(category_articles)
            log.info(f"Category '{category}': {len(category_articles)} articles")
        
        # Trim to exact number if using num_articles mode (not per-category mode)
        if articles_per_category is None:
            articles_fetched = articles_fetched[:num_articles]
        log.info(f"Successfully fetched {len(articles_fetched)} article URLs from {len(categories)} categories")
        
    except Exception as e:
        log.error(f"Error fetching headlines: {e}")
        return
    
    # Step 2: Extract full content from URLs
    log.info("Step 2: Extracting full article content...")
    urls = [article.url for article in articles_fetched]
    try:
        extracted_articles = extractor.extract_articles(urls)
        log.info(f"Successfully extracted {len(extracted_articles)} articles")
        
        if not extracted_articles:
            log.error("No articles were successfully extracted. Aborting.")
            return
        
    except Exception as e:
        log.error(f"Error extracting articles: {e}")
        return
    
    # Step 3: Ingest into RAG
    log.info("Step 3: Ingesting articles into RAG...")
    try:
        ingested_count = rag.ingest_articles(extracted_articles)
        log.info(f"Successfully ingested {ingested_count} articles into RAG")
        
        # Get collection info
        info = rag.get_collection_info()
        log.info(f"Collection info: {info}")
        
    except Exception as e:
        log.error(f"Error ingesting articles: {e}")
        return
    
    log.info("=" * 60)
    log.info("RAG refresh completed successfully!")
    log.info(f"Summary:")
    log.info(f"  - Fetched: {len(articles_fetched)} headlines")
    log.info(f"  - Extracted: {len(extracted_articles)} articles")
    log.info(f"  - Ingested: {ingested_count} articles")
    log.info("=" * 60)


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Refresh RAG with new articles")
    parser.add_argument(
        "--num-articles",
        type=int,
        default=int(os.getenv("RAG_REFRESH_NUM_ARTICLES", "100")),
        help="Number of articles to fetch and ingest (default: 100, or RAG_REFRESH_NUM_ARTICLES env var)",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=os.getenv("RAG_REFRESH_COUNTRY", "us"),
        help="Country code for headlines (default: us, or RAG_REFRESH_COUNTRY env var)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Categories to fetch from (business, technology, etc.). If not specified, fetches from all categories.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing collection before ingesting new articles",
    )
    parser.add_argument(
        "--articles-per-category",
        type=int,
        default=None,
        help="Number of articles to fetch per category. If set, ignores --num-articles and fetches this many from each category.",
    )
    
    args = parser.parse_args()
    
    refresh_rag(
        num_articles=args.num_articles,
        country=args.country,
        categories=args.categories,
        clear_existing=args.clear,
        articles_per_category=args.articles_per_category,
    )


if __name__ == "__main__":
    main()

