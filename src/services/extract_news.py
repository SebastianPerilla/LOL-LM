"""News API service for fetching and extracting news articles."""

import os
import json
from typing import List, Optional
from datetime import datetime, timedelta
from urllib.parse import urlparse
import requests
import trafilatura

# Load .env file
from utils import env

from schemas.news_api import NewsAPIResponse, NewsAPIArticle
from schemas.article import Article
from logger import log


class ExtractorService:
    """Service for extracting news articles from URLs."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize News API service.
        
        Args:
            api_key: News API key. If None, will try to get from environment variable NEWS_API_KEY.
        """
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        if not self.api_key:
            log.warning("News API key not found. Set NEWS_API_KEY environment variable.")
        self.base_url = "https://newsapi.org/v2"
    
    def fetch_top_headlines(
        self,
        country: Optional[str] = None,
        category: Optional[str] = None,
        sources: Optional[str] = None,
        q: Optional[str] = None,
        page_size: Optional[int] = None,
        page: Optional[int] = None,
    ) -> List[Article]:
        """
        Fetch top headlines from News API /v2/top-headlines endpoint.
        
        NOTE: News API does NOT provide full article content. It only provides:
        - Title, description (snippet), URL, source, published date, author
        - Content field is truncated to 200 chars if available
        - Use extract_article() to extract full content from the URLs
        
        Args:
            country: The 2-letter ISO 3166-1 code of the country (e.g., "us").
                     Cannot be mixed with sources param.
            category: The category (business, entertainment, general, health, 
                     science, sports, technology). Cannot be mixed with sources param.
            sources: Comma-separated string of source identifiers. 
                    Cannot be mixed with country or category params.
            q: Keywords or a phrase to search for.
            page_size: Number of results per page (default: 20, max: 100).
            page: Page number for pagination.
            
        Returns:
            List of Article objects (with description only, not full content)
            Use extract_article() to get full content from URLs
            
        Raises:
            ValueError: If invalid parameter combinations are provided
        """
        if not self.api_key:
            log.error("News API key not configured. Cannot fetch headlines.")
            return []
        
        # Validate parameter combinations
        if sources and (country or category):
            raise ValueError("Cannot mix 'sources' parameter with 'country' or 'category'")
        
        # Build request parameters
        params = {"apiKey": self.api_key}
        
        if country:
            params["country"] = country
        if category:
            params["category"] = category
        if sources:
            params["sources"] = sources
        if q:
            params["q"] = q
        if page_size is not None:
            params["pageSize"] = min(page_size, 100)  # Max is 100
        if page is not None:
            params["page"] = page
        
        # Make API request
        url = f"{self.base_url}/top-headlines"
        log.info(f"Fetching top headlines with params: {', '.join(k for k in params.keys() if k != 'apiKey')}")
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse response with Pydantic
            api_response = NewsAPIResponse.model_validate(response.json())
            
            if api_response.status == "error":
                log.error(f"News API error: {api_response.message} (code: {api_response.code})")
                return []
            
            # Convert NewsAPIArticle to Article objects
            articles = []
            for api_article in api_response.articles:
                # Use content if available (truncated to 200 chars), otherwise use description
                content = api_article.content or api_article.description or ""
                
                article = Article(
                    title=api_article.title,
                    content=content,
                    url=api_article.url,
                    source=api_article.source.name,
                    published_at=api_article.published_at,
                    author=api_article.author,
                    description=api_article.description,
                )
                articles.append(article)
            
            log.info(f"Successfully fetched {len(articles)} articles (total available: {api_response.total_results})")
            log.warning("News API returns metadata only. Use extract_article() to extract full content.")
            return articles
            
        except requests.exceptions.RequestException as e:
            log.error(f"Error fetching top headlines: {e}")
            return []
        except Exception as e:
            log.error(f"Unexpected error parsing News API response: {e}")
            return []
    
    def extract_article(self, url: str) -> Optional[Article]:
        """
        Extract full article content from a URL using trafilatura.
        
        Args:
            url: URL of the article to extract
            
        Returns:
            Article object with extracted content, or None if extraction fails
        """
        log.info(f"Extracting article from URL: {url}")
        
        try:
            # Fetch and extract article content using trafilatura
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                log.warning(f"Failed to fetch content from {url}")
                return None
            
            # Extract as JSON with metadata
            extracted_json = trafilatura.extract(downloaded, output_format="json", with_metadata=True, include_comments=False, include_tables=False, no_fallback=True)
            if not extracted_json:
                log.warning(f"Failed to extract content from {url}")
                return None
            
            # Parse JSON response
            article_data = json.loads(extracted_json)
            
            # Extract fields from trafilatura JSON
            title = article_data.get("title", "")
            content = article_data.get("raw_text", "") or article_data.get("text", "")
            author = article_data.get("author")
            published_date = article_data.get("date")
            description = article_data.get("description")
            
            # Get source from hostname (trafilatura provides this) or parse from URL
            hostname = article_data.get("hostname")
            if hostname:
                source = hostname.replace("www.", "")
            else:
                # Fallback: parse from URL
                parsed_url = urlparse(url)
                source = parsed_url.netloc.replace("www.", "")
            
            # Parse published date if available
            published_at = None
            if published_date:
                try:
                    # trafilatura returns dates in various formats, try to parse
                    published_at = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    try:
                        # Try alternative parsing
                        published_at = datetime.strptime(published_date, "%Y-%m-%d")
                    except (ValueError, AttributeError):
                        log.debug(f"Could not parse date: {published_date}")
            
            # Create Article object
            article = Article(
                title=title,
                content=content,
                url=url,
                source=source,  # Use hostname as source
                published_at=published_at,
                author=author,
                description=description,
                hostname=hostname,  # Also store original hostname
            )
            
            log.info(f"Successfully extracted article: {title[:50]}...")
            return article
            
        except Exception as e:
            log.error(f"Error extracting article from {url}: {e}")
            return None
    
    def extract_articles(self, urls: List[str]) -> List[Article]:
        """
        Extract multiple articles from URLs.
        
        Args:
            urls: List of article URLs to extract
            
        Returns:
            List of Article objects (may be shorter than input if some extractions fail)
        """
        log.info(f"Extracting {len(urls)} articles")
        articles = []
        
        for url in urls:
            article = self.extract_article(url)
            if article:
                articles.append(article)
            else:
                log.warning(f"Failed to extract article from {url}")
        
        log.info(f"Successfully extracted {len(articles)}/{len(urls)} articles")
        return articles

