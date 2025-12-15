"""Data models and schemas for the application."""

from schemas.article import Article
from schemas.news_api import NewsAPIResponse, NewsAPIArticle, NewsAPISource

__all__ = ["Article", "NewsAPIResponse", "NewsAPIArticle", "NewsAPISource"]

