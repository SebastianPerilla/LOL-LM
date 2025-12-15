"""Pydantic schemas for News API responses."""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class NewsAPISource(BaseModel):
    """News API source object."""
    id: Optional[str] = None
    name: str


class NewsAPIArticle(BaseModel):
    """News API article object from top-headlines endpoint."""
    source: NewsAPISource
    author: Optional[str] = None
    title: str
    description: Optional[str] = None
    url: str
    url_to_image: Optional[str] = Field(None, alias="urlToImage")
    published_at: datetime = Field(..., alias="publishedAt")
    content: Optional[str] = None  # Truncated to 200 chars if available
    
    @field_validator('published_at', mode='before')
    @classmethod
    def parse_published_at(cls, v):
        """Parse publishedAt string to datetime."""
        if isinstance(v, str):
            # News API format: "2024-01-15T10:30:00Z" or "2024-01-15T10:30:00+00:00"
            try:
                # Handle Z suffix (UTC)
                if v.endswith('Z'):
                    v = v[:-1] + '+00:00'
                # Parse ISO format
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                # If parsing fails, return as-is and let Pydantic handle the error
                return v
        return v
    
    class Config:
        populate_by_name = True  # Allow both field name and alias


class NewsAPIResponse(BaseModel):
    """News API top-headlines response object."""
    status: str  # "ok" or "error"
    total_results: int = Field(..., alias="totalResults")
    articles: List[NewsAPIArticle]
    code: Optional[str] = None  # Error code if status is "error"
    message: Optional[str] = None  # Error message if status is "error"
    
    class Config:
        populate_by_name = True

