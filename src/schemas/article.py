"""Article data model for news articles."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Article:
    """Represents a news article with metadata."""
    
    title: str
    content: str
    url: str
    source: Optional[str] = None
    published_at: Optional[datetime] = None
    author: Optional[str] = None
    description: Optional[str] = None
    hostname: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert article to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "hostname": self.hostname,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "author": self.author,
            "description": self.description,
        }
    
    def __str__(self) -> str:
        """String representation of the article."""
        return f"{self.title} - {self.source} ({self.published_at}) - {self.hostname}"

