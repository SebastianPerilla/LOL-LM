"""RAG service for ingesting and retrieving articles using ChromaDB."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from schemas.article import Article
from utils.device import get_device, get_device_info
from logger import log


class RAGService:
    """RAG service for managing article embeddings and retrieval using ChromaDB."""
    
    def __init__(
        self,
        collection_name: str = "articles",
        persist_directory: Optional[Path] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        device: Optional[str] = None,
    ):
        """
        Initialize RAG service with ChromaDB and embeddings.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data (default: data/chromadb)
            embedding_model_name: Name of the HuggingFace embedding model
            chunk_size: Size of text chunks for splitting articles
            chunk_overlap: Overlap between chunks
            device: Device to use ("cuda", "mps", "cpu"). If None, auto-detects best device.
        """
        # Set up persist directory
        if persist_directory is None:
            base_dir = Path(__file__).parent.parent.parent
            persist_directory = base_dir / "data" / "chromadb"
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Detect and set device
        self.device = get_device(device) if device else get_device()
        device_info = get_device_info()
        log.info(f"Device detection: {device_info}")
        log.info(f"Using device: {self.device}")
        
        # Initialize embeddings model
        log.info(f"Initializing embeddings model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": self.device},
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Initialize ChromaDB connection
        log.info(f"Initializing ChromaDB at: {self.persist_directory}")
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )
        
        log.info("RAG service initialized successfully")
    
    def ingest_articles(self, articles: List[Article]) -> int:
        """
        Ingest articles into ChromaDB by converting them to LangChain documents.
        
        Args:
            articles: List of Article objects to ingest
            
        Returns:
            Number of articles successfully ingested
        """
        if not articles:
            log.warning("No articles provided for ingestion")
            return 0
        
        log.info(f"Ingesting {len(articles)} articles into ChromaDB")
        
        # Convert articles to LangChain documents
        documents = []
        for article in articles:
            # Create document with article content
            doc = Document(
                page_content=article.content,
                metadata={
                    "title": article.title,
                    "url": article.url,
                    "source": article.source,
                    "author": article.author or "",
                    "description": article.description or "",
                    "published_at": article.published_at.isoformat() if article.published_at else "",
                }
            )
            documents.append(doc)
        
        # Split documents into chunks
        log.info(f"Splitting {len(documents)} documents into chunks...")
        split_docs = self.text_splitter.split_documents(documents)
        log.info(f"Created {len(split_docs)} chunks from {len(documents)} articles")
        
        # Add chunks to ChromaDB
        try:
            self.vectorstore.add_documents(split_docs)
            # ChromaDB persists automatically, but we can force it
            if hasattr(self.vectorstore, 'persist'):
                self.vectorstore.persist()
            
            log.info(f"Successfully ingested {len(articles)} articles ({len(split_docs)} chunks)")
            return len(articles)
            
        except Exception as e:
            log.error(f"Error ingesting articles: {e}")
            return 0
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Search for relevant documents in ChromaDB.
        
        Args:
            query: Search query string
            k: Number of results to return (default: 5)
            filter: Optional metadata filter (e.g., {"source": "BBC"})
            
        Returns:
            List of Document objects with relevant content and metadata
        """
        if not query:
            log.warning("Empty query provided")
            return []
        
        log.info(f"Searching for top {k} documents matching: {query[:50]}...")
        
        try:
            # Perform similarity search
            if filter:
                results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k,
                    filter=filter,
                )
                # Unpack (doc, score) tuples
                documents = [doc for doc, score in results]
            else:
                documents = self.vectorstore.similarity_search(query, k=k)
            
            log.info(f"Found {len(documents)} relevant documents")
            return documents
            
        except Exception as e:
            log.error(f"Error searching ChromaDB: {e}")
            return []
    
    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """
        Search for relevant documents with similarity scores.
        
        Args:
            query: Search query string
            k: Number of results to return (default: 5)
            filter: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples
        """
        if not query:
            log.warning("Empty query provided")
            return []
        
        log.info(f"Searching with scores for top {k} documents matching: {query[:50]}...")
        
        try:
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=filter,
            )
            
            log.info(f"Found {len(results)} relevant documents with scores")
            return results
            
        except Exception as e:
            log.error(f"Error searching ChromaDB: {e}")
            return []
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the ChromaDB collection.
        
        Returns:
            True if successful, False otherwise
        """
        log.warning(f"Clearing collection: {self.collection_name}")
        
        try:
            # Get all document IDs and delete them
            collection = self.vectorstore._collection
            if collection:
                # Get all IDs
                results = collection.get()
                if results and 'ids' in results:
                    ids = results['ids']
                    if ids:
                        collection.delete(ids=ids)
                        log.info(f"Deleted {len(ids)} documents from collection")
                
                # Alternative: delete the entire collection and recreate
                # This is more thorough but requires recreation
                # self.vectorstore.delete_collection()
                # self.vectorstore = Chroma(
                #     collection_name=self.collection_name,
                #     embedding_function=self.embeddings,
                #     persist_directory=str(self.persist_directory),
                # )
            
            log.info("Collection cleared successfully")
            return True
            
        except Exception as e:
            log.error(f"Error clearing collection: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": str(self.persist_directory),
            }
        except Exception as e:
            log.error(f"Error getting collection info: {e}")
            return {}
    
    def remove_duplicates(self) -> Dict[str, Any]:
        """
        Remove duplicate documents from the collection based on URL.
        
        Keeps the first document for each unique URL and removes all others.
        
        Returns:
            Dictionary with removal statistics
        """
        log.info("Scanning for duplicate documents by URL...")
        
        try:
            collection = self.vectorstore._collection
            
            # Get all documents with their metadata
            results = collection.get(include=["metadatas"])
            
            if not results or not results.get('ids'):
                log.info("No documents found in collection")
                return {"duplicates_removed": 0, "unique_urls": 0}
            
            ids = results['ids']
            metadatas = results.get('metadatas', [])
            
            # Group document IDs by URL
            url_to_ids: Dict[str, List[str]] = {}
            for doc_id, metadata in zip(ids, metadatas):
                url = metadata.get('url', '') if metadata else ''
                if url:
                    if url not in url_to_ids:
                        url_to_ids[url] = []
                    url_to_ids[url].append(doc_id)
            
            # Find duplicate IDs (all IDs after the first one for each URL)
            duplicate_ids = []
            for url, doc_ids in url_to_ids.items():
                if len(doc_ids) > 1:
                    # Keep the first, mark the rest as duplicates
                    duplicate_ids.extend(doc_ids[1:])
                    log.debug(f"URL '{url[:50]}...' has {len(doc_ids)} documents, removing {len(doc_ids) - 1}")
            
            if not duplicate_ids:
                log.info("No duplicate documents found")
                return {
                    "duplicates_removed": 0,
                    "unique_urls": len(url_to_ids),
                    "total_documents_before": len(ids),
                    "total_documents_after": len(ids),
                }
            
            # Delete duplicates
            log.info(f"Removing {len(duplicate_ids)} duplicate documents...")
            collection.delete(ids=duplicate_ids)
            
            # Get new count
            new_count = collection.count()
            
            stats = {
                "duplicates_removed": len(duplicate_ids),
                "unique_urls": len(url_to_ids),
                "total_documents_before": len(ids),
                "total_documents_after": new_count,
            }
            
            log.info(f"Successfully removed {len(duplicate_ids)} duplicates")
            log.info(f"Collection now has {new_count} documents from {len(url_to_ids)} unique URLs")
            
            return stats
            
        except Exception as e:
            log.error(f"Error removing duplicates: {e}")
            return {"error": str(e)}

