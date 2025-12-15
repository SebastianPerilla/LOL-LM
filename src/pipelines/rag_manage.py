"""RAG management CLI for collection info and maintenance tasks."""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import env  # noqa: F401 - loads .env file
from services.rag import RAGService
from logger import log


def get_collection_info() -> None:
    """Display collection information."""
    log.info("=" * 60)
    log.info("RAG Collection Info")
    log.info("=" * 60)
    
    rag = RAGService()
    info = rag.get_collection_info()
    
    log.info(f"Collection Name: {info.get('collection_name', 'N/A')}")
    log.info(f"Document Count: {info.get('document_count', 'N/A')}")
    log.info(f"Persist Directory: {info.get('persist_directory', 'N/A')}")
    log.info("=" * 60)


def remove_duplicates() -> None:
    """Remove duplicate documents from the collection."""
    log.info("=" * 60)
    log.info("Removing Duplicate Documents")
    log.info("=" * 60)
    
    rag = RAGService()
    
    # Show before stats
    info_before = rag.get_collection_info()
    log.info(f"Documents before: {info_before.get('document_count', 'N/A')}")
    
    # Remove duplicates
    stats = rag.remove_duplicates()
    
    if "error" in stats:
        log.error(f"Failed to remove duplicates: {stats['error']}")
        return
    
    log.info("-" * 60)
    log.info("Results:")
    log.info(f"  Duplicates removed: {stats.get('duplicates_removed', 0)}")
    log.info(f"  Unique URLs: {stats.get('unique_urls', 0)}")
    log.info(f"  Documents before: {stats.get('total_documents_before', 0)}")
    log.info(f"  Documents after: {stats.get('total_documents_after', 0)}")
    log.info("=" * 60)


def clear_collection() -> None:
    """Clear all documents from the collection."""
    log.info("=" * 60)
    log.info("Clearing Collection")
    log.info("=" * 60)
    
    rag = RAGService()
    
    # Show before stats
    info_before = rag.get_collection_info()
    log.info(f"Documents before: {info_before.get('document_count', 'N/A')}")
    
    # Clear collection
    success = rag.clear_collection()
    
    if success:
        log.info("Collection cleared successfully")
    else:
        log.error("Failed to clear collection")
    
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="RAG collection management commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_manage.py info              # Show collection info
  python rag_manage.py remove-duplicates # Remove duplicate documents by URL
  python rag_manage.py clear             # Clear all documents
        """,
    )
    
    parser.add_argument(
        "command",
        choices=["info", "remove-duplicates", "clear"],
        help="Command to run",
    )
    
    args = parser.parse_args()
    
    if args.command == "info":
        get_collection_info()
    elif args.command == "remove-duplicates":
        remove_duplicates()
    elif args.command == "clear":
        clear_collection()


if __name__ == "__main__":
    main()

