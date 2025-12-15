.PHONY: refresh-rag help rag-info rag-remove-duplicates rag-clear refresh-rag-all

# Default number of articles to fetch (can be overridden)
NUM_ARTICLES ?= 100

# Default country code
COUNTRY ?= us

help:
	@echo "Available commands:"
	@echo ""
	@echo "  Ingestion:"
	@echo "    make refresh-rag              - Refresh RAG with articles (default: 100 articles)"
	@echo "    make refresh-rag NUM_ARTICLES=50  - Refresh with custom number of articles"
	@echo "    make refresh-rag COUNTRY=gb   - Refresh with articles from specific country"
	@echo "    make refresh-rag-clear        - Clear existing collection and refresh"
	@echo "    make refresh-rag-all          - Clear collection and fetch from ALL categories"
	@echo ""
	@echo "  Management:"
	@echo "    make rag-info                 - Show collection info (document count, etc.)"
	@echo "    make rag-remove-duplicates    - Remove duplicate documents by URL"
	@echo "    make rag-clear                - Clear all documents from collection"
	@echo ""
	@echo "Examples:"
	@echo "  make refresh-rag NUM_ARTICLES=200"
	@echo "  make refresh-rag COUNTRY=gb NUM_ARTICLES=50"
	@echo "  make refresh-rag-clear NUM_ARTICLES=50"
	@echo "  make refresh-rag-all NUM_ARTICLES=200"
	@echo "  make rag-info"
	@echo "  make rag-remove-duplicates"
	@echo ""
	@echo "Note: Categories are configured in src/pipelines/refresh_rag.py"
	@echo "      Available: business, entertainment, general, health, science, sports, technology"

refresh-rag:
	@echo "Refreshing RAG with $(NUM_ARTICLES) articles from all categories..."
	@uv run python src/pipelines/refresh_rag.py \
		--num-articles $(NUM_ARTICLES) \
		--country $(COUNTRY)

refresh-rag-clear:
	@echo "Clearing existing collection and refreshing RAG with $(NUM_ARTICLES) articles..."
	@uv run python src/pipelines/refresh_rag.py \
		--num-articles $(NUM_ARTICLES) \
		--country $(COUNTRY) \
		--clear

refresh-rag-all:
	@echo "Clearing collection and fetching $(NUM_ARTICLES) articles from ALL categories..."
	@uv run python src/pipelines/refresh_rag.py \
		--num-articles $(NUM_ARTICLES) \
		--country $(COUNTRY) \
		--clear \
		--categories business entertainment general health science sports technology

rag-info:
	@echo "Getting RAG collection info..."
	@uv run python src/pipelines/rag_manage.py info

rag-remove-duplicates:
	@echo "Removing duplicate documents from RAG..."
	@uv run python src/pipelines/rag_manage.py remove-duplicates

rag-clear:
	@echo "Clearing RAG collection..."
	@uv run python src/pipelines/rag_manage.py clear
