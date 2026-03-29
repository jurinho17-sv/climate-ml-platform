.PHONY: run install clean lint test

run:
	streamlit run frontend/app.py

install:
	uv sync

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

lint:
	ruff check src/ api/ tests/
	ruff format --check src/ api/ tests/

test:
	pytest tests/ -v
