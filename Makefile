.PHONY: run serve install clean lint test warehouse

run:
	streamlit run frontend/app.py

serve:
	uvicorn api.main:app --reload --port 8000

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

warehouse:
	mkdir -p data/warehouse
	cd warehouse/co2_warehouse && dbt deps --profiles-dir ../.. && dbt run --profiles-dir ../.. && dbt test --profiles-dir ../..
