.PHONY: up down build migrate seed test lint format

up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build

migrate:
	alembic upgrade head

seed:
	python scripts/seed_db.py

test:
	pytest tests/ -v --cov=app --cov-report=term-missing

lint:
	ruff check app/ tests/
	mypy app/

format:
	black app/ tests/ scripts/
	isort app/ tests/ scripts/

generate-sample:
	python scripts/generate_sample_dataset.py

worker:
	celery -A app.worker.celery_app worker --loglevel=info

beat:
	celery -A app.worker.celery_app beat --loglevel=info

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
