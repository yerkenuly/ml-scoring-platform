"""Seed the database with an initial API key."""
import hashlib
import secrets
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.config import settings


def seed():
    engine = create_engine(settings.sync_database_url)

    raw_key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    with Session(engine) as session:
        session.execute(
            text("""
                INSERT INTO api_keys (id, key_hash, name, is_active, rate_limit, created_at)
                VALUES (:id, :hash, :name, true, 1000, :created_at)
                ON CONFLICT DO NOTHING
            """),
            {
                "id": str(uuid.uuid4()),
                "hash": key_hash,
                "name": "default",
                "created_at": datetime.now(timezone.utc),
            },
        )
        session.commit()

    print(f"API Key created: {raw_key}")
    print("Store this key — it will not be shown again.")


if __name__ == "__main__":
    seed()
