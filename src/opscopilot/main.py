import logging

import psycopg
from fastapi import FastAPI

from opscopilot.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="OpsCopilot")


def db_ok(url: str) -> bool:
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except psycopg.Error:
        return False


@app.get("/health")
def health():
    # geo_db is optional (compose profile "geo"), so false there is not an error yet
    return {
        "app_db": db_ok(settings.database_url),
        "geo_db": db_ok(settings.geo_database_url),
    }
