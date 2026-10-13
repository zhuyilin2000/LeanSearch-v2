"""Shared PostgreSQL helpers for the pg backend.

Connection settings come from the ``pg.*`` config section (config.yaml /
config.local.yaml), each overridable by the matching upper-case env var
(PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD). The password is read
from the inline ``pg.password`` value or, failing that, from the env var
named by ``pg.password_env`` (default ``LEANSEARCH_PG_PASSWORD``).

Every connection opened through :func:`connect` or :func:`pool` registers
the pgvector psycopg adapter, so ``pgvector.Vector`` values can be bound
and read directly.
"""

from __future__ import annotations

import os
from typing import Any

import psycopg
from psycopg_pool import ConnectionPool

from .config import get


def conninfo_from_config(*, dbname: str | None = None) -> dict[str, Any]:
    password = get("PG_PASSWORD", "pg", "password", default=None)
    if password in (None, ""):
        password_env = str(
            get("PG_PASSWORD_ENV", "pg", "password_env", default="LEANSEARCH_PG_PASSWORD")
        )
        password = os.environ.get(password_env)
    return {
        "host": str(get("PG_HOST", "pg", "host", default="localhost")),
        "port": int(get("PG_PORT", "pg", "port", default=5432)),
        "dbname": dbname
        or str(get("PG_DBNAME", "pg", "dbname", default="leansearch")),
        "user": str(get("PG_USER", "pg", "user", default="deepanalyze")),
        "password": password,
    }


def _configure(conn: psycopg.Connection) -> None:
    from pgvector.psycopg import register_vector

    register_vector(conn)


def connect(*, dbname: str | None = None, autocommit: bool = False) -> psycopg.Connection:
    conn = psycopg.connect(**conninfo_from_config(dbname=dbname), autocommit=autocommit)
    _configure(conn)
    return conn


def pool(*, min_size: int = 1, max_size: int = 8, open: bool = True) -> ConnectionPool:
    return ConnectionPool(
        kwargs=conninfo_from_config(),
        min_size=min_size,
        max_size=max_size,
        open=open,
        configure=_configure,
    )
