import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


database_url = os.environ["DATABASE_URL"]
engine_options = {"pool_pre_ping": True}
if database_url.startswith("sqlite"):
    engine_options.update(
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
engine = create_engine(database_url, **engine_options)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session() -> Generator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
