import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.knowledge_bases import router as knowledge_bases_router

app = FastAPI(
    title="AI Knowledge Workspace API",
    description="Core API for the AI Knowledge Workspace.",
    version="0.1.0",
)

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(knowledge_bases_router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Report whether the API process is ready to serve requests."""
    return {"status": "ok"}
