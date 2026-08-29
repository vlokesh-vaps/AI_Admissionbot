"""Application entry point for the admission chatbot API."""

import uvicorn
from src.api.app import app
from src.config import settings

__all__ = ["app"]

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
