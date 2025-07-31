import os
from pathlib import Path

from fastapi import APIRouter, Response
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/favicon.ico")
async def favicon():
    """Serve favicon.ico from static/favicon directory"""
    favicon_path = Path("static/favicon/favicon.ico")
    if favicon_path.exists():
        return FileResponse(favicon_path)
    else:
        # Return empty response if favicon doesn't exist yet
        return Response(status_code=204)
