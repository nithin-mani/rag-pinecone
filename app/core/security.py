from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.core.config import get_settings

settings = get_settings()
_api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    if not api_key or api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key