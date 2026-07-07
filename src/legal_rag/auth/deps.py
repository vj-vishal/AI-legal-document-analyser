from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.legal_rag.auth.jwt import decode_token


bearer_scheme = HTTPBearer()


def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    """
    This is a FastAPI dependency.
    
    It does two things:
    1. Extracts the Bearer token from the Authorization header automatically
    2. Calls decode_token() to validate it and get user_id
    
    Returns: user_id as a string
    Raises: HTTP 401 if token is missing, invalid, or expired
    
    How to use in any endpoint:
        user_id: str = Depends(get_current_user_id)
    
    FastAPI will:
    - Read "Authorization: Bearer eyJhbGci..." from the request header
    - Pass the token string to this function
    - Run this function before the endpoint
    - Inject the returned user_id into the endpoint
    """
    return decode_token(credentials.credentials)
