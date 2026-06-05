from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from fastapi import HTTPException, status
import os
from dotenv import load_dotenv

load_dotenv()  

# ─── Config ────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production")

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60
# Token expires in 60 minutes — user must log in again after this


# ─── Function 1: Create a token ─────────────────────────────
def create_access_token(user_id: str) -> str:
    """
    Called after successful login.
    Takes user_id (e.g., "user-uuid-abc123")
    Returns a JWT string to send back to the user.
    """
    payload = {
        "sub": user_id,                                              
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)  
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ─── Function 2: Decode a token ──────────────────────────────
def decode_token(token: str) -> str:
    """
    Called on every protected request.
    Takes the JWT string the user sends in their request header.
    Returns user_id if token is valid.
    Raises HTTP 401 if token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        user_id: str = payload.get("sub")  # extract user_id from "sub" field
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing user identity"
            )
        
        return user_id  
    
    except JWTError:
        # Covers: expired token, wrong signature, malformed token
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
if __name__ == "__main__":
    # Example usage
    user_id = "user-abc123"
    token = create_access_token(user_id)
    print(f"Generated token: {token}")  

    # Decode the token
    decoded_user_id = decode_token(token)
    print(f"Decoded user ID: {decoded_user_id}")
    