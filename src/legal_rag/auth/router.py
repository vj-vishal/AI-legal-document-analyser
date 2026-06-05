from fastapi import APIRouter, FastAPI, HTTPException
from src.legal_rag.auth.hashing import hash_password, verify_password
from src.legal_rag.auth.jwt import create_access_token
from src.legal_rag.auth.store import create_user, get_user
from pydantic import BaseModel, EmailStr, Field

# ─── Signup ───────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: EmailStr                          # validates it's a real email format
    password: str = Field(..., min_length=6) # at least 6 chars


class SignupResponse(BaseModel):
    user_id: str
    message: str


# ─── Login ────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# router = APIRouter(prefix="/auth", tags=["Auth"])
app= FastAPI()


@app.post("/signup", response_model=SignupResponse)
def signup(req: SignupRequest):
    if get_user(req.email):
        raise HTTPException(status_code=400, detail="Email already registered.")
    user = create_user(req.email, hash_password(req.password))
    return SignupResponse(user_id=user["user_id"], message="Signup successful")


@app.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    user = get_user(req.email)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = create_access_token(user_id=user["user_id"])
    return TokenResponse(access_token=token, token_type="bearer")