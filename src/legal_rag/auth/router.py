from fastapi import APIRouter, FastAPI, HTTPException, Depends
from src.legal_rag.auth.hashing import hash_password, verify_password
from src.legal_rag.auth.jwt import create_access_token, decode_token
from src.legal_rag.auth.deps import get_current_user_id
from pydantic import BaseModel, EmailStr, Field
from src.legal_rag.database import get_user_by_email, load_user_data, engine, load_knowledge_data, load_chat_session_data

# ─── Signup ───────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: EmailStr                          
    password: str = Field(..., min_length=6) 
    name: str = Field(..., min_length=3)


class SignupResponse(BaseModel):
    # user_id: str
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
    if get_user_by_email(engine, req.email):
        raise HTTPException(status_code=400, detail="Email already registered.")
    password_hash = hash_password(req.password)
    result_message = load_user_data(engine, req.name, req.email, password_hash, role="admin")
    return SignupResponse(message=result_message)


@app.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    # user = get_user(req.email)
    result = get_user_by_email(engine, req.email)
    hash_password = result["password_hash"] if result else None
    if not req.email or not verify_password(req.password, hash_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = create_access_token(user_id=str(result["id"]))
    return TokenResponse(access_token=token, token_type="bearer")
