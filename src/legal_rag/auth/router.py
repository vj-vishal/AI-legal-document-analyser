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

# ─── Load Knowledge Base ────────────────────────────────────────────────

class LoadKBRequest(BaseModel):
    name: str = Field(..., min_length=3)
    description: str = Field(..., min_length=5)


class LoadKBResponse(BaseModel):
    message: str

# ─── Load Chat Session ────────────────────────────────────────────────

class LoadChatSessionRequest(BaseModel):
    title: str = Field(..., min_length=3)


class LoadChatSessionResponse(BaseModel):
    message: str

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


@app.post("/load_kb", response_model=LoadKBResponse)
def load_kb(req: LoadKBRequest, user_id: str = Depends(get_current_user_id)):
    result_message= load_knowledge_data(engine, user_id=user_id, name=req.name, description=req.description, is_active=True)
    return LoadKBResponse(message=result_message)


@app.post("/load_chat_session", response_model=LoadChatSessionResponse)
def load_chat_session(req: LoadChatSessionRequest, user_id: str = Depends(get_current_user_id), knowledge_base_id: str = "a0f24d4f-9bea-4776-a721-315c63bb821d"):# change knowledge_base_id 
    result_message= load_chat_session_data(engine, user_id=user_id, knowledge_base_id=knowledge_base_id, title=req.title)
    return LoadChatSessionResponse(message=result_message)