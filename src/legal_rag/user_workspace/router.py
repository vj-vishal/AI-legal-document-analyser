import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from typing import Annotated, Dict
import os
from src.legal_rag.user_workspace.database import get_or_create_kb_data, engine, load_user_data, get_user_by_email, update_document_status, get_user_kb_docs
from src.legal_rag.user_workspace.user_data_embedding import orchestrator
from src.legal_rag.auth.deps import get_current_user_id
from src.legal_rag.auth.hashing import hash_password, verify_password
from src.legal_rag.auth.jwt import create_access_token, decode_token
from pydantic import BaseModel, EmailStr, Field
from src.legal_rag.config import USER_KB_DIR
from fastapi.middleware.cors import CORSMiddleware
from src.legal_rag.database import create_new_session, update_session_title, log_user_query, get_chat_history, log_ai_response, log_analysis_record, get_chat_session_view, get_chat_message, get_user_profile
from src.legal_rag.main import chat_orchestrator
from pydantic import BaseModel
from typing import Optional
from fastapi import Depends, HTTPException, status
import logging
from dotenv import load_dotenv
import traceback
from src.legal_rag.guardrails.pre_guardrail import run_guardrail
from src.legal_rag.utils import count_tokens_locally
from src.legal_rag.rate_limit.usage_limiter import check_and_reserve, adjust_actual_usage, _day_key, _month_key
from src.legal_rag.rate_limit.config import FREE_LIMITS, ESTIMATED_CHAT_COST
from src.legal_rag.rate_limit.redis_client import redis_client
import redis

load_dotenv()

INTERNAL_KB_ID = os.getenv("INTERNAL_KB_ID")

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True,
)

# ─── Signup ───────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: EmailStr                          
    password: str = Field(..., min_length=6) 
    name: str = Field(..., min_length=3)


class SignupResponse(BaseModel):
    # user_id: str
    message: Dict[str, str] 


# ─── Login ────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

app= FastAPI()

# Add this entire block right below app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # This is the VIP pass for React
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (POST, GET, etc.)
    allow_headers=["*"], # Allows all headers
)

STORAGE_BASE_DIR = USER_KB_DIR

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
    hash_password = result.password_hash if result else None
    if not req.email or not verify_password(req.password, hash_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = create_access_token(user_id=str(result.id))
    return TokenResponse(access_token=token, token_type="bearer")


@app.post("/load_kb")
def load_kb(file: Annotated[UploadFile, File()],
            user_id: str = Depends(get_current_user_id)):
    
    # ── rate limit check: upload quota ──
    check_and_reserve(
        user_id=user_id,
        resource="upload",
        amount=1,
        daily_limit=FREE_LIMITS["upload_daily"],
        monthly_limit=FREE_LIMITS["upload_monthly"],
    )
    
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    pdf_bytes = file.file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty PDF uploaded")
    
    try:
        # 3. RUN FUNCTION 1: Get data with verified user_id from the token
        db_result = get_or_create_kb_data(
            engine=engine, 
            user_id=user_id, 
            filename=file.filename
        )

        if db_result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=db_result.get("message", "Database operations failed.")
            )

        kb_id = db_result["knowledge_base_id"]
        doc_id = db_result["document_id"]

        # 4. Save file to tenant-isolated directory
        kb_folder_path = os.path.join(STORAGE_BASE_DIR, str(kb_id))
        os.makedirs(kb_folder_path, exist_ok=True) 
        
        safe_file_path = os.path.join(kb_folder_path, f"{doc_id}.pdf")

        with open(safe_file_path, "wb") as local_file:
            local_file.write(pdf_bytes)

        orchestrator(
            pdf_path=safe_file_path,
            collection_name=str(kb_id), 
            kb_document_id=str(doc_id),
            kb_id=str(kb_id),
            user_id=str(user_id)
        )

        update_document_status(engine, document_id=doc_id)

        return {
            "status": "success",
            "message": "File saved locally and successfully ingested into the vector database.",
            "data": {
                "knowledge_base_id": str(kb_id),
                "document_id": str(doc_id),
                "saved_path": safe_file_path
            }
        }

    except HTTPException as he:
        # Catch and re-raise anticipated HTTP errors (like 400 Bad Request)
        raise he
    except Exception as e:
        # Catch unexpected crashes during orchestration or database writes
        logging.error(f"Pipeline processing failure for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while running ingestion: {str(e)}"
        )

@app.get("/user_kb_docs")
def user_kb_docs( user_id: str = Depends(get_current_user_id)):
    try:
        docs= get_user_kb_docs(engine, user_id= user_id)
        if docs:
            return {
                "status": "success",
                "message": "User documents retrieved successfully.",
                "data": [{
                    "document_id": str(doc.id),
                    "knowledge_base_id": str(doc.knowledge_base_id),
                    "title": doc.title,
                    "document_type": doc.document_type,
                    "status": doc.status,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "updated_at": doc.updated_at.isoformat() if doc.updated_at else None
                } for doc in docs]
            }
    except Exception as e:
        logging.error(f"Error retrieving user documents for {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving user documents: {str(e)}"
        )

# Define the Expected JSON Payload Structure
class ChatRequest(BaseModel):
    query: str
    knowledge_base_id: Optional[str] = None
    kb_document_id: Optional[str] = None
    session_id: Optional[str] = None

@app.post("/chat")
def chat(request: ChatRequest, user_id: str = Depends(get_current_user_id)):

    # ── rate limit check: chat token quota (pre-check with estimate) ──
    check_and_reserve(
        user_id=user_id,
        resource="chat_tokens",
        amount=ESTIMATED_CHAT_COST,
        daily_limit=FREE_LIMITS["chat_tokens_daily"],
        monthly_limit=FREE_LIMITS["chat_tokens_monthly"],
    )
    
    # 2. Unpack the variables so the rest of your code works perfectly without changes
    query = request.query
    knowledge_base_id = request.knowledge_base_id
    kb_document_id = request.kb_document_id
    session_id = request.session_id

    if not knowledge_base_id and not kb_document_id:
        knowledge_base_id= INTERNAL_KB_ID
        kb_document_id= None    

    try:
        is_new_session = False

        # 1. Session Strategy Logic
        if not session_id or session_id == "new":
            # If no session exists, create a new one
            session_id = create_new_session(
                engine, 
                user_id=user_id, 
                knowledge_base_id=knowledge_base_id
            )
            is_new_session = True

        # 3. Dynamic Titling (Only triggers on the very first message)
        if is_new_session:
            update_session_title(engine, session_id, first_user_message=query)

        initial_response= run_guardrail(query)
        if initial_response.get("gate")=="fail":
            response_text= initial_response.get("response")
            log_ai_response(engine, session_id, response_text, tokens=query_token_count)
            # ── refund the reserved estimate since LLM was never called ──
            day_adjusted_token, month_adjusted_token= adjust_actual_usage(user_id, "chat_tokens", delta=-ESTIMATED_CHAT_COST)
            return {
                "status": "fail",
                "session_id": session_id,
                "message": "Guardrail triggered. Query not processed.",
                "answer": response_text,
                "day_adjusted_token": FREE_LIMITS["chat_tokens_daily"] - day_adjusted_token,
                "month_adjusted_token": FREE_LIMITS["chat_tokens_monthly"] - month_adjusted_token
            }

        # 4. Fetch Conversation Memory (Crucial for multi-turn chat)
        chat_history = get_chat_history(engine, session_id, limit=5)

        history_token_count = count_tokens_locally(chat_history, model_name="gpt-4o") 

        query_token_count = count_tokens_locally(query, model_name="gpt-4o") 

        total_estimated_tokens = query_token_count + history_token_count 

        # 2. Log User Query immediately to the current session
        log_user_query(engine, session_id=session_id, query=query, token=total_estimated_tokens)

        # 5. Execute RAG Pipeline / LLM Orchestration
        # Pass chat_history into your orchestrator so the LLM remembers previous turns
        response_text = chat_orchestrator(
            query=query, 
            user_id=user_id, 
            kb_id=knowledge_base_id, 
            kb_document_id=kb_document_id,
            chat_history=chat_history 
        )

        response_token_count = count_tokens_locally(response_text, model_name="gpt-4o") 

        # 6. Log AI Response & Audit Trail
        log_ai_response(engine, session_id, response_text, tokens=response_token_count)

        # ── correct the reservation with actual tokens used ──
        actual_tokens = query_token_count + response_token_count + history_token_count 
        day_adjusted_token, month_adjusted_token = adjust_actual_usage(user_id, "chat_tokens", delta=actual_tokens - ESTIMATED_CHAT_COST)

        # Only log to the audit analysis table if they are actually querying a document
        if knowledge_base_id and kb_document_id:
            log_analysis_record(
                engine, 
                user_id, 
                kb_document_id, 
                session_id, 
                query, 
                response=response_text, 
                sources={"S1": "doc1", "S2": "doc2"}, 
                confidence_score=0.8
            )

        # 7. Return payload to frontend
        return {
            "status": "success",
            "session_id": session_id, # Send this back so frontend knows what ID to use next time
            "message": "Data logged successfully",
            "answer": response_text,
            "day_adjusted_token": FREE_LIMITS["chat_tokens_daily"] - day_adjusted_token,
            "month_adjusted_token": FREE_LIMITS["chat_tokens_monthly"] - month_adjusted_token
        }

    # except HTTPException as he:
    #     raise he
    # except Exception as e:
    #     logging.error(f"Pipeline processing failure for user {user_id}: {str(e)}")
    #     raise HTTPException(
    #         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #         detail=f"An error occurred while providing query response: {str(e)}"
    #     )
    
    except Exception as e:
    
        traceback.print_exc()   # <-- shows exact file + line number
        
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )

@app.get("/chat_session_view")
def chat_session_view( user_id: str = Depends(get_current_user_id)):
    try:
        sessions= get_chat_session_view(engine, user_id= user_id)
        if sessions:
            return {
                "status": "success",
                "message": "User documents retrieved successfully.",
                "data": [
                    {
                        "id": str(session.id), 
                        "title": session.title,
                        "knowledge_base_id": str(session.knowledge_base_id)
                    } 
                    for session in sessions
                         ]
            }
    except Exception as e:
        logging.error(f"Error retrieving user documents for {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving user documents: {str(e)}"
        )

@app.get("/chat_message_view")
def chat_message_view( session_id, user_id: str = Depends(get_current_user_id)):
    try:
        messages= get_chat_message(engine, session_id= session_id)
        if messages:
            return {
                "status": "success",
                "message": "User documents retrieved successfully.",
                "data": [
                    {
                        "id": str(message.id), 
                        "role": message.role,
                        "messages": message.message,
                        "token_used": message.tokens_used
                    } 
                    for message in messages
                         ]
            }
    except Exception as e:
        logging.error(f"Error retrieving user documents for {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving user documents: {str(e)}"
        )
    
@app.get("/user_profile")
def user_profile(user_id: str = Depends(get_current_user_id)):
    try:
        docs= get_user_profile(engine, user_id)
        if docs:
            return {
                "status": "success",
                "message": "User documents retrieved successfully.",
                "data": [
                    { 
                        "name": docs.name,
                        "role": docs.role
                    } 
                         ]
            }
    except Exception as e:
        logging.error(f"Error retrieving user profile for {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving user profile: {str(e)}"
        )
    
@app.get("/rate_limit_status")
async def get_rate_limit_status(user_id: str = Depends(get_current_user_id)):
    """
    Fetches the current remaining chat credits for the logged-in user.
    """
    resource = "chat_queries" 
    day_limit = 1000
    
    try:
        # 2. Construct the exact Redis key
        day_key = _day_key(user_id=user_id, resource=resource)
        
        # 3. Fetch the current usage from Redis
        used_day = redis_client.get(day_key)
        
        # 4. Calculate the remaining credits
        if used_day is None:
            # Key doesn't exist yet, meaning 0 queries used this day
            remaining_credits = day_limit
            used_amount = 0
        else:
            used_amount = int(used_day)
            remaining_credits = max(0, day_limit - used_amount)

        return {
            "status": "success",
            "remaining_credits": remaining_credits,
            "used_credits": used_amount,
            "daily_limit": day_limit
        }

    except Exception as e:
        # Fallback to prevent UI crash if Redis is temporarily unreachable
        print(f"Redis connection error: {e}")
        return {
            "status": "error",
            "remaining_credits": day_limit, 
            "detail": "Could not fetch current usage."
        }