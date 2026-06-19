import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from typing import Annotated
import os
from src.legal_rag.user_workspace.database import get_or_create_kb_data, engine, load_user_data, get_user_by_email, update_document_status, get_user_kb_docs
from src.legal_rag.user_workspace.user_data_embedding import orchestrator
from src.legal_rag.auth.deps import get_current_user_id
from src.legal_rag.auth.hashing import hash_password, verify_password
from src.legal_rag.auth.jwt import create_access_token, decode_token
from pydantic import BaseModel, EmailStr, Field
from src.legal_rag.config import USER_KB_DIR

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

app= FastAPI()

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
            kb_id=str(kb_id)
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