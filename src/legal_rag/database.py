from sqlalchemy import select, desc
from sqlalchemy.orm import Session
from uuid import UUID
import logging
from dotenv import load_dotenv
import os
from src.legal_rag.models import KnowledgeBase, KbDocuments, Users, ChatSessions, ChatMessages, Analyses
from sqlalchemy import MetaData, Table, insert, create_engine, text
from sqlalchemy.exc import IntegrityError
from langchain_groq import ChatGroq

load_dotenv()  

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

llm = ChatGroq(
            model= "openai/gpt-oss-20b",
            temperature= 0.7,
            max_tokens= 20,
            streaming= False
        )

def create_new_session(engine, user_id: str, knowledge_base_id: str):
    """Initializes a new chat container before any messages are sent.""" 

    with Session(engine) as db:
        try:
            new_session = ChatSessions(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            title="New Conversation" 
            )
    
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
    
            # Return the session ID to the frontend so it can route the user to the active chat UI
            return new_session.id

        except Exception as e:
            # If ANYTHING fails above, undo the entire transaction
            db.rollback()
            logging.error(f"Database transaction failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to save to database. Details: {str(e)}"
            }
        
def update_session_title(engine, session_id: str, first_user_message: str):
    """Generates a short title based on the first query and updates the DB."""
    
    with Session(engine) as db:
        try:
            # 1. Fetch the session
            query = select(ChatSessions).where(ChatSessions.id == session_id)
            result= db.execute(query)

            document = result.scalars().first()
            
            if not document:
                return None

            # 2. Ask your LLM to generate a short title (e.g., using Llama-3)
            prompt = f"Summarize this query into a 3-5 word title: {first_user_message}"
            generated_title_response = llm.invoke(prompt)
            
            generated_title = generated_title_response.content

            generated_title = generated_title.strip().replace('"', '')
            
            # 3. Update the ORM model and commit
            document.title = generated_title
            db.commit()
            
            return {
                    "status": "success",
                    "message": f"Session {session_id} title updated to '{generated_title}'."
                }
        
        except Exception as e:
            db.rollback() # Good practice to roll back if the transaction gets interrupted
            logging.error(f"Database error during updating {session_id} title: {str(e)}")
            raise e
        
def log_user_query(engine, session_id: str, query: str, token: int):
    """Saves the user's message to the database."""

    with Session(engine) as db:
        try:
            new_message = ChatMessages(
            session_id=session_id,
            role="user",
            message=query,
            tokens_used= token
        )
        
            db.add(new_message)
            db.commit()
            db.refresh(new_message) # Fetches the auto-generated ID/timestamps
            
            return new_message

        except Exception as e:
            # If ANYTHING fails above, undo the entire transaction
            db.rollback()
            logging.error(f"Database transaction failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to save to database. Details: {str(e)}"
            }

def get_chat_history(engine, session_id: str, limit: int = 5):
    """Fetches the last N messages for context."""

    with Session(engine) as db:
        try:
            query = select(ChatMessages).where(ChatMessages.session_id == session_id).order_by(desc(ChatMessages.created_at)).limit(limit)
            result= db.execute(query)

            documents = result.scalars().all()

            if not documents:
                return "No prior conversation history."

            # Reverse them to chronological order (oldest first)
            chronological_messages = reversed(documents)
            
            # Extract the text into a clean format
            formatted_history = ""
            for doc in chronological_messages:
                # Capitalize the role (User/Assistant) and append the message
                formatted_history += f"{doc.role.capitalize()}: {doc.message}\n\n"
                
            return formatted_history.strip()
            
        except Exception as e:
            logging.error(f"Database error during retrieval of user messages for {session_id}: {str(e)}")
            return ""
        
def log_ai_response(engine, session_id: str, response_text: str, tokens: int):
    """Saves the AI's response to the database."""
    
    with Session(engine) as db:
        try:
            new_message = ChatMessages(
                session_id=session_id,
                role="assistant",
                message=response_text,
                tokens_used=tokens
            )
            
            db.add(new_message)
            db.commit()
            db.refresh(new_message)
            
            return new_message
        
        except Exception as e:
            # If ANYTHING fails above, undo the entire transaction
            db.rollback()
            logging.error(f"Database transaction failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to save to database. Details: {str(e)}"
            }
        
def log_analysis_record(engine, user_id: str, kb_document_id: str, session_id: str, query: str, response: str, sources: list, confidence_score: float):
    """Logs the detailed RAG retrieval data for audit and citation purposes."""
    with Session(engine) as db:
        try:

            new_analysis = Analyses(
                user_id=user_id,
                document_id=kb_document_id,
                session_id=session_id,
                query=query,
                response=response,
                sources_used=sources,
                confidence_score=confidence_score
            )
            
            db.add(new_analysis)
            db.commit()
            return new_analysis
        
        except Exception as e:
            # If ANYTHING fails above, undo the entire transaction
            db.rollback()
            logging.error(f"Database transaction failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to save to database. Details: {str(e)}"
            }
        
def get_chat_session_view(engine, user_id: str | UUID) -> list[dict]:
    with Session(engine) as db:
        try:
            query = select(ChatSessions).where(ChatSessions.user_id == user_id)#.order_by(ChatSessions.updated_at.desc())
            result = db.execute(query)

            docs = result.scalars().all()
            return docs

        except Exception as e:
            logging.error(f"Database error during retrieval of user documents for {user_id}: {str(e)}")
            return None

def get_chat_message(engine, session_id: str | UUID) -> list[dict]:
    with Session(engine) as db:
        try:
            query = select(ChatMessages).where(ChatMessages.session_id == session_id)
            result = db.execute(query)

            docs = result.scalars().all()
            return docs

        except Exception as e:
            logging.error(f"Database error during retrieval of user documents for {session_id}: {str(e)}")
            return None

def get_user_profile(engine, user_id: str | UUID) -> list[dict]:
    with Session(engine) as db:
        try:
            query = select(Users).where(Users.id == user_id)
            result = db.execute(query)

            docs = result.scalars().first()
            return docs

        except Exception as e:
            logging.error(f"Database error during retrieval of user documents for {user_id}: {str(e)}")
            return None