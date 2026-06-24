from sqlalchemy import select
from sqlalchemy.orm import Session
from uuid import UUID
import logging
from dotenv import load_dotenv
import os
from src.legal_rag.models import KnowledgeBase, KbDocuments, Users, ChatSessions, ChatMessages
from sqlalchemy import MetaData, Table, insert, create_engine, text
from sqlalchemy.exc import IntegrityError

load_dotenv()  

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def create_chat_session(engine, user_id: str | UUID, knowledge_base_id: str | UUID) -> dict: 

    with Session(engine) as db:
        try:
            cs = ChatSessions(
                user_id=user_id,
                knowledge_base_id=knowledge_base_id, 
                title=f"{user_id} chat session"
            )
            db.add(cs)
            db.flush() 
        
            new_chat= ChatMessages(
                session_id= cs.id,
                role= "user",
                message= "type by user",
                tokens_used= "estimated by tokenizer"
            )
            db.add(new_chat)

        except Exception as e:
            # If ANYTHING fails above, undo the entire transaction
            db.rollback()
            logging.error(f"Database transaction failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to save to database. Details: {str(e)}"
            }