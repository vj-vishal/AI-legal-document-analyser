import os
from celery import Celery
import logging
from src.legal_rag.user_workspace.database import update_document_status, engine
from src.legal_rag.user_workspace.user_data_embedding import orchestrator

# 1. Initialize Celery to use your existing Redis instance as the broker
redis_url = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}/0"

celery_app = Celery(
    "legal_rag_worker",
    broker=redis_url,
    backend=redis_url
)

# 2. Register the heavy function as a Celery task
@celery_app.task(bind=True, max_retries=3)
def run_heavy_ingestion_task(self, safe_file_path: str, kb_id: str, doc_id: str, user_id: str):
    try:
        logging.info(f"Worker picked up document {doc_id} for processing.")
        
        # Execute your heavy vision-parsing and embedding pipeline
        orchestrator(
            pdf_path=safe_file_path,
            collection_name=str(kb_id), 
            kb_document_id=str(doc_id),
            kb_id=str(kb_id),
            user_id=str(user_id)
        )
        
        # Update DB to 'completed' so the frontend unlocks the chat
        update_document_status(engine, document_id=doc_id, status="completed")
        logging.info(f"Worker successfully completed document {doc_id}.")
        
    except Exception as exc:
        logging.error(f"Worker failed on document {doc_id}: {str(exc)}")
        # Update DB to 'failed' so the frontend stops spinning
        update_document_status(engine, document_id=doc_id, status="failed")
        # Optional: Tell Celery to retry the task if it was a transient error
        raise self.retry(exc=exc, countdown=60)