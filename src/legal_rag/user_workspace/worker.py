import os
from dotenv import load_dotenv
load_dotenv()

# 3. Import your application frameworks and local modules LAST
from celery import Celery
from celery.signals import worker_process_init
import logging
from src.legal_rag.user_workspace.database import update_document_status, engine
from src.legal_rag.user_workspace.user_data_embedding import orchestrator

from src.legal_rag.tracing import setup_tracing

# Import the Celery instrumentor
from opentelemetry.instrumentation.celery import CeleryInstrumentor
# Optional but highly recommended: Instrument Redis to trace DB calls
from opentelemetry.instrumentation.redis import RedisInstrumentor


# 1. Initialize Celery to use your existing Redis instance as the broker
redis_url = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}/0"

celery_app = Celery(
    "legal_rag_worker",
    broker=redis_url,
    backend=redis_url
)

# 2. Initialize OpenTelemetry inside the worker init signal
@worker_process_init.connect(weak=False)
def init_celery_tracing(*args, **kwargs):
    """
    This ensures the Tracer and its background thread are started 
    safely INSIDE each spawned worker process.
    """
    setup_tracing("legal_rag_worker")
    CeleryInstrumentor().instrument()
    RedisInstrumentor().instrument()
    logging.info("OpenTelemetry initialized for Celery worker process.")

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