from src.legal_rag.rag.engine import LLMGenerator
from src.legal_rag.kb.main import KnowledgeBaseManager 
from src.legal_rag.utils import format_data

# 1. Import trace from OpenTelemetry
from opentelemetry import trace

# 2. Initialize the tracer for this module
tracer = trace.get_tracer("legal_rag_api.chat_orchestrator")

def chat_orchestrator(query: str, user_id: str, kb_id: str, kb_document_id: str, chat_history: list, manager: KnowledgeBaseManager):
    # query = """Under Section 60, which properties of a judgment-debtor are exempt from attachment and sale during execution of a decree?"""

    # with tracer.start_as_current_span("initiating_kb_manager") as manager_span:   
    #     manager= KnowledgeBaseManager()

    with tracer.start_as_current_span("chunk_retrieval") as retriever_span:
        chunks= manager.route_and_retrieve(query, user_id, kb_id, kb_document_id)

    formatted_chunks= format_data(chunks)

    with tracer.start_as_current_span("initiating_generator") as generator_span:
        generator = LLMGenerator()

    with tracer.start_as_current_span("get_answer") as get_answer_span:
        answer = generator.generate_answer(query, formatted_chunks, chat_history)

    return answer