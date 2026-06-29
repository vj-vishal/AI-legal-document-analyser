from src.legal_rag.rag.engine import LLMGenerator
from src.legal_rag.kb.main import KnowledgeBaseManager 
from src.legal_rag.utils import format_data 

def chat_orchestrator(query: str, user_id: str, kb_id: str, kb_document_id: str, chat_history: list):
    # query = """Under Section 60, which properties of a judgment-debtor are exempt from attachment and sale during execution of a decree?"""

    generator = LLMGenerator()
    manager= KnowledgeBaseManager()

    chunks= manager.route_and_retrieve(query, user_id, kb_id, kb_document_id)

    formatted_chunks= format_data(chunks)
    answer = generator.generate_answer(query, formatted_chunks, chat_history)

    return answer