from src.legal_rag.rag.engine import LLMGenerator
from src.legal_rag.kb.main import KnowledgeBaseManager 
from src.legal_rag.utils import format_data 

query = """briefly explain section 302 and its punishment?"""

generator = LLMGenerator()
manager= KnowledgeBaseManager()

chunks= manager.route_and_retrieve(query)

formatted_chunks= format_data(chunks)
answer = generator.generate_answer(query, formatted_chunks)

print("\nANSWER:\n", answer)