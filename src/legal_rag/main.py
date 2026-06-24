from src.legal_rag.rag.engine import LLMGenerator
from src.legal_rag.kb.main import KnowledgeBaseManager 
from src.legal_rag.utils import format_data 

query = """Under Section 60, which properties of a judgment-debtor are exempt from attachment and sale during execution of a decree?"""

generator = LLMGenerator()
manager= KnowledgeBaseManager()

chunks= manager.route_and_retrieve(query, user_id= "6544c0d7-aa3c-4dc9-b0aa-faed878d7ff3", kb_id= "0d3493be-c6b9-47dd-9387-7301b812b52a", kb_document_id= "3ce373ed-0ab6-4f7a-b55b-720e2a9445a6")

formatted_chunks= format_data(chunks)
answer = generator.generate_answer(query, formatted_chunks)

print("\nANSWER:\n", answer)