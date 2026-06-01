from langchain_classic.prompts import PromptTemplate

# "6. Use metadata (Title, Chapter, Section, Page) to strengthen your answer.\n"

_MEDICAL_PREFIX = (
    "You are a medical research assistant.\n"
    "Answer ONLY using the provided context.\n"
    "Do NOT use prior knowledge.\n"
    "Do NOT infer beyond the given text.\n"
    "If the answer is not explicitly stated, say: 'I don't know based on the provided documents.'\n\n"
)

_BASE_TEMPLATE = (
    "You are given medical document excerpts and a question.\n\n"
    
    "INSTRUCTIONS:\n"
    "1. Use ONLY the provided context.\n"
    "2. Support every key statement with evidence from the documents.\n"
    "3. Prefer specific and detailed sources over general ones.\n"
    "4. If multiple documents agree, combine their evidence.\n"
    "5. If documents conflict, mention the conflict.\n"
    "6. Do NOT generate a SOURCES section.\n\n"

    "QUESTION:\n{question}\n\n"
    
    "CONTEXT:\n"
    "=========\n"
    "{summaries}\n"
    "=========\n\n"

    "FINAL ANSWER:\n"
)

PROMPT = PromptTemplate(
    template=_MEDICAL_PREFIX + _BASE_TEMPLATE,
    input_variables=["summaries", "question"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template=(
        "Content: {page_content}\n"
        "Title: {title}\n"
        "Part: {part}\n"
        "Chapter: {chapter}\n"
        "Section: {section}\n"
        "Page: {page}\n"
    ),
    input_variables=["page_content", "title", "part", "chapter", "section", "page"]
)
