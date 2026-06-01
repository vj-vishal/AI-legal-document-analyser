from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder


encoder= HuggingFaceEncoder(
    name="BAAI/bge-small-en-v1.5",
    device="cuda"
)

kb=Route(
    name="knowledge_base",
    utterances=[
        "Explain the refund policy in simple terms.",
        "What does clause 4.2 about termination mean?",
        "Is the employer allowed to terminate without notice under this agreement?",
        "What are the eligibility conditions for claiming damages?",
        "How is interest calculated on delayed payments in this contract?",
        "What is the difference between indemnity and warranty in this document?",
        "Summarize the main obligations of the tenant.",
        "Which section deals with dispute resolution and arbitration?",
        "What rights does the consumer have under this policy?",
        "Under what conditions can the contract be terminated early?"
    ]
)

temp=Route(
    name="template_doc",
    utterances=[
        "Draft a notice for termination of employment based on company policy.",
        "Create a template for a legal notice to tenant for non‑payment of rent.",
        "Give me a template for a service agreement between company and contractor.",
        "Provide a format for a consent letter for medical treatment.",
        "Generate a template for a non‑disclosure agreement (NDA).",
        "I need a standard format for a power of attorney.",
        "Share a template for a legal disclaimer for a website.",
        "Give me a sample draft of a partnership agreement.",
        "Provide a template for a demand notice for outstanding dues.",
        "Create a basic template for a leave and license agreement."
    ]
)

router = SemanticRouter(routes=[kb,temp], encoder=encoder,auto_sync="local")


if __name__=="__main__":
    print(router("give me the structure of the nda template?").name)
    print(router("Who is eligible to receive legal services according to Section 12 of the Legal Services Authorities Act, 1987?").name)
    


