import json
from langchain_groq import ChatGroq
from typing import List, Dict
from dotenv import load_dotenv
import src.legal_rag.config as config

load_dotenv()

llm = ChatGroq(
            model= "llama-3.3-70b-versatile",
            temperature= 0.2,
            max_tokens= 2000,
            streaming=False
        )

def _clean_json(content: str) -> str:
    return content.replace("```json", "").replace("```", "").strip()

def extract_data_from_raw_text(chunk_text: str) -> Dict[str, any]:
    
    prompt = f"""
You are an expert Legal Document Intelligence Agent. 
Your task is to analyze the provided legal text and extract its core components into a universal, structured format.

EXTRACTION & FORMATTING RULES:
1. ARRAYS ARE DYNAMIC: If a document has no financial terms (like an NDA), return an empty array [] for "financial_details". DO NOT hallucinate data.
2. DATE NORMALIZATION: All dates MUST be converted to strict ISO 8601 format (YYYY-MM-DD). If a date is relative, calculate it based on the execution date. If a date is completely unknown, use null.
3. OBJECTIVITY: Summarize clauses neutrally. Do not offer legal advice.
4. STRICT JSON: Return ONLY valid JSON matching the exact schema below. No markdown, no preambles.

JSON SCHEMA:
{{
  "document_metadata": {{
    "inferred_title": "string",
    "document_category": "string (e.g., Contract, Notice, Corporate Record, Court Filing)",
    "governing_law_jurisdiction": "string | null",
    "executive_summary": "string (2-3 sentences explaining the core purpose)"
  }},
  "parties_involved": [
    {{
      "entity_name": "string",
      "legal_role": "string (e.g., Disclosing Party, Lessee, Plaintiff, Employer)",
      "entity_type": "string (e.g., Individual, Corporation, Government)"
    }}
  ],
  "critical_dates": [
    {{
      "event_description": "string (e.g., Execution Date, Expiration Date, Notice Period)",
      "normalized_date": "YYYY-MM-DD | null"
    }}
  ],
  "financial_details": [
    {{
      "category": "string (e.g., Base Salary, Monthly Rent, Settlement Amount, Penalty)",
      "amount": "float",
      "currency": "string"
    }}
  ],
  "extracted_clauses": [
    {{
      "clause_theme": "string (e.g., Termination, Confidentiality, Non-Compete, Force Majeure)",
      "brief_summary": "string"
    }}
  ],
  "risk_and_compliance": {{
    "is_signed": "boolean | null",
    "unusual_or_high_risk_terms": [
      "string (List any severe penalties, perpetual terms, or one-sided obligations. If none, leave empty)"
    ]
  }}
}}

DOCUMENT TEXT:
{chunk_text}
"""

    try:
        response = llm.invoke(prompt)
        content  = _clean_json(response.content)
        
        parser = json.loads(content)
        return parser
        
    except Exception as e:
        print(f"Error generating LLM judgment: {e}")

if __name__ == "__main__":
  
  chunk_text = """
    [Page 1]
Non-Disclosure Agreement (NDA)
Disclaimer: This is a fictional sample document created for software testing, demo,
and prototyping purposes only. It is not legal advice and should not be used as a real
legal agreement without review by a qualified lawyer.
This Non-Disclosure Agreement (the "Agreement") is entered into as of 30 April 2026, by
and between:
• Disclosing Party: Asterion Analytics Private Limited, having its principal office at
14 Park Street, 5th Floor, Kolkata, West Bengal 700016, India
• Receiving Party: Neel Verma Consulting LLP, having its office at 221 Residency
Road, Bengaluru, Karnataka 560025, India
The parties agree as follows:
1. Purpose
The parties wish to discuss and evaluate a potential collaboration involving AI
document processing, confidential product design, retrieval systems, and enterprise
software integration, and in connection with that purpose may disclose certain
confidential or proprietary information.
2. Definition of Confidential Information
For purposes of this Agreement, "Confidential Information" means any non-public
information disclosed by the Disclosing Party to the Receiving Party, whether in written,
oral, electronic, visual, or any other form, including but not limited to:
• Business plans, financial projections, pricing models, and customer strategies
• Source code, model pipelines, prompts, embeddings, datasets, APIs,
architecture diagrams, and technical documentation
• Product specifications, unreleased features, prototypes, experiments, and
internal reports
• Client records, vendor terms, operational workflows, and contract details
• Notes, summaries, analyses, extracts, and derivative materials prepared from
such information
Confidential Information does not include information that the Receiving Party can
demonstrate:
• Was publicly available at the time of disclosure or later becomes publicly
available without breach of this Agreement
• Was already lawfully known to the Receiving Party before disclosure

[Page 2]
• Was independently developed without use of or reference to the Confidential
Information
• Was lawfully obtained from a third party without restriction on disclosure
3. Obligations of Receiving Party
The Receiving Party shall:
• Use the Confidential Information solely for the Purpose stated in this Agreement
• Protect the Confidential Information using at least reasonable care, and no less
than the care used to protect its own similar confidential information
• Not disclose the Confidential Information to any third party except to employees,
contractors, advisors, or affiliates who have a strict need to know for the Purpose
and who are bound by confidentiality obligations at least as protective as those
in this Agreement
• Not copy, reproduce, reverse engineer, decompile, disassemble, or otherwise
misuse the Confidential Information except as expressly permitted in writing by
the Disclosing Party
• Promptly notify the Disclosing Party of any unauthorized use or disclosure of
which it becomes aware
4. Permitted Disclosure
If the Receiving Party is required by law, regulation, or court order to disclose
Confidential Information, it may do so only to the extent legally required and, where
permitted, shall provide prompt written notice to the Disclosing Party so that protective
measures may be sought.
5. Term and Survival
This Agreement begins on 30 April 2026 and continues for 3 years, unless earlier
terminated in writing by either party. The Receiving Party's duty to protect Confidential
Information shall continue for 5 years after termination of this Agreement, or for so long
as the information remains a trade secret under applicable law, whichever period is
longer to the extent permitted by law.
6. Return or Destruction of Materials
Upon written request of the Disclosing Party, the Receiving Party shall promptly return
or destroy all Confidential Information and all copies, extracts, and summaries of it,
except for archival backups maintained in the ordinary course of business or records
required by law.
7. Ownership and No License

[Page 3]
All Confidential Information remains the property of the Disclosing Party. No license,
assignment, transfer, or other rights in any intellectual property are granted under this
Agreement except the limited right to use the Confidential Information for the Purpose.
8. No Warranty
All Confidential Information is provided "as is" without any representation or warranty,
express or implied, regarding its accuracy or completeness.
9. Remedies
The parties acknowledge that unauthorized use or disclosure of Confidential
Information may cause irreparable harm for which monetary damages may be
inadequate. Accordingly, the Disclosing Party may seek injunctive or equitable relief, in
addition to any other remedies available at law or in equity.
10. Governing Law and Jurisdiction
This Agreement shall be governed by the laws of India. Subject to applicable law, the
courts at Kolkata, West Bengal shall have exclusive jurisdiction over disputes arising out
of or in connection with this Agreement.
11. General Provisions
• This Agreement constitutes the entire understanding between the parties
regarding its subject matter and supersedes prior discussions or agreements on
that subject.
• Any amendment must be in writing and signed by both parties.
• If any provision is found unenforceable, the remaining provisions shall remain in
effect.
• Failure to enforce any provision shall not constitute a waiver of future
enforcement.
• Neither party may assign this Agreement without prior written consent, except in
connection with a merger, acquisition, or sale of substantially all assets.
12. Signatures
Disclosing Party
Asterion Analytics Private Limited
Name: Rohan Sen
Title: Director, Product Strategy
Digital Signature: /s/ Rohan Sen
Date: 30 April 2026

[Page 4]
Receiving Party
Neel Verma Consulting LLP
Name: Neel Verma
Title: Managing Partner
Digital Signature: /s/ Neel Verma
Date: 30 April 2026"""

    # For demonstration, we'll just analyze the first chunk. In production, you'd loop through all chunks.

  extracted_data = extract_data_from_raw_text()
  print(json.dumps(extracted_data, indent=2))

