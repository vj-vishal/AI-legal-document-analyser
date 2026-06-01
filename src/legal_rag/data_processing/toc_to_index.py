import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

system_template = """You are an expert document parsing algorithm. Your sole function is to extract the Table of Contents (TOC), Index, or Synopsis from raw OCR text and convert it into a precise, nested JSON array. 

You must obey these strict rules:
1. NO CONVERSATION: Output ONLY valid JSON. Do not include greetings, explanations, or markdown blocks outside the JSON.
2. STRIP NOISE: Remove all page numbers and dot leaders (e.g., ".......").
3. PRESERVE ANOMALIES: Keep exact spelling, spacing errors, and weird characters (e.g., "IVX", "Purchaser ' s") exactly as they appear in the text. Do not correct typos.
4. FRONT/BACK MATTER: Treat sections like "Preface", "Index", "Appendices", or "Bibliography" as top-level chapter/part keys.
5. FALLBACK: If the provided text does not contain a discernible Table of Contents, Index, or Synopsis, output exactly: {"status": "TOC missing"}
"""

user_template = """
Analyze the following raw OCR text and extract the Table of Contents into this exact JSON structure:

[
  {{
    "Exact Chapter/Part Name": {{
      "chapter heading": ["Exact Heading 1", "Exact Heading 2"],
      "section": ["Exact Section 1", "Exact Section 2"],
      "sub section": ["Exact Sub-section 1","Exact Sub-section 2" ]
    }}
  }}
]

Hierarchy Mapping Rules:
- Top-level keys (Chapter/Part/Preface) MUST be the dictionary keys.
- Do NOT repeat the Top-level key inside the "chapter heading" array.
- If a level does not exist (e.g., no chapter headings or no sub-sections), return an empty array `[]`.
- If a chapter heading exists as a string instead of a list, format it as an array with one string inside.
- Maintain the strict chronological sequence of the document.

Raw OCR Text to parse:
<RAW_TEXT_HERE>
{RAW_TEXT_HERE}
"""

def _clean_json(content: str) -> str:
    return content.replace("```json", "").replace("```", "").strip()

class LLMGenerator:
    def __init__(self):
        # FIX: Ensure GROQ_API_KEY is in your .env file
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",  # FIX: Valid Groq model
            temperature=0.0,               # FIX: Set to 0 for maximum determinism
            max_tokens=3000,
            # FIX: Explicitly enforce JSON output mode
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    # FIX: Corrected type hints
    def generate_answer(self, raw_text: str) -> list:
        messages = [
            SystemMessage(content=system_template),
            HumanMessage(content=user_template.format(
                RAW_TEXT_HERE=raw_text
            ))
        ]
        
        response = self.llm.invoke(messages)
        content = _clean_json(response.content)

        try:
            # Groq's JSON mode returns a dictionary, so we wrap it in a list to match your blueprint structure if needed
            index_json = json.loads(content)
            
            # If the LLM returned a dict (like {"status": "TOC missing"} or a root dict instead of a list)
            if isinstance(index_json, dict):
                if index_json.get("status") == "TOC missing":
                    return index_json # Return the fallback dict directly
                index_json = [index_json] # Wrap in list to maintain schema

        except Exception as e:
            print(f"Ground truth JSON parse error: {e}")
            print(f"Raw Output was:\n{content}")
            index_json = []

        return index_json
    
if __name__ == "__main__":
    raw_text = """
3. The education profile of the D.A.D personnel is indeed impressive. However, the number

of employees having a formal law qualification is very small. The level of legal knowledge in

the Department certainly needs to be improved. Recognizing this, the Department has made

conscious efforts to familiarize its manpower with legal procedures and processes through

training programmes. The numbers trained so far is, however, not commensurate with the

requirement. As a result, most of the DAD offices find themselves inadequately equipped to

handle legal cases.

4. The publication of this handbook is an initiative taken to upgrade legal knowledge in the

Department. The handbook looks at the legal system and procedures from a layman's point of

view. It incorporates most of the topics that our employees need to be familiar with for handling

legal cases. Various Government orders & instructions for handling and monitoring legal cases

have also been included.

5. It must be added here that this is entirely an in-house effort of the Administration

Section of the Headquarters office. The information set out in this publication has been

borrowed from various sources. In this regard, the study material on the subject prepared by

the Institute of Judicial Training & Research, Lucknow, Institute of Secretarial Training &

Management, New Delhi and that available on various Internet sites has been consulted.

Suggestions for improving the content and presentation of this publication are welcome



NDEX

Chapter Subject

Page No

Preface

I

Introduction

1

7

II

Judicial system in India

2.1 Heirarchy

8

2.2 District and Subordinate Courts

6

2.3 Central Administrative Tribunals

9-25

2.4 High Court

26-31

2.5 Supreme Court of India

32-37

III

Remedies available under the Constitution (3.1 to5)

38-40

IV Ministry of Law & Justice (4.1 to 4.16)

41-44

Annexures to Chapter-IV

45-85

Annexure-I 45-47

Supervision of Central Government Litigation work at

different High Courts

Annexure -II 48-49

Revised Scheme containing terms & engagement of CGSC

& Central Government Pleaders in Delhi High Court

w.e.f 1/10/99

Annexure -III & IV 50-63

Revised Scheme containing terms & engagement of Senior

Counsel in Delhi High Court w.e.f 1/10/99

Annexure-V

Revised Scheme containing terms & engagement of Senior

64-66

CGSC/Addl CGSC in various High Courts (except High

Courts of Delhi, Mumbai, Calcutta, Chennai and Karnataka)

w.e.f 1/10/99

G81 Annexure -VI & VII 67-79

Revised Scheme containing terms & engagement of Senior

Counsel in various High Courts (except High Courts ofDelhi,

Mumbai, Calcutta, Chennai and Karnataka) w.e.f 1/10/99
"""
    generator = LLMGenerator()
    answer = generator.generate_answer(raw_text)
    print(json.dumps(answer, indent=2))