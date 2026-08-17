import hashlib
import json
import logging
import re
from faker import Faker
from gliner2 import GLiNER2
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, EntityRecognizer, RecognizerResult
import torch
import gc
from dotenv import load_dotenv

load_dotenv()

# Suppress Hugging Face verbose logs and print only errors.
logging.getLogger("transformers").setLevel(logging.ERROR)

print("Booting up Perfected Hybrid Pipeline...")

# cutomizing the target labels for GLiNER2
TARGET_LABELS =  [
    "person", 
    "company", 
    "email", 
    "document date",
    "street address",
    "routing number",
    "tax id", 
    "digital signature"
]
TARGET_ENTITIES_UPPER = [label.upper() for label in TARGET_LABELS]

# ==========================================
# STEP 1: The GLiNER2 Presidio Bridge
# ==========================================
class GLiNER2PresidioRecognizer(EntityRecognizer):
    def __init__(self, model_name="fastino/gliner2-privacy-filter-PII-multi", labels=TARGET_LABELS):
        super().__init__(supported_entities=TARGET_ENTITIES_UPPER, supported_language="en")
        # self.model = GLiNER2.from_pretrained(model_name)
        self.labels = labels

        
        # print(f"Loading GLiNER2 on {self.device}...")
        self.model = GLiNER2.from_pretrained(model_name,
                                             map_location="cuda" if torch.cuda.is_available() else "cpu",
                                            quantize=True,)


    def analyze(self, text: str, entities: list[str], nlp_artifacts=None) -> list[RecognizerResult]:
        with torch.inference_mode():
            extraction = self.model.extract_entities(
                text, self.labels, threshold=0.6, include_confidence=True, include_spans=True
            )
        
        results = []
        for label, items in extraction.get("entities", {}).items():
            for item in items:
                results.append(
                    RecognizerResult(
                        entity_type=label.upper(),
                        start=item["start"],
                        end=item["end"],
                        score=item["confidence"]
                    )
                )
        return results

# Initialize the Presidio Analyzer ONLY (We drop the AnonymizerEngine)
registry = RecognizerRegistry()
registry.load_predefined_recognizers()
registry.add_recognizer(GLiNER2PresidioRecognizer())

try:
    registry.remove_recognizer("SpacyRecognizer")
except ValueError:
    pass

presidio_analyzer = AnalyzerEngine(registry=registry)


# ==========================================
# STEP 2: The Sequential Masking Pipeline
# ==========================================
def analyze_and_mask_bulletproof(text: str) -> tuple[str, dict]:
    session_mapping = {}
    label_counters = {} # Dictionary to track the sequential index for each label
    
    # 1. PRESIDIO ANALYSIS
    analyzer_results = presidio_analyzer.analyze(
        text=text, 
        language="en",
        entities=TARGET_ENTITIES_UPPER,
        return_decision_process=False
    )
    
    # 2. EXTRACT DE-DUPLICATED STRINGS
    extracted_entities = []
    for res in analyzer_results:
        real_text = text[res.start:res.end]
        extracted_entities.append({
            "label": res.entity_type,
            "text": real_text
        })
        
    # 3. GENERATE SEQUENTIAL PLACEHOLDERS (e.g., [PERSON_1])
    for entity in extracted_entities:
        real_val = entity["text"]
        label = entity["label"]
        
        # Only assign a new operator if we haven't processed this exact string yet
        if real_val not in session_mapping.values():
            
            # Initialize counter for this label if it doesn't exist
            if label not in label_counters:
                label_counters[label] = 1
                
            # Create the operator string: e.g., [COMPANY_1]
            placeholder = f"[{label}_{label_counters[label]}]"
            
            # Store in mapping and increment the counter for the next one
            session_mapping[placeholder] = real_val
            label_counters[label] += 1

    # 4. GLOBAL PROPAGATION MASKING
    masked_text = text
    
    # Flip dictionary to iterate {real_name: placeholder}
    reverse_mapping = {v: k for k, v in session_mapping.items()}
    
    # CRITICAL: Sort by length descending to prevent partial word replacements
    sorted_real_names = sorted(reverse_mapping.keys(), key=len, reverse=True)
    
    for real_name in sorted_real_names:
        placeholder = reverse_mapping[real_name]
        
        # Global string replace using regex. 
        pattern = re.compile(re.escape(real_name))
        masked_text = pattern.sub(placeholder, masked_text)
        
    return masked_text, session_mapping


# ==========================================
# STEP 3: The Read Path (Unmasking)
# ==========================================
def restore_text(llm_output: str, mapping_dict: dict) -> str:
    restored_text = llm_output
    for placeholder, real_name in mapping_dict.items():
        if placeholder in restored_text:
            restored_text = restored_text.replace(placeholder, real_name)
    return restored_text

# ==========================================
# Run Test
# ==========================================
if __name__ == "__main__":
    raw_text= """[Page 1]
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
Date: 30 April 2026
    """

    llm_output = """This is a fictional **Non-Disclosure Agreement** sample used for testing and prototyping, not a real legal contract. 
    It says two parties, a Kolkata-based disclosing party and a Bengaluru-based receiving party, want to explore a potential collaboration 
    around AI document processing, confidential product design, retrieval systems, and enterprise software integration.

## Main points

The agreement defines confidential information broadly as non-public information shared in any form, including business plans, 
financial projections, pricing, customer strategies, source code, model pipelines, prompts, embeddings, datasets, APIs, architecture 
diagrams, technical documentation, product specs, unreleased features, prototypes, experiments, and internal reports.

It also includes signature placeholders for both sides:
- Disclosing Party: `[COMPANY_1]`, signed by `[PERSON_1]`, Director, Product Strategy.
- Receiving Party: `[COMPANY_2]`, signed by `[PERSON_2]`, Managing Partner.
- Signature date: `[DOCUMENT DATE_1]`.

## Short test-friendly summary

A fictional NDA between two companies in Kolkata and Bengaluru for discussing an AI/document-processing collaboration, where confidential 
information includes business, financial, and technical materials such as code, models, datasets, APIs, and product plans.

## Unmasked sample

Non-Disclosure Agreement between a disclosing party in Kolkata and a receiving party in Bengaluru for a possible AI and enterprise 
software collaboration. It covers broad confidential information such as business plans, source code, model pipelines, embeddings, 
datasets, APIs, architecture diagrams, product specifications, prototypes, and internal reports. It ends with placeholder signatures 
for both companies and their representatives.
"""
    
    # tenant_id = "tenant_xyz"
    
    # Paste your 4-page document string here in your script
    # original_document = raw_text
    
    safe_text, memory_vault = analyze_and_mask_bulletproof(raw_text)
    
    print(f"\n[STEP 1] Perfected Session Mappings:")
    print(json.dumps(memory_vault, indent=2))
    
    print(f"\n[STEP 2] Safe Text (Ready for ChromaDB):")
    # You will now see EVERY date and company globally masked, and NO US Driver's licenses
    print(safe_text)

    print(f"\n[STEP 3] Restored Text (After LLM Output):")
    restored_text = restore_text(llm_output, memory_vault)
    print(restored_text)

