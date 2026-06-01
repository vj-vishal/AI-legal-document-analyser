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

# Suppress Hugging Face verbose logs
logging.getLogger("transformers").setLevel(logging.ERROR)

print("Booting up Perfected Hybrid Pipeline...")
faker_tool = Faker()


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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading GLiNER2 on {self.device}...")
        self.model = GLiNER2.from_pretrained(model_name)

        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        if self.device == "cuda":
            try:
                self.model = self.model.half()
            except Exception:
                pass

        if hasattr(self.model, "eval"):
            self.model.eval()

        # gc.collect()

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
# STEP 2: Deterministic Faker Logic
# ==========================================
def get_fake_name(user_id: str, real_text: str, label: str) -> str:
    seed_value = int(hashlib.md5(f"{user_id}:{real_text}".encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
    Faker.seed(seed_value)
    
    label = label.lower()
    if "company" in label: 
        return faker_tool.company()
    elif "person" in label: 
        return faker_tool.name()
    elif "email" in label: 
        return faker_tool.company_email()
    elif "date" in label: 
        return faker_tool.date()
    elif "address" in label:
        return faker_tool.street_address()
    elif "routing number" in label:
        return faker_tool.iban()
    elif "tax id" in label:
        return faker_tool.ssn()
    else:
        return f"[{label.upper()}_{faker_tool.word()}]"


# ==========================================
# STEP 3: The Bulletproof Masking Pipeline
# ==========================================
def analyze_and_mask_bulletproof(text: str, user_id: str) -> tuple[str, dict]:
    session_mapping = {}
    
    # 1. PRESIDIO ANALYSIS
    # We pass `entities=TARGET_ENTITIES_UPPER` to strictly block default recognizers 
    # like US_DRIVER_LICENSE from interfering.
    # Presidio automatically drops overlapping index collisions natively here!
    analyzer_results = presidio_analyzer.analyze(
        text=text, 
        language="en",
        entities=TARGET_ENTITIES_UPPER,
        return_decision_process=False
    )
    
    # 2. EXTRACT DE-DUPLICATED STRINGS
    extracted_entities = []
    for res in analyzer_results:
        # We slice the original text using Presidio's collision-free indices
        real_text = text[res.start:res.end]
        extracted_entities.append({
            "label": res.entity_type,
            "text": real_text
        })
        
    # 3. GENERATE FAKE NAMES
    for entity in extracted_entities:
        real_val = entity["text"]
        
        # Only generate a fake name if we haven't processed this exact string yet
        if real_val not in session_mapping.values():
            fake_val = get_fake_name(user_id, real_val, entity["label"])
            session_mapping[fake_val] = real_val

    # 4. GLOBAL PROPAGATION MASKING
    masked_text = text
    
    # Flip dictionary to iterate {real_name: fake_name}
    reverse_mapping = {v: k for k, v in session_mapping.items()}
    
    # CRITICAL: Sort by length descending so we replace "Asterion Analytics Private Limited"
    # before we accidentally replace just the word "Asterion" if it exists.
    sorted_real_names = sorted(reverse_mapping.keys(), key=len, reverse=True)
    
    for real_name in sorted_real_names:
        fake_name = reverse_mapping[real_name]
        
        # Global string replace using regex. 
        # This guarantees that if Presidio found a date on Page 2, 
        # it gets masked on Pages 1, 3, and 4 automatically.
        pattern = re.compile(re.escape(real_name))
        masked_text = pattern.sub(fake_name, masked_text)
        
    return masked_text, session_mapping


# ==========================================
# STEP 4: The Read Path (Unmasking)
# ==========================================
def restore_text(llm_output: str, mapping_dict: dict) -> str:
    restored_text = llm_output
    for fake_name, real_name in mapping_dict.items():
        if fake_name in restored_text:
            restored_text = restored_text.replace(fake_name, real_name)
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
    
    tenant_id = "tenant_xyz"
    
    # Paste your 4-page document string here in your script
    original_document = raw_text
    
    safe_text, memory_vault = analyze_and_mask_bulletproof(original_document, tenant_id)
    
    print(f"\n[STEP 1] Perfected Session Mappings:")
    print(json.dumps(memory_vault, indent=2))
    
    print(f"\n[STEP 2] Safe Text (Ready for ChromaDB):")
    # You will now see EVERY date and company globally masked, and NO US Driver's licenses
    print(safe_text)

