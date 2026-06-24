from pathlib import Path
from uuid import uuid4
from typing import List, Dict, Any, Tuple, Optional
import pickle
from groq import Groq
import ijson
import bm25s as _bm25s
from dotenv import load_dotenv
from pydantic import ConfigDict
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_classic.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
import torch
import src.legal_rag.config as config
from langchain_classic.schema import SystemMessage, HumanMessage
import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum

load_dotenv()


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class OverallDecision(str, Enum):
    PASS = "PASS"
    RETRY = "RETRY"
    FALLBACK = "FALLBACK"


# ──────────────────────────────────────────────
# Input schema
# ──────────────────────────────────────────────

class ChunkPayload(BaseModel):
    """Represents a single chunk sent to the judge LLM."""
    chunk_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id")
    @classmethod
    def chunk_id_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunk_id must not be empty or whitespace")
        return v

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunk text must not be empty or whitespace")
        return v


class JudgeInput(BaseModel):
    """Full payload sent to the judge."""
    query: str
    chunks: List[ChunkPayload]

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v

    @field_validator("chunks")
    @classmethod
    def chunks_must_not_be_empty(cls, v: List[ChunkPayload]) -> List[ChunkPayload]:
        if not v:
            raise ValueError("at least one chunk is required for judging")
        return v
    

# ──────────────────────────────────────────────
# Output schema (judge response)
# ──────────────────────────────────────────────

class ChunkJudgeResult(BaseModel):
    """Per-chunk evaluation result returned by the judge LLM."""
    chunk_id: str
    relevant: bool
    supports_answer: bool
    score: float = Field(..., ge=0.0, le=1.0)
    chunk_type_match: bool
    reason: str
    missing_aspects: List[str] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def score_must_be_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"score must be between 0.0 and 1.0, got {v}")
        return round(v, 4)

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reason must not be empty")
        return v


class JudgeOutput(BaseModel):
    """Full structured output from the judge LLM."""
    query: str
    overall_decision: OverallDecision
    confidence: float = Field(..., ge=0.0, le=1.0)
    retry_reason: Optional[str] = Field(default=None)
    chunk_results: List[ChunkJudgeResult]

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return round(v, 4)

    @field_validator("chunk_results")
    @classmethod
    def chunk_results_must_not_be_empty(cls, v: List[ChunkJudgeResult]) -> List[ChunkJudgeResult]:
        if not v:
            raise ValueError("chunk_results must contain at least one entry")
        return v

    @model_validator(mode="after")
    def retry_reason_required_on_non_pass(self) -> "JudgeOutput":
        if self.overall_decision != OverallDecision.PASS:
            if not self.retry_reason or not self.retry_reason.strip():
                raise ValueError(
                    f"retry_reason is required when overall_decision is '{self.overall_decision}'"
                )
        return self
    
# ──────────────────────────────────────────────
# Decision manager
# ──────────────────────────────────────────────

class DecisionManagerConfig(BaseModel):
    pass_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    retry_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    min_relevant_chunks_for_pass: int = Field(default=1, ge=1)
    require_supporting_chunk: bool = True
    require_chunk_type_match: bool = True

    @model_validator(mode="after")
    def thresholds_must_be_ordered(self) -> "DecisionManagerConfig":
        if self.retry_threshold >= self.pass_threshold:
            raise ValueError(
                f"retry_threshold ({self.retry_threshold}) must be "
                f"strictly less than pass_threshold ({self.pass_threshold})"
            )
        return self
    
# ──────────────────────────────────────────────
# Decision output
# ──────────────────────────────────────────────

class EvaluationResult(BaseModel):
    """Output of RetrievalDecisionManager.evaluate()."""
    decision: OverallDecision
    final_chunks: List[ChunkJudgeResult] = Field(default_factory=list)
    retry_chunks: List[ChunkJudgeResult] = Field(default_factory=list)
    fallback_message: Optional[str] = None

    @model_validator(mode="after")
    def fallback_message_required_on_fallback(self) -> "EvaluationResult":
        if self.decision == OverallDecision.FALLBACK and not self.fallback_message:
            raise ValueError("fallback_message must be set when decision is FALLBACK")
        return self
    


JUDGE_SYSTEM_PROMPT = """You are a strict retrieval judge for a legal RAG system.
Your task is NOT to answer the user.
Your task is ONLY to judge whether each retrieved chunk is useful for answering the query.

Evaluate each chunk on:
1. Relevance to the exact user query.
2. Sufficiency: whether the chunk contains usable substance, not just vague topical overlap.
3. Legal/document fit: correct provision, clause type, template type, section, entity, or context when metadata suggests it.
4. Mismatch detection: wrong document type, wrong legal context, wrong template, wrong jurisdiction, or wrong section.

Scoring rubric:
- 0.90 to 1.00: directly answers or strongly supports the answer.
- 0.75 to 0.89: clearly relevant and useful, though not complete alone.
- 0.50 to 0.74: somewhat relevant but incomplete, indirect, or partially mismatched.
- 0.00 to 0.49: mostly irrelevant, misleading, or wrong-context.

Return ONLY valid JSON matching the schema provided by the user. No markdown. No commentary."""

class GroqRetrievalJudge:
    def __init__(self):
        # self.config = config or RetrievalJudgeConfig()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=api_key)

    def _build_payload(self, query: str, chunks: List[Document]) -> JudgeInput:
        """Validate and build the judge input from LangChain Documents."""
        raw_chunks = [
            {
                "chunk_id": c.metadata["chunk_id"],
                "text": c.page_content,
                "metadata": c.metadata
            }
            for c in chunks
        ]
        # Pydantic validates structure and field constraints here
        return JudgeInput(query=query, chunks=raw_chunks)

    def judge(self, query: str, chunks: List[Document]) -> JudgeOutput :

        # Validate inputs
        judge_input = self._build_payload(query, chunks)

        schema = {
            "query": query,
            "overall_decision": "PASS | RETRY | FALLBACK",
            "confidence": 0.0,
            "retry_reason": "short reason",
            "chunk_results": [
                {
                    "chunk_id": "string",
                    "relevant": True,
                    "supports_answer": True,
                    "score": 0.0,
                    "chunk_type_match": True,
                    "reason": "short explanation",
                    "missing_aspects": ["list of missing aspects if any"],
                }
            ],
        }

        user_prompt = (
            "User query:\n"
            f"{query}\n\n"
            "Retrieved chunks:\n"
            f"{json.dumps([c.model_dump() for c in judge_input.chunks], ensure_ascii=False)}\n\n"
            "Return JSON with this schema exactly:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )

        response = self.client.chat.completions.create(
            model=config.LLM_AS_JUDGE,
            temperature=config.JUDGE_TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content
        data = json.loads(content)
        return data
    
class RetrievalDecisionManager:
    def __init__(self, setting: Optional[DecisionManagerConfig] = None):
        self.setting = setting or DecisionManagerConfig()
        
    def _valid_chunks(self, judge_result: JudgeOutput) -> List[ChunkJudgeResult]:
        chunks = [c for c in judge_result["chunk_results"] if c["relevant"] is True]

        if self.setting.require_chunk_type_match:
            chunks = [c for c in chunks if c["chunk_type_match"] is True]

        return chunks

    def decide(self, judge_result: JudgeOutput) -> OverallDecision:
        valid_chunks = self._valid_chunks(judge_result)
        supporting_chunks = [c for c in valid_chunks if c["supports_answer"] is True]
        top_score = max((c["score"] for c in valid_chunks), default=0.0)
        confidence = judge_result["confidence"]

        if (
            confidence >= self.setting.pass_threshold
            and len(valid_chunks) >= self.setting.min_relevant_chunks_for_pass
            and (
                not self.setting.require_supporting_chunk
                or len(supporting_chunks) >= 1
            )
            and top_score >= self.setting.pass_threshold
        ):
            return OverallDecision.PASS

        if confidence >= self.setting.retry_threshold or top_score >= self.setting.retry_threshold:
            return OverallDecision.RETRY

        return OverallDecision.FALLBACK

    def filter_final_chunks(self, judge_result: JudgeOutput) -> List[ChunkJudgeResult]:
        valid_chunks = self._valid_chunks(judge_result)
        final_chunks = [
            c for c in valid_chunks
            if c["supports_answer"] is True and c["score"] >= self.setting.pass_threshold
        ]
        final_chunks.sort(key=lambda x: x["score"], reverse=True)
        return final_chunks

    def filter_retry_chunks(self, judge_result: JudgeOutput) -> List[ChunkJudgeResult]:
        valid_chunks = self._valid_chunks(judge_result)
        retry_chunks = [
            c for c in valid_chunks
            if self.setting.retry_threshold <= c["score"] < self.setting.pass_threshold
        ]
        retry_chunks.sort(key=lambda x: x["score"], reverse=True)
        return retry_chunks

    def evaluate(self, judge_result: JudgeOutput) -> EvaluationResult:
        decision = self.decide(judge_result)
        return EvaluationResult(
            decision=decision,
            final_chunks=self.filter_final_chunks(judge_result),
            retry_chunks=self.filter_retry_chunks(judge_result),
            fallback_message=(
                "The retrieved legal documents do not contain sufficient information "
                "to fulfill this request."
                if decision == OverallDecision.FALLBACK
                else None
            ),
        )
    
    def collect_final_chunks(
        self,
        original_chunks: List[Document],
        eval_result: EvaluationResult,
    ) -> List[Document]:
        """Filter original LangChain Documents to only those that passed judging."""
        keep_ids = {c.chunk_id for c in eval_result.final_chunks}
        return [
            chunk for chunk in original_chunks
            if chunk.metadata["chunk_id"] in keep_ids
        ]

if __name__ == "__main__":
    judge = GroqRetrievalJudge()
    decision_manager = RetrievalDecisionManager()
    query= """Under Section 60, which properties of a judgment-debtor are exempt from attachment and sale during execution of a decree?"""
    chunks = [Document(id='9518e0c2-398b-4bf5-8d13-0bd55168444d', metadata={'chunk_id': 'Handbook on Legal System & Procedure_chunk_103', 'chapter_title': 'PROCEDURE IN CIVIL SUITS', 'pages': '100, 101', 'domain': 'Legal Concepts & Explanations', 'chapter_id': 'CHAPTER - VI', 'source': 'Handbook on Legal System & Procedure.pdf'}, page_content='Attachment\nProperty liable to attachment and sale in execution of decree- (Section 60 of CPC): (1) The following property is liable to attachment and salein execution of a decree, mainly lands, houses or other buildings, goods, money, bank notes, cheques, bills ofexchange, hundis, promissory notes, Govt securities,bonds or other securities for money, debts, shares in a corporation and, same as hereinafter mentioned, all . other saleable property, movable or immovable, belonging to the judgement debtor, or over which or the profits of which, he has a disposing power which he may exercise for his own benefit, whether the same be held in the name of the judgement debtor or by another person in trust for him or on his behalf.\nProvided that the following property shall not be liable to such attachment or sale, namely: -\n- (a) the necessary wearing-apparel, cooking vessels, beds and bedding of the judgementdebtor, his wife and children, and such personal ornaments as, in accordance with religious usage, cannot be parted with by any woman;\n- (b) tools of artisans, and, where the judgement-debtor is an agriculturist, his implements of husbandry and such cattle and seed-grain as may, in the opinion of the Court, be necessary to enable him to earn his livelihood as such, and suchportion of agricultural produce or of any class of agricultural produce as may have been declared to be free from liability;\n- (c) houses and other buildings (with the materials and the sites thereof and the land immediately appurtenant thereto and necessary for their enjoyment) belonging to an agriculturist or a\n- (d) books of account ;\n- () a mere right to sue for damages ;\n- (1) any right of personal service ;\n- (g) stipends and gratuities allowed to pensioners ofthe Government or ofa local authority or of any other employer, or payable out of any service family pension fund notified in the Official Gazette by the Central Government orthe State Government in this behalf, and political pensions;\n- (h) the wages of labourers and domestic servants, whether payable in money or in kind ;\n- (① . salary to the extent of the first one thousand rupees and two thirds of the remainder in execution of any decree other than a decree for maintenance:\nProvided that where any part of such portion ofthe salary as is liable to attachment has been under attachment, whether continuously or intermittently, for a total period of twenty four months, such portion shall be exempt from attachment until the expiry of a further period of twelve months, and, where such attachment has been made in execution of one and the same decree, shall,after the attachment has continued for a total period of twenty four months, be finally exempt from attachment in execution of that decree;'), Document(id='f345ed73-4481-495d-94a3-c7f292b0be72', metadata={'pages': '101, 102', 'source': 'Handbook on Legal System & Procedure.pdf', 'chapter_id': 'CHAPTER - VI', 'chunk_id': 'Handbook on Legal System & Procedure_chunk_104', 'chapter_title': 'PROCEDURE IN CIVIL SUITS', 'domain': 'Legal Concepts & Explanations'}, page_content='- (ia) one third of the salary in execution of any decree for maintenance;\n- () the pay & allowances of persons to whom the Air Force Act, 1950 (45 of 1950), or the Army Act, 1950 (46 of 1950), or the Navy Act, 1957 (62 of 1957), applies ;\n- (Y) all compulsory deposits ant other sums in or derived from any fund to which the Provident Fund Act, (26) [1925] (19 of 1925), for the time being applies in so far as they are declared by the said Act not to be liable to attachment;\n- (ka) all deposits and other sums in or derived from any fund to which the Public Provident s os \'s q r (1 1 n declared by the said Act as not to be liable to attachment;\n- (kb) all moneys payable under a policy of insurance on the life of the judgement-debtor;\n- (kc) the interest ofa lease ofa residential building to which the provisions of law for the time being in force relating to control of rents and accommodation apply;\n- (① any allowance forming part ofthe emoluments of any servant ofthe Government or of any servant of a railway company or local authority which the appropriate Government may by notification in the official gazette declare to be exempt from attachment, and any subsistence grant for allowance made to any such servant while under suspension.\n- (m)\n- (m) a right to future maintenance;\n3 001.\n::\n1. (0):8 any allowance declared by (32)[any Indian law] to be exempt from liability to attachment or sale in execution of a decree; and\n- (d) where the judgement debtor is a person liable for the payment of land revenue, any movable property which, under any law for the time being applicable to him, is exempt from sale for the recovery of an arrear of such revenue.\nExplanation I1. - The moneys payable in relation to the matters mentioned in clauses (g), (h), (i), (ia), (), (I) & (o) are exempt from attachment or sale, whether before or after they are actually payable, and, in the case of salary, the attachable portion thereof is liable to attachment, whether before or after it is actually payable.\nExplanation II - In clause (M) and (ia)], "salary\'means the total monthly emoluments, excluding any allowance declared exempt from attachment under the provisions of clause (l), derived by a person from his employment whether on duty or on leave.]\n- (i as respect any person in the service of the Central Government , or any servant of a Railway Administration or of a cantonment authority or of the port-authority ofa major port, the Central Government ;\n- (i) as respects any other servant of the Government or a servant of any other local authority, the State Government.\nExplanation IV - For the purposes if this proviso,""wages\' includes bonus, and " labourer\' includes a skilled, unskiled or semi-skilledlabourer.\nExplanation V - For the purposes of this proviso, the expression ""agriculturist\' means a person who cultivates land personally and who depends for his livelihood mainly on the income form agricultural land, whether as owner, tenant, partner or agricultural labourer.\nExplanation VI - For the purposes ofExplanation V, an agriculturist shall be deemed to cultivate land personally, ifhe cultivates land -\n- (a) by his own labour, or\n- (b) by the labour of any member of hisfamily, or\n- (c) by servants or labourers on wages payable in cash or in kind (not being as a share of the produce), or both\n(1-A) Not withstanding anything contained in any other law for the time being in force, an agreement by which a person agrees to waive the benefit of any exemption under this section shall be void.')]
    result = judge.judge(query, chunks)
    # print(json.dumps(result, indent=2))
    # print(50*"==")
    final_result = decision_manager.evaluate(result)
    print(final_result)
    print(50*"==")
    if final_result.decision == "PASS":
        matching_chunks = decision_manager.collect_final_chunks(chunks, final_result)
        print(matching_chunks)
    elif final_result.decision == "FALLBACK":
        print(final_result.decision)
    else: 
        print("none")
    # print(json.dumps(final_result, indent=2))
    # print(json.dumps(result, indent=2))