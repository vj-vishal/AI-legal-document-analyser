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

load_dotenv()

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

    def judge(self, query: str, chunks: List[Document]) -> Dict[str, Any]:

            payload_chunks = []
            for c in chunks:
                payload_chunks.append(
                    {
                        "chunk_id": c.metadata.get("chunk_id", "unknown"),
                        "text": c.page_content,
                        "metadata": c.metadata
                    }
                )

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
                f"{json.dumps(payload_chunks, ensure_ascii=False)}\n\n"
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
    def __init__(
        self,
        pass_threshold: float = 0.78,
        retry_threshold: float = 0.50,
        min_relevant_chunks_for_pass: int = 1,
        require_supporting_chunk: bool = True,
        require_chunk_type_match: bool = True,
    ):
        self.pass_threshold = pass_threshold
        self.retry_threshold = retry_threshold
        self.min_relevant_chunks_for_pass = min_relevant_chunks_for_pass
        self.require_supporting_chunk = require_supporting_chunk
        self.require_chunk_type_match = require_chunk_type_match

    def _valid_chunks(self, judge_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = [c for c in judge_result.get("chunk_results", []) if c.get("relevant") is True]

        if self.require_chunk_type_match:
            chunks = [c for c in chunks if c.get("chunk_type_match") is True]

        return chunks

    def decide(self, judge_result: Dict[str, Any]) -> str:
        valid_chunks = self._valid_chunks(judge_result)
        supporting_chunks = [c for c in valid_chunks if c.get("supports_answer") is True]
        top_score = max((c.get("score", 0.0) for c in valid_chunks), default=0.0)
        confidence = judge_result.get("confidence", 0.0)

        if (
            confidence >= self.pass_threshold
            and len(valid_chunks) >= self.min_relevant_chunks_for_pass
            and (
                not self.require_supporting_chunk
                or len(supporting_chunks) >= 1
            )
            and top_score >= self.pass_threshold
        ):
            return "PASS"

        if confidence >= self.retry_threshold or top_score >= self.retry_threshold:
            return "RETRY"

        return "FALLBACK"

    def filter_final_chunks(self, judge_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        valid_chunks = self._valid_chunks(judge_result)
        final_chunks = [
            c for c in valid_chunks
            if c.get("supports_answer") is True and c.get("score", 0.0) >= self.pass_threshold
        ]
        final_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return final_chunks

    def filter_retry_chunks(self, judge_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        valid_chunks = self._valid_chunks(judge_result)
        retry_chunks = [
            c for c in valid_chunks
            if self.retry_threshold <= c.get("score", 0.0) < self.pass_threshold
        ]
        retry_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return retry_chunks

    def evaluate(self, judge_result: Dict[str, Any]) -> Dict[str, Any]:
        decision = self.decide(judge_result)
        result = {
            "decision": decision,
            "final_chunks": self.filter_final_chunks(judge_result),
            "retry_chunks": self.filter_retry_chunks(judge_result),
        }

        if decision == "FALLBACK":
            result["fallback_message"] = (
                "The retrieved legal documents do not contain sufficient information to fulfill this request."
            )

        return result
    
    def collect_final_chunks(self, original_chunks: List[Dict[str, Any]], judge_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        keep_ids = {c["chunk_id"] for c in judge_output.get("final_chunks", [])}
        return [chunk for chunk in original_chunks if chunk.metadata.get("chunk_id") in keep_ids]

if __name__ == "__main__":
    judge = GroqRetrievalJudge()
    decision_manager = RetrievalDecisionManager()
    query= """What is the total revised honorarium payable to a Panel Advocate for a matter that is finally disposed of on a regular basis (after granting leave to appeal), and on what date did this new rate come into effect?"""
    chunks = [Document(metadata={'source': 'Practice and procedure-2022.pdf', 'pages': '8, 9', 'chapter_id': 'CHAPTER II', 'chapter_title': 'COMPOSITION, POWERS AND FUNCTIONS OF THE COMMITTEE', 'clause': '7. Funds, audit and accounts of the Committee', 'domain': 'Standard Legal Procedures', 'chunk_id': 'Practice and procedure-2022_chunk_11'}, page_content="Regulation 10 deals with Funds, Audit and Accounts of the Committee as under:\n- (1) The Committee shall maintain a Fund to be called Supreme Court Legal Services Committee Fund to which shall be credited, - (a) such amounts as may be allocated and granted to it by the Central Authority; (b) all such amounts received by the Committee by way of donations; (c) all such amounts by way of costs, charges and expenses recovered from the persons to whom legal service is provided or the opposite party.\n- (2) All the amounts credited to the said Fund, shall be deposited in a nationalized bank.\nExplanation: -- In the sub-regulation 'nationalized bank' means a corresponding new  bank  as  defined  in  the  Banking  Companies  (Acquisition  and  Transfer  of Undertakings) Act, 1970, and the Banking Companies (Acquisition and Transfer of Undertakings) Act, 1980.\n- (3) For the purpose of meeting incidental minor charges, such as court-fee, stamps and expenditure necessary for obtaining copies of documents etc., a permanent advance of Rupees twenty-five thousand shall be placed at the disposal of the Secretary of the Committee.\n- (4) All expenditure on legal service, accommodation and staff of the Committee as also expenditure necessary for carrying out the various functions of the Committee shall be met out of the Funds of the Committee and in accordance with the prior approval of the Chairman.\n- (5) The funds of the Committee may be utilized for meeting the expense incurred on or incidental to travels undertaken by the Chairman or other members of the Committee or the Secretary in connection with legal service activities. The travelling allowance and the dearness allowance payable to the Chairman, the ex-\nofficio members and the Secretary shall be such as to which they are entitled by virtue of their respective offices held.\n- (6) The Secretary of the Committee and one member of the Committee designated by the Chairman for this purpose shall jointly operate the bank accounts of the Committee in accordance with the directions of the Chairman.\n- (7) The Committee shall cause to be kept and maintained true and correct accounts of all receipts and disbursements and furnish quarterly returns to the Central Authority.\n- (8) The accounts of the Committee shall be audited annually by a qualified Auditor and submitted to the Central Authority."), Document(metadata={'source': 'Practice and procedure-2022.pdf', 'pages': '14, 15', 'chapter_id': 'CHAPTER V', 'chapter_title': 'DUTIES AND RESPONSIBILITIES OF OFFICERS AND STAFF AND FUNCTIONS OF THE COMMITTEE', 'clause': '2. Secretary', 'domain': 'Standard Legal Procedures', 'chunk_id': 'Practice and procedure-2022_chunk_16'}, page_content='Regulation  8  of  the  Regulations  deals  with  the  functioning  and  powers  of  the Secretary of the Committee. He shall be the Principal Officer of the Committee appointed by the Chief Justice of India. He shall be the custodian of all assets, accounts, records and funds at the disposal of the Committee. He shall  maintain accounts  of  the  receipts  and  disbursements  of  the  funds  of  the  Committee.    Besides looking  after  the  day-to-day  functioning  of  the  Committee,  he  also  acts  as Coordinator of the Supreme Court Mediation Centre.  He shall also be required to perform the following duties:\n- (i) to sign and approve the bills of advocates, translators and other payments;\n- (ii) to operate the Bank Account of the Committee jointly with one member of the Committee appointed by the Chairman;\n- (iii) to sign and approve vouchers pertaining to receipts and payments;\n- (iv) to sign and approve Salary Bills, Cashbook, Ledger Books;\n- (v) to prepare and approve balance sheet, income/expenditure and receipt and payment statement and bank reconciliation statements;\n- (vi) to ensure correct deduction of TDS;\n- (vii) to deal with the Income Tax Authorities regarding Income Tax matters and Auditors;\n- (viii) to convene meetings of the Committee with the prior approval of the Chairman;\n- (ix) to prepare Notice/Agenda of the meeting;\n- (x) to record and prepare minutes of the meeting;\n- (xi) to ensure implementation of the decisions taken and resolutions passed in the meeting;\n- (xii) to maintain a Register of applications for legal services and to maintain status of such applications;\n- (xiii) to seek legal opinion in a matter from the Legal Service Counsel-cum-Consultant;\n- (xiv) to nominate/assign matters to the panel advocates for the purpose of filing or defending cases before the Supreme Court and to issue Assignment Letters in this regard;\n- (xv) to nominate senior Advocate for arguing any matter filed or defended by the Committee;\n- (xvi) to sign the Certificate of Exemption of payment of Court Fees in the matters to be filed before the Supreme Court through the Committee;\n- (xvii) to record the reasons in writing for rejecting legal services and lodge the incomplete, irregular and non-maintainable legal aid applications;\n- (xviii) to liaison with the Registry, advocates and other Authorities of Government of India, including Central Authority and other Legal Services Authorities in connection with the work of the Committee;\n- (xix) to prepare brief note on any subject of importance to be taken up with the Chairman or the Registry;\n- (xx) to prepare note on the appeal to the Chairman under Regulation 12(6) of the Regulations in case an appeal is received by the Committee from a person aggrieved by an order of rejection of legal services to him;\n- (xxi) to act as the Appellate Authority under the Right to Information Act, 2005;\n- (xxii) to perform such other duties as may be assigned by the Chairman.'), Document(metadata={'source': 'Handbook on Legal System & Procedure.pdf', 'chapter_id': 'CHAPTER - IV', 'chapter_title': 'MINISTRY OF LAW AND JUSTICE', 'section': 'Annexure - III: Revised Scheme containing terms & engagement of Senior Counsel in Delhi High Court w.e.f 1/10/99', 'domain': 'Legal Concepts & Explanations', 'chunk_id': 'Handbook on Legal System & Procedure_chunk_66'}, page_content='(v), Appeals from decrees from suits  and proceedings including Second Appeals, except L.P.A. from petition under Articles 226& 227  as mentioned in item (i) above and appeals from declaratory decrees or such decrees in either there is no valuation or valuation is notional or which are mainly on question of law and such appeals which have been specifically or separately provided herein = Civil or Criminal Revision Petitions. (v), For each case same fee as in item (ii) above or fee fixed by the Court, whichever is higher = Rs.1050/- per case. (vi), Appeals from decrees from suits  and proceedings including Second Appeals, except L.P.A. from petition under Articles 226& 227  as mentioned in item (i) above and appeals from declaratory decrees or such decrees in either there is no valuation or valuation is notional or which are mainly on question of law and such appeals which have been specifically or separately provided herein = Civil Miscellaneous applications or  petitions under the Indian Succession Act, Contempt of Court proceedings and other proceedings of an original nature not specifically provided otherwise.. (vi), For each case same fee as in item (ii) above or fee fixed by the Court, whichever is higher = Rs.750/- per case. (vii), Appeals from decrees from suits  and proceedings including Second Appeals, except L.P.A. from petition under Articles 226& 227  as mentioned in item (i) above and appeals from declaratory decrees or such decrees in either there is no valuation or valuation is notional or which are mainly on question of law and such appeals which have been specifically or separately provided herein = References to the High Court under Sales Tax Act and Banking Company Petitions.. (vii), For each case same fee as in item (ii) above or fee fixed by the Court, whichever is higher = Rs.1050/- per case or the amount fixed by the High Court whichever is higher.. , Appeals from decrees from suits  and proceedings including Second Appeals, except L.P.A. from petition under Articles 226& 227  as mentioned in item (i) above and appeals from declaratory decrees or such decrees in either there is no valuation or valuation is notional or which are mainly on question of law and such appeals which have been specifically or separately provided herein = (vi)  Company Petitions .. , For each case same fee as in item (ii) above or fee fixed by the Court, whichever is higher = To be regulated by the rule contained in Appendix HI of the Company (Court) Rules, 1959.. (ix), Appeals from decrees from suits  and proceedings including Second Appeals, except L.P.A. from petition under Articles 226& 227  as mentioned in item (i) above and appeals from declaratory decrees or such decrees in either there is no valuation or valuation is notional or which are mainly on question of law and such appeals which have been specifically or separately provided herein = All cases ofthe nature where no substantial legal work is involved and no substantial legal work is actually done till the disposal of the case and miscellaneous petitions or work not otherwise provided for like Forma Pauperis, Transfer Petitions, Settlement of list of Supreme Court case, execution proceedings.. (ix), For each case same fee as in item (ii) above or fee fixed by the Court, whichever is higher = Rs.300/- per petition.. , Appeals from decrees from suits  and proceedings including Second Appeals, except L.P.A. from petition under Articles 226& 227  as mentioned in item (i) above and appeals from declaratory decrees or such decrees in either there is no valuation or valuation is notional or which are mainly on question of law and such appeals which have been specifically or separately provided herein = (x)  (a) Cases under Arbitration Act. , For each case same fee as in item (ii) above or fee fixed by the Court, whichever is higher = In cases under Section 34 of The  1:Arbitration and Conciliation Act, 1996 registered as Suits, the fee payable per case shall be 1/4 of the fee according to the scale mentioned in VIlI (i) ifthe case is uncontested subject to a minimum ofRs. 1050/- and a maximum ofRs. 3000/-')]
    result = judge.judge(query, chunks)
    # print(json.dumps(result, indent=2))
    # print(50*"==")
    final_result = decision_manager.evaluate(result)
    print(final_result)
    print(50*"==")
    if final_result["decision"] == "PASS":
        matching_chunks = decision_manager.collect_final_chunks(chunks, final_result)
        print(matching_chunks)
    elif final_result["decision"] == "FALLBACK":
        print(final_result["decision"])
    else: 
        print("none")
    # print(json.dumps(final_result, indent=2))
    # print(json.dumps(result, indent=2))