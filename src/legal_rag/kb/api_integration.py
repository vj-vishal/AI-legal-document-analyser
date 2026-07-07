import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

class IndianKanoonClient:
    BASE_URL = "https://api.indiankanoon.org"

    def __init__(self):
        self.token = os.getenv("INDIANKANOON_API_TOKEN")
        self.headers = {"Authorization": f"Token {self.token}",
                        "Accept": "application/json"}

    # 1. Search Query
    def search(self, query: str, page: int = 0) -> dict:
        response = requests.post(
            f"{self.BASE_URL}/search/",
            headers=self.headers,
            data={"formInput": query, "pagenum": page}  # pagenum starts at 0
        )
        return response.json()

    # 2. Full Document
    def get_document(self, doc_id: int) -> dict:
        response = requests.post(
            f"{self.BASE_URL}/doc/{doc_id}/",
            headers=self.headers
        )
        return response.json()

    # 3. Court Copy (Original)
    def get_court_copy(self, doc_id: int) -> dict:
        response = requests.post(
            f"{self.BASE_URL}/origdoc/{doc_id}/",
            headers=self.headers
        )
        return response.json()

    # 4. Document Fragment
    def get_fragment(self, doc_id: int, query: str) -> dict:
        response = requests.post(
            f"{self.BASE_URL}/docfragment/{doc_id}/",
            headers=self.headers,
            data={"formInput": query}
        )
        return response.json()

    # 5. Document Metainfo
    def get_meta(self, doc_id: int) -> dict:
        response = requests.post(
            f"{self.BASE_URL}/docmeta/{doc_id}/",
            headers=self.headers
        )
        return response.json()
    
def clean_fragment(fragment_response: dict) -> dict:
    raw_html = " ".join(fragment_response.get("headline", []))
    
    # Strip all HTML tags
    soup = BeautifulSoup(raw_html, "html.parser")
    clean_text = soup.get_text(separator=" ", strip=True)
    
    return {
        "text": clean_text,
        "title": fragment_response.get("title", ""),
        "doc_id": fragment_response.get("tid", ""),
        "url": f"https://indiankanoon.org/doc/{fragment_response.get('tid', '')}/"
    }
    
# Initialize the client

# ik_client = IndianKanoonClient()

def retrieve_from_kanoon(query: str, client: IndianKanoonClient) -> list[dict]:
    results = client.search(query, page=0)
    docs = results.get("docs", [])

    chunks = []
    for doc in docs[:3]:
        if doc.get("fragment") is not True:
            continue

        doc_id = doc["tid"]

        try:
            fragment = client.get_fragment(doc_id, query)
            clean_data = clean_fragment(fragment)
        except Exception:
            continue

        chunks.append({
            "page_content": clean_data.get("text", ""),
            "metadata": {
                "source": clean_data.get("title", ""),       # from search result directly
                "court": doc.get("docsource", ""),     # from search result directly
                "date": doc.get("publishdate", ""),    # from search result directly
                "doc_id": doc_id,
                "url": f"https://indiankanoon.org/doc/{doc_id}/"
                }
        })

    return chunks


if __name__ == "__main__":

    client = IndianKanoonClient()

    doc = retrieve_from_kanoon("""briefly explain section 302 and its punishment?""", client)
    print(doc)