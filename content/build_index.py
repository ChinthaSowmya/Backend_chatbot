import requests
from bs4 import BeautifulSoup
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

SITEMAP = "https://scriptbees.com/sitemap.xml"

print("📡 Fetching sitemap...")
xml = requests.get(SITEMAP).text
soup = BeautifulSoup(xml, "xml")

links = [loc.text for loc in soup.find_all("loc")]
print(f"Found {len(links)} pages")

pages = []
meta = []
texts = []

for idx, url in enumerate(links):
    try:
        print("Scraping:", url)
        html = requests.get(url, timeout=10).text
        bs = BeautifulSoup(html, "html.parser")
        text = bs.get_text(separator=" ", strip=True)

        pages.append({
            "id": idx,
            "text": text
        })

        title = bs.title.string if bs.title else url

        meta.append({
            "id": idx,
            "url": url,
            "title": title
        })

        texts.append(text)

    except Exception as e:
        print("Error:", e)

print("Embedding pages...")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("Saving...")

with open("content/pages.json", "w", encoding="utf-8") as f:
    json.dump(pages, f, indent=2)

with open("content/pages_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

faiss.write_index(index, "content/pages.faiss")

print("DONE! Your ScriptBees RAG is ready.")
