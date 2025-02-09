import arxiv
import sqlite3
import torch
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import pytz
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def generate_embedding(text):
    return embedding_model.encode(text[:2048], dtype=torch.float16).tolist()  # Truncate to first 2048 chars

# Database setup
DB_FILE = "main_database.db"

# Step 1: Database Initialization
def setup_database():
    """Creates the database and updates schema. Run this only once."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS arxiv_papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            authors TEXT,
            summary TEXT,
            url TEXT UNIQUE,
            published TEXT,
            categories TEXT,
            embedding BLOB,
            cluster INTEGER
        )
    """)
    conn.commit()
    conn.close()

# Step 2: Fetch and Store New Papers
LAST_24_HOURS = datetime.now(pytz.UTC) - timedelta(days=5)

def paper_exists(url):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM arxiv_papers WHERE url = ?", (url,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def save_paper(paper):
    if paper_exists(paper["url"]):
        return False

    #embedding = generate_embedding(paper["title"])
    embedding = generate_embedding(paper["summary"])

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO arxiv_papers (title, authors, summary, url, published, categories, embedding, cluster)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (paper["title"], paper["authors"], paper["summary"], paper["url"],
           paper["published"], paper["categories"], sqlite3.Binary(np.array(embedding, dtype=np.float32)), -1))
    conn.commit()
    conn.close()
    return True

def fetch_arxiv_papers(query="all"):
    print("🔍 Querying arXiv for new papers...")
    search = arxiv.Search(query=query, sort_by=arxiv.SortCriterion.SubmittedDate, max_results=500)
    client = arxiv.Client()
    papers = []

    try:
        for paper in client.results(search):
            if paper.published < LAST_24_HOURS:
                continue

            papers.append({
                "title": paper.title,
                "authors": ", ".join([author.name for author in paper.authors]),
                "summary": paper.summary,
                "url": paper.entry_id,
                "published": paper.published.strftime("%Y-%m-%d %H:%M:%S"),
                "categories": ", ".join(paper.categories)
            })
        print(f"✅ Found {len(papers)} new papers from arXiv.")
    except Exception as e:
        print(f"❌ Error while fetching papers: {e}")

    return papers

def process_new_papers():
    print("🔍 Fetching recent arXiv papers...")
    papers = fetch_arxiv_papers()

    if not papers:
        print("⚠️ No new papers found in the last 24 hours.")
        return

    saved_count = 0
    for paper in papers:
        if save_paper(paper):
            saved_count += 1

    print(f"✅ {saved_count} new papers saved.")

# Step 3: Clustering (Includes All Papers, Old & New)
def cluster_papers():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, embedding FROM arxiv_papers")
    rows = cursor.fetchall()

    if not rows:
        return

    embeddings = []
    paper_ids = []

    for row in rows:
        paper_ids.append(row[0])
        embeddings.append(np.frombuffer(row[1], dtype=np.float32))

    embeddings = np.vstack(embeddings)

    # Normalize embeddings (L2 normalization)
    embeddings = normalize(embeddings, norm='l2')

    #Perform clustering
    clustering = DBSCAN(eps=1.0, min_samples=2).fit(embeddings)
    #clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1).fit(embeddings)

    for i, paper_id in enumerate(paper_ids):
        cursor.execute("UPDATE arxiv_papers SET cluster = ? WHERE id = ?", (int(clustering.labels_[i]), paper_id))

    conn.commit()
    conn.close()
    print("✅ Clustering completed!")

if __name__ == "__main__":
    setup_database()
    process_new_papers()
    cluster_papers()
