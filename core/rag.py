"""Task 04 - RAG over the curated palmistry knowledge base (TF-IDF retrieval, no external service)."""
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "palmistry.md")


class PalmistryKB:
    def __init__(self, path: str = KB_PATH):
        text = open(path, encoding="utf-8").read()
        self.chunks = []
        for block in re.split(r"^## ", text, flags=re.M)[1:]:
            title, _, body = block.partition("\n")
            self.chunks.append({"title": title.strip(), "text": body.strip()})
        docs = [c["title"].replace("_", " ") + " " + c["text"] for c in self.chunks]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.mat = self.vec.fit_transform(docs)

    def retrieve(self, query: str, k: int = 6) -> list[dict]:
        q = self.vec.transform([query.replace("_", " ")])
        sims = cosine_similarity(q, self.mat)[0]
        idx = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": round(float(sims[i]), 3)} for i in idx if sims[i] > 0]


def features_to_query(features_by_hand: dict) -> str:
    """Turn structured CV features into a retrieval query (always include section overviews)."""
    words = ["career financial relationship personality future left versus right uncertainty"]
    for hand, f in features_by_hand.items():
        for ln in f["lines"]:
            if ln["detected"]:
                words.append(f"{ln['name']} overview")
                if ln["length_ratio"] >= 0.6: words.append(f"{ln['name']} long")
                else: words.append(f"{ln['name']} short")
                words.append("curved" if ln["curvature"] > 1.08 else "straight")
                words.append(ln["depth"])
            else:
                words.append(f"{ln['name']} absent")
        words.append(f["palm_shape"].get("shape", "").split(" ")[0] + " hand shape")
    return " ".join(words)
