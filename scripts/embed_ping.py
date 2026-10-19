"""Check the embedding model and see similarity in action: python scripts/embed_ping.py"""
from opscopilot.embeddings import get_embedder

emb = get_embedder()
texts = [
    "Il lavoro da remoto è consentito due giorni a settimana.",   # Italian
    "Remote work is allowed two days per week.",                  # same meaning, English
    "La fattura deve essere pagata entro 30 giorni.",             # different topic
]
vecs = emb.embed_passages(texts)
print("shape:", vecs.shape)

# vectors are normalised, so the dot product is the cosine similarity
print("IT vs EN, same meaning:", round(float(vecs[0] @ vecs[1]), 3))
print("IT vs IT, other topic: ", round(float(vecs[0] @ vecs[2]), 3))
