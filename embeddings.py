import logging
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

DOMAIN_TERMS = [
    'budget', 'fiscal', 'revenue', 'expenditure', 'deficit', 'surplus',
    'tax', 'policy', 'growth', 'economic', 'projection', 'gdp',
    'inflation', 'election', 'votes', 'region', 'district', 'candidate',
    'government', 'finance', 'spending', 'investment'
]


def create_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    """Create embeddings for text chunks using SentenceTransformer."""
    logger.info(f"Starting embedding pipeline: model={model_name}, chunks={len(chunks)}")
    model = SentenceTransformer(model_name)
    logger.debug(f"Model loaded: {model_name}")

    embeddings = model.encode(chunks)
    logger.info(f"Embeddings created: shape={embeddings.shape}, dtype={embeddings.dtype}")
    return embeddings, model


def compute_domain_score(text, query, domain_terms=None):
    """Compute a heuristic domain score based on shared budget/election terminology."""
    terms = domain_terms or DOMAIN_TERMS
    text_lower = text.lower()
    query_lower = query.lower()

    doc_terms = {term for term in terms if term in text_lower}
    query_terms = {term for term in terms if term in query_lower}
    if query_terms:
        score = len(doc_terms & query_terms) / len(query_terms)
    else:
        score = min(1.0, len(doc_terms) / 4)

    return round(score, 4)


def compute_keyword_score(text, query):
    """Compute keyword overlap between query and document text."""
    query_tokens = {token.strip('.,?"\'').lower() for token in query.split() if len(token.strip('.,?"\'')) > 2}
    doc_tokens = {token.strip('.,?"\'').lower() for token in text.split() if len(token.strip('.,?"\'')) > 2}
    if not query_tokens:
        return 0.0
    overlap = query_tokens & doc_tokens
    return round(len(overlap) / len(query_tokens), 4)


def store_in_chroma(chunks, embeddings, collection_name='budget_chunks'):
    """Store chunks and embeddings in Chroma."""
    logger.info(f"Starting Chroma storage: collection={collection_name}, chunks={len(chunks)}")
    client = chromadb.PersistentClient(path='./chroma_db')
    logger.debug("Chroma client initialized")

    collection = client.get_or_create_collection(name=collection_name)
    logger.debug(f"Collection '{collection_name}' created/retrieved")

    ids = [f'chunk_{i}' for i in range(len(chunks))]
    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=ids
    )
    logger.info(f"Chroma storage completed: {len(chunks)} chunks stored in '{collection_name}'")
    return collection


def load_chroma_collection(collection_name='budget_chunks'):
    """Load an existing Chromadb collection without adding data."""
    logger.info(f"Loading Chroma collection: {collection_name}")
    client = chromadb.PersistentClient(path='./chroma_db')
    collection = client.get_or_create_collection(name=collection_name)
    return collection


def retrieve_top_k(collection, query, model, k=5, use_domain_scoring=True, domain_weight=0.3, keyword_weight=0.2):
    """Retrieve top-k similar chunks and optionally apply domain-specific scoring."""
    logger.debug(f"Retrieving top-{k}: query='{query}' with domain scoring={use_domain_scoring}")
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )

    docs = results['documents'][0]
    distances = results['distances'][0]
    similarities = [1 - d for d in distances]
    domain_scores = [compute_domain_score(doc, query) for doc in docs]
    keyword_scores = [compute_keyword_score(doc, query) for doc in docs]

    combined_scores = []
    for sim, dom, key in zip(similarities, domain_scores, keyword_scores):
        if use_domain_scoring:
            combined = ((1.0 - domain_weight - keyword_weight) * sim
                        + domain_weight * dom
                        + keyword_weight * key)
        else:
            combined = sim
        combined_scores.append(round(combined, 4))

    sorted_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
    docs = [docs[i] for i in sorted_indices]
    distances = [distances[i] for i in sorted_indices]
    similarities = [similarities[i] for i in sorted_indices]
    domain_scores = [domain_scores[i] for i in sorted_indices]
    keyword_scores = [keyword_scores[i] for i in sorted_indices]
    combined_scores = [combined_scores[i] for i in sorted_indices]

    logger.debug(f"Retrieved {len(docs)} chunks. Top similarity: {similarities[0]:.3f}, top domain: {domain_scores[0]:.3f}, top keyword: {keyword_scores[0]:.3f}, top combined: {combined_scores[0]:.3f}")
    return docs, distances, similarities, domain_scores, keyword_scores, combined_scores


def hybrid_search(collection, query, k=5):
    """Perform hybrid search (keyword + vector) using Chroma."""
    logger.debug(f"Hybrid search: query='{query}', k={k}")
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=['documents', 'distances', 'metadatas']
    )
    docs = results['documents'][0]
    distances = results['distances'][0]
    logger.debug(f"Hybrid search retrieved {len(docs)} chunks")
    return docs, distances
