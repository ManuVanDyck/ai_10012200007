import os
import logging
from datetime import datetime
from data_processing import (
    clean_csv,
    extract_clean_pdf,
    chunk_fixed_size,
    chunk_by_sentences,
    chunk_by_paragraphs,
    filter_chunks,
)
from embeddings import create_embeddings, store_in_chroma, retrieve_top_k, hybrid_search
from pipeline import configure_logging, end_to_end_pipeline, evaluate_adversarial_queries


def main():
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = configure_logging(log_dir=log_dir)

    logger = logging.getLogger(__name__)
    logger.info("RAG PIPELINE EXECUTION STARTED")

    csv_file = 'Ghana_Election_Result.csv'
    pdf_file = '2025-Budget-Statement-and-Economic-Policy_v4.pdf'

    logger.info("STAGE 1: DATA CLEANING")
    df_clean = clean_csv(csv_file)
    logger.info(f"Cleaned CSV: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

    logger.info("STAGE 2: PDF EXTRACTION")
    pdf_text = extract_clean_pdf(pdf_file)
    logger.info(f"Extracted PDF text length: {len(pdf_text)} characters")

    logger.info("STAGE 3: CHUNKING")
    chunks_fixed = chunk_fixed_size(pdf_text, chunk_size=500, overlap=50)
    chunks_sentences = chunk_by_sentences(pdf_text, sentences_per_chunk=6, overlap=2)
    chunks_paragraphs = chunk_by_paragraphs(pdf_text, min_chars=220)
    combined_chunks = list(dict.fromkeys(chunks_sentences + chunks_paragraphs + chunks_fixed))
    chunks = filter_chunks(combined_chunks)
    if not chunks:
        logger.error("No chunks generated. Exiting.")
        return

    logger.info(f"Using {len(chunks)} filtered chunks for embedding")

    logger.info("STAGE 4: EMBEDDING CREATION")
    embeddings, model = create_embeddings(chunks)

    logger.info("STAGE 5: VECTOR STORAGE")
    collection = store_in_chroma(chunks, embeddings)

    logger.info("STAGE 6: RETRIEVAL TESTING")
    test_queries = [
        "economic growth projections",
        "government budget allocation",
        "fiscal policy measures",
        "unrelated query like sports results",
    ]
    for query in test_queries:
        logger.info(f"Retrieving for query: '{query}'")
        top_docs, distances, similarities, domain_scores, keyword_scores, combined_scores = retrieve_top_k(
            collection,
            query,
            model,
            k=3,
            use_domain_scoring=True,
            domain_weight=0.3,
            keyword_weight=0.2
        )
        logger.info(f"Top combined scores: {[f'{s:.3f}' for s in combined_scores]}")
        logger.info(f"Top similarity scores: {[f'{s:.3f}' for s in similarities]}")
        logger.info(f"Top domain scores: {[f'{d:.3f}' for d in domain_scores]}")
        logger.info(f"Top keyword scores: {[f'{k:.3f}' for k in keyword_scores]}")
        hybrid_docs, hybrid_distances = hybrid_search(collection, query, k=3)
        logger.info(f"Hybrid distances: {[f'{d:.3f}' for d in hybrid_distances]}")

    logger.info("STAGE 7: COMPLETE PIPELINE DEMONSTRATION")
    end_to_end_queries = [
        "What are the economic growth projections?",
        "How is government budget allocated?",
    ]
    for query in end_to_end_queries:
        result = end_to_end_pipeline(query, collection, model, prompt_version='v4')
        logger.info(f"Pipeline result: top_similarity={result['pipeline_metadata']['top_similarity']:.4f}")

    logger.info("STAGE 8: ADVERSARIAL QUERY EVALUATION")
    adversarial_cases = [
        {
            'query': "What is the current budget?",
            'expected': "Insufficient information in context.",
        },
        {
            'query': "What football-related spending is included in the budget?",
            'expected': "No relevant information found in context.",
        },
    ]
    evaluations = evaluate_adversarial_queries(collection, model, adversarial_cases, prompt_version='v2', repeats=3)
    for eval_item in evaluations:
        logger.info("Adversarial evaluation result:")
        logger.info(f"  Query: {eval_item['query']}")
        logger.info(f"  Expected: {eval_item['expected']}")
        logger.info(f"  Response: {eval_item['response']}")
        logger.info(f"  Accuracy: {eval_item['accuracy']}")
        logger.info(f"  Hallucination: {eval_item['hallucination']}")
        logger.info(f"  Consistency: {eval_item['consistency']:.2f}")

    logger.info("RAG PIPELINE EXECUTION COMPLETED")
    logger.info(f"Log file: {log_file}")


if __name__ == '__main__':
    main()
