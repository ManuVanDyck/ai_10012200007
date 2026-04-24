import logging
import os
from datetime import datetime
from data_processing import manage_context_window
from embeddings import retrieve_top_k
from prompting import (
    generate_prompt_template_v1,
    generate_prompt_template_v2,
    generate_prompt_template_v3,
    generate_prompt_template_v4,
    simulate_llm_response,
)

logger = logging.getLogger(__name__)


def configure_logging(log_dir='./logs', level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'rag_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Logging configured: {log_file}")
    return log_file


def end_to_end_pipeline(user_query, collection, model, prompt_version='v2', display=True):
    """Full pipeline from query to response with logging and optional display."""
    logger.info("\n" + "="*80)
    logger.info("COMPLETE END-TO-END PIPELINE EXECUTION")
    logger.info("="*80)

    logger.info("[STAGE 1] USER QUERY INPUT")
    logger.info(f"Query: '{user_query}'")
    if display:
        print(f"\n{'='*80}")
        print(f"USER QUERY: {user_query}")
        print(f"{'='*80}\n")

    logger.info("[STAGE 2] RETRIEVAL")
    top_docs, distances, similarities, domain_scores, keyword_scores, combined_scores = retrieve_top_k(
        collection,
        user_query,
        model,
        k=5,
        use_domain_scoring=True,
        domain_weight=0.3,
        keyword_weight=0.2
    )
    logger.info(f"Retrieved {len(top_docs)} documents")
    logger.info(f"Similarity scores: {[f'{s:.4f}' for s in similarities]}")
    logger.info(f"Domain scores: {[f'{d:.4f}' for d in domain_scores]}")
    logger.info(f"Keyword scores: {[f'{k:.4f}' for k in keyword_scores]}")
    logger.info(f"Combined scores: {[f'{c:.4f}' for c in combined_scores]}")
    for i, (doc, sim, dom, key, comb, dist) in enumerate(zip(top_docs, similarities, domain_scores, keyword_scores, combined_scores, distances)):
        logger.info(f"  Result {i+1}: Similarity={sim:.4f}, Domain={dom:.4f}, Keyword={key:.4f}, Combined={comb:.4f}, Distance={dist:.4f}")

    if display:
        print(f"RETRIEVED DOCUMENTS:")
        for i, (doc, sim, dom, key, comb) in enumerate(zip(top_docs, similarities, domain_scores, keyword_scores, combined_scores), start=1):
            print(f"\n[Document {i}] Similarity: {sim:.4f}, Domain: {dom:.4f}, Keyword: {key:.4f}, Combined: {comb:.4f}")
            print(doc[:300] + ('...' if len(doc) > 300 else ''))

    context = manage_context_window(top_docs, max_length=1500)
    logger.info(f"[STAGE 3] CONTEXT SELECTION: {len(context)} chars selected")
    logger.debug(f"Selected context preview: {context[:300]}...")

    if display:
        print(f"\n{'='*80}")
        print(f"CONTEXT SELECTION")
        print(f"{'='*80}")
        print(f"Context length: {len(context)} characters")
        print(context[:300] + ('...' if len(context) > 300 else ''))

    if prompt_version == 'v1':
        prompt = generate_prompt_template_v1(user_query, context)
    elif prompt_version == 'v2':
        prompt = generate_prompt_template_v2(user_query, context)
    elif prompt_version == 'v3':
        prompt = generate_prompt_template_v3(user_query, context)
    elif prompt_version == 'v4':
        prompt = generate_prompt_template_v4(user_query, context)
    else:
        prompt = generate_prompt_template_v4(user_query, context)
    logger.info(f"[STAGE 4] PROMPT GENERATED: template {prompt_version}")
    logger.debug(f"Prompt content:\n{prompt}")

    if display:
        print(f"\n{'='*80}")
        print(f"FINAL PROMPT SENT TO LLM")
        print(f"{'='*80}\n")
        print(prompt)

    logger.info("[STAGE 5] LLM RESPONSE GENERATION")
    response = simulate_llm_response(prompt)
    logger.info("LLM response generated")
    logger.info(f"LLM response: {response}")
    if display:
        print(f"\n{'='*80}")
        print(f"LLM RESPONSE")
        print(f"{'='*80}")
        print(response)

    result = {
        'user_query': user_query,
        'retrieved_documents': top_docs,
        'similarity_scores': similarities,
        'domain_scores': domain_scores,
        'keyword_scores': keyword_scores,
        'combined_scores': combined_scores,
        'context': context,
        'prompt': prompt,
        'prompt_version': prompt_version,
        'response': response,
        'pipeline_metadata': {
            'num_docs_retrieved': len(top_docs),
            'top_similarity': similarities[0] if similarities else 0.0,
            'top_domain_score': domain_scores[0] if domain_scores else 0.0,
            'top_keyword_score': keyword_scores[0] if keyword_scores else 0.0,
            'top_combined_score': combined_scores[0] if combined_scores else 0.0,
            'context_length': len(context),
            'prompt_length': len(prompt),
            'response_length': len(response),
        }
    }

    logger.info("[STAGE 6] PIPELINE SUMMARY")
    logger.info(f"Top similarity score: {result['pipeline_metadata']['top_similarity']:.4f}")
    logger.info(f"Prompt length: {result['pipeline_metadata']['prompt_length']} chars")
    return result


def evaluate_adversarial_queries(collection, model, cases, prompt_version='v2', repeats=3):
    """Evaluate a set of adversarial queries for accuracy, hallucination, and consistency."""
    logger.info("\n" + "="*80)
    logger.info("ADVERSARIAL QUERY EVALUATION")
    logger.info("="*80)

    evaluations = []
    safe_responses = {
        "I don't know.",
        "No relevant information found in context.",
        "Insufficient information in context."
    }

    for case in cases:
        query = case['query']
        expected = case.get('expected', '').strip()

        logger.info(f"Evaluating adversarial query: '{query}'")
        responses = []
        for _ in range(repeats):
            result = end_to_end_pipeline(query, collection, model, prompt_version=prompt_version, display=False)
            responses.append(result['response'])

        from collections import Counter
        response = responses[0] if responses else ''
        count = Counter(responses)
        most_common_count = count.most_common(1)[0][1] if count else 0
        consistency = most_common_count / repeats if repeats > 0 else 0.0
        accuracy = 1.0 if response.strip() == expected else 0.0
        hallucination = 0.0 if response in safe_responses else 1.0

        evaluation = {
            'query': query,
            'expected': expected,
            'response': response,
            'repeated_responses': responses,
            'accuracy': accuracy,
            'hallucination': hallucination,
            'consistency': consistency,
        }

        logger.info(f"  Response: {response}")
        logger.info(f"  Accuracy: {accuracy}")
        logger.info(f"  Hallucination: {hallucination}")
        logger.info(f"  Consistency: {consistency:.2f}")
        evaluations.append(evaluation)

    return evaluations
