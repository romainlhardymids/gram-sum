import ast
import argparse
import logging
import networkx as nx
import nltk
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import torch
import utils

from datasets import Dataset
from itertools import combinations
from scipy.special import binom
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm
from torchmetrics.text.rouge import ROUGEScore
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    PegasusForConditionalGeneration,
    T5ForConditionalGeneration, 
)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_SAMPLES = 10000
K = 10


def save_book_results(results, file_path):
    """Saves book-level results to a specified path."""
    results.to_csv(file_path, index=None)


def generate_book_candidates_perplexity(model, tokenizer, chapters_list, candidate_length=512):
    """Generates book-level candidate summaries and compute their perplexities."""
    book_candidates = []
    for chapters in chapters_list:
        zipped = [(c['candidate'], c['score']) for c in chapters]
        paragraph_candidates, perplexities = zip(*zipped)
        sorted_idx = np.argsort(perplexities)
        top_paragraph_idx = []
        n = 0
        for i in sorted_idx[:K]:
            # if n + len(nltk.word_tokenize(paragraph_candidates[i])) > candidate_length:
            #     break
            n += len(nltk.word_tokenize(paragraph_candidates[i]))
            top_paragraph_idx.append(i)
        book_candidates.append(' '.join([paragraph_candidates[i] for i in sorted(top_paragraph_idx)]))
    book_candidate_tokens = tokenizer(book_candidates, truncation=False, padding='longest', return_tensors='pt').input_ids
    perplexities = []
    for i in range(book_candidate_tokens.size(0)):
        perplexities.append(utils.perplexity(model, book_candidate_tokens[i:i + 1, :], device=DEVICE))
    return book_candidates, perplexities


def evaluate_books_perplexity(
    model,
    tokenizer,
    book_loader,
    use_stemmer=True,
    candidate_length=2048,
):
    """Generate and evaluate book-level summaries."""
    rouge = ROUGEScore(use_stemmer=use_stemmer)
    t = tqdm(book_loader)
    results = {'candidate': {}, 'score': {}}
    for batch in t:
        book_ids = batch['book_id']
        texts = batch['text']
        references = batch['reference']
        chapters_list = batch['chapters']
        summary_sources = batch['summary_source']
        if len(texts) == 0:
            continue
        book_candidates, perplexities = generate_book_candidates_perplexity(
            model, 
            tokenizer, 
            chapters_list, 
            candidate_length
        )
        book_candidates = ['\n'.join(nltk.sent_tokenize(c.strip())) for c in book_candidates]
        references = ['\n'.join(nltk.sent_tokenize(r.strip())) for r in references]
        rouge.update(book_candidates, references)
        m = rouge.compute()
        rouge_1 = m['rouge1_fmeasure'].item()
        rouge_2 = m['rouge2_fmeasure'].item()
        rouge_L = m['rougeL_fmeasure'].item()
        t.set_description(f"ROUGE-1 = {rouge_1 * 100.0:0.4f} – ROUGE-2 = {rouge_2 * 100.0:0.4f} – ROUGE-L = {rouge_L * 100.0:0.4f}")
        for i in range(len(texts)):
            results['candidate'][(book_ids[i], summary_sources[i])] = book_candidates[i]
            results['score'][(book_ids[i], summary_sources[i])] = perplexities[i]
    return results


def create_book_graph(chapters, book_entities_deduped, book_entities_counts):
    """Creates an unnormalized bipartite graph linking entities to chapters."""
    chapter_candidates = [c['candidate'] for c in chapters]
    graph = nx.DiGraph()
    for i, candidate in enumerate(chapter_candidates):
        segment_node = f'SECTOR_{i + 1}'
        graph.add_node(segment_node, type='sector')
        for entity in book_entities_deduped:
            if entity in nltk.word_tokenize(candidate):
                unified_entity = book_entities_deduped[entity]
                if unified_entity not in graph:
                    graph.add_node(unified_entity, type='entity')
                if not graph.has_edge(unified_entity, segment_node):
                    graph.add_edge(unified_entity, segment_node, weight=book_entities_counts[unified_entity])
                if not graph.has_edge(segment_node, unified_entity):
                    graph.add_edge(segment_node, unified_entity, weight=1)
                else:
                    graph[segment_node][unified_entity]['weight'] += 1
    return graph


def normalize_graph(graph):
    """Normalizes a graph."""
    for source_node in graph:
        total_weight = sum([graph[source_node][target_node]['weight'] for target_node in graph[source_node]])
        for target_node in graph[source_node]:
            graph[source_node][target_node]['weight'] /= total_weight
    return graph


def compute_steady_state_distribution(T, k, epsilon=0.1):
    """Computes the steady state distribution of a RWR starting at node k on a graph."""
    n = T.shape[0]
    I = np.identity(n)
    e = np.zeros(n)
    e[k] = 1
    steady_state = epsilon * np.matmul(np.linalg.inv(I - (1 - epsilon) * T), e.T)
    return steady_state.A1


def split_nodes_by_type(graph):
    """Splits nodes by type (sector or entity)."""
    node_list = list(graph.nodes())
    sector_nodes = [node for node in node_list if graph.nodes[node]['type'] == 'sector']
    entity_nodes = [node for node in node_list if graph.nodes[node]['type'] == 'entity']
    return node_list, sector_nodes, entity_nodes


def compute_progression_scores(graph, epsilon=0.1):
    """Computes progression scores for a chapter-level graph."""
    graph = normalize_graph(graph.copy())
    epsilon = 0.1
    T = nx.adjacency_matrix(graph).tolil()
    node_list, sector_nodes, entity_nodes = split_nodes_by_type(graph)
    steady_state_sectors = {}
    for i, sector_i in enumerate(sector_nodes):
        k = node_list.index(sector_i)
        s = compute_steady_state_distribution(T, k, epsilon)
        steady_state_sectors[sector_i] = s
    steady_state_entities = {}
    for i, sector_i in enumerate(sector_nodes):
        k = node_list.index(sector_i)
        steady_state_entities.setdefault(sector_i, {})
        for entity in entity_nodes:
            c = node_list.index(entity)
            T_c = T.copy()
            T_c[c] = np.zeros(T.shape[0])
            s_c = compute_steady_state_distribution(T_c, k, epsilon)
            steady_state_entities[sector_i][entity] = s_c
    progression_scores = {}
    for i, sector_i in enumerate(sector_nodes):
        s = steady_state_sectors[sector_i]
        for j, sector_j in enumerate(sector_nodes[i + 1:]):
            l = node_list.index(sector_j)
            progression_scores.setdefault((sector_i, sector_j), 0.0)
            for entity in entity_nodes:
                s_c = steady_state_entities[sector_i][entity]
                progression_scores[(sector_i, sector_j)] += s[l] - s_c[l]
    return progression_scores


def get_entity_sets(graph):
    """Returns entity sets by sector."""
    node_list, sector_nodes, entity_nodes = split_nodes_by_type(graph)
    entity_sets = {}
    for sector in sector_nodes:
        in_edges = list(graph.in_edges(sector))
        entity_sets[sector] = set([edge[0] for edge in in_edges if edge[0] in entity_nodes])
    return entity_sets


def entity_overlap_score(entities_i, entities_j):
    """Returns the overlap score between two sets of entities."""
    a = len(entities_i.intersection(entities_j))
    b = len(entities_i.union(entities_j))
    if b == 0:
        return 0
    return 1.0 - a / b


def compute_entity_overlap_scores(graph):
    """Computes entity overlap scores for all pairs of sectors in a graph."""
    node_list, sector_nodes, entity_nodes = split_nodes_by_type(graph)
    entity_sets = get_entity_sets(graph)
    entity_overlap_scores = {}
    for i, sector_i in enumerate(sector_nodes):
        entities_i = entity_sets[sector_i]
        for j, sector_j in enumerate(sector_nodes[i + 1:]):
            entities_j = entity_sets[sector_j]
            entity_overlap_scores[(sector_i, sector_j)] = entity_overlap_score(entities_i, entities_j)
    return entity_overlap_scores


def compute_average_sentence_embeddings(model, chapters):
    """Computes average sentence embeddings for a set of paragraph summaries."""
    average_sentence_embeddings = []
    for c in chapters:
        sentence_embeddings = [model.encode(s, show_progress_bar=False) for s in nltk.sent_tokenize(c['candidate'])]
        average_embeddings = np.mean(sentence_embeddings, axis=0)
        average_sentence_embeddings.append(average_embeddings)
    return average_sentence_embeddings


def compute_sentiment_overlap_scores(average_embeddings):
    """Computes sentiment overlap scores for all pairs of sectors in a graph."""
    sentiment_overlap_scores = {}
    n = len(average_embeddings)
    for i in range(n):
        embedding_i = average_embeddings[i]
        for j in range(i + 1, n):
            embedding_j = average_embeddings[j]
            key = (f'SECTOR_{i + 1}', f'SECTOR_{j + 1}')
            sentiment_overlap_scores[key] = min(1.0 - util.dot_score(embedding_i, embedding_j).item(), 1.0)
    return sentiment_overlap_scores


def compute_diversity_scores(entity_overlap_scores, sentiment_overlap_scores):
    """Computes diversity scores for all pairs of sectors in a graph."""
    diversity_scores = {}
    for key in entity_overlap_scores:
        diversity_scores[key] = 0.5 * (entity_overlap_scores[key] + sentiment_overlap_scores[key])
    return diversity_scores


def importance_score(sector_entity_weight, total_entity_weight):
    """Returns the importance score for a given sector."""
    if total_entity_weight == 0:
        return 1.0
    else:
        return sector_entity_weight / total_entity_weight


def compute_importance_scores(graph, entity_counts):
    """Computes the importance scores for all pairs of sectors in a graph."""
    node_list, sector_nodes, entity_nodes = split_nodes_by_type(graph)
    total_entity_weight = sum([v for k, v in entity_counts.items() if k in entity_nodes])
    importance_scores = {}
    for sector in sector_nodes:
        out_edges = list(graph.out_edges(sector, data=True))
        sector_entity_weight = sum([entity_counts[edge[1]] for edge in out_edges if edge[1] in entity_nodes])
        importance_scores[sector] = importance_score(sector_entity_weight, total_entity_weight)
    return importance_scores


def get_chain_score(scores, chain, dependent=True):
    """Returns the total score for a chain given a dictionary of pairwise scores."""
    chain_score = 0.0
    if dependent:
        for i, sector_i in enumerate(chain[:-1]):
            j = i + 1
            sector_j = chain[j]
            chain_score += scores[(f'SECTOR_{sector_i + 1}', f'SECTOR_{sector_j + 1}')]
    else:
        for sector_i in chain:
            chain_score += scores[f'SECTOR_{sector_i + 1}']
    return chain_score


def choose_optimal_chain(pdi_scores, chapter_candidates, weights, candidate_length):
    """Returns the optimal chain given progression, diversity, and importance scores for all sectors."""
    num_sectors = len(pdi_scores['importance'])
    best_chain = []
    best_candidate = ''
    best_score = 0.0
    chain_length = min(K, num_sectors)
    num_possible_chains = int(binom(num_sectors, chain_length))
    counter = 0
    while counter < min(2 * num_possible_chains, NUM_SAMPLES):
        chain = sorted(random.sample(range(num_sectors), chain_length))
        book_candidate = ' '.join([chapter_candidates[i] for i in chain])
        # if len(nltk.word_tokenize(book_candidate)) > candidate_length:
        #     counter += 1
        #     continue
        progression_score = get_chain_score(pdi_scores['progression'], chain)
        diversity_score = get_chain_score(pdi_scores['diversity'], chain)
        importance_score = get_chain_score(pdi_scores['importance'], chain, dependent=False)
        score = weights['progression'] * progression_score + \
            weights['diversity'] * diversity_score + \
            weights['importance'] * importance_score
        if score > best_score:
            best_chain = chain
            best_candidate = book_candidate
            best_score = score
        counter += 1
    return best_chain, best_candidate, best_score


def generate_book_candidates_graph(
    model,
    chapters_list, 
    book_entities_deduped, 
    book_entities_counts,
    weights,
    epsilon=0.1,
    candidate_length=512
):
    """Generates chapter-level candidate summaries using a graph-based strategy."""
    book_candidates = []
    scores = []
    for chapters, entities_dict, counts in zip(chapters_list, book_entities_deduped, book_entities_counts):
        chapter_candidates = [c['candidate'] for c in chapters]
        graph = create_book_graph(chapters, entities_dict, counts)
        progression_scores = compute_progression_scores(graph, epsilon)
        entity_overlap_scores = compute_entity_overlap_scores(graph)
        average_embeddings = compute_average_sentence_embeddings(model, chapters)
        sentiment_overlap_scores = compute_sentiment_overlap_scores(average_embeddings)
        diversity_scores = compute_diversity_scores(entity_overlap_scores, sentiment_overlap_scores)
        importance_scores = compute_importance_scores(graph, counts)
        pdi_scores = {
            'progression': progression_scores,
            'diversity': diversity_scores,
            'importance': importance_scores,
        }
        chain, chapter_candidate, score = choose_optimal_chain(
            pdi_scores, 
            chapter_candidates=chapter_candidates, 
            weights=weights,
            candidate_length=candidate_length
        )
        book_candidates.append(chapter_candidate)
        scores.append(score)
    return book_candidates, scores
        

def evaluate_books_graph(
    model, 
    book_loader,
    weights,
    use_stemmer=True, 
    epsilon=0.1,
    candidate_length=512
):
    """Generates and evaluates chapter-level summaries using a graph-based strategy."""
    rouge = ROUGEScore(use_stemmer=use_stemmer)
    t = tqdm(book_loader)
    results = {'candidate': {}, 'score': {}}
    for i, batch in enumerate(t):
        book_ids = batch['book_id']
        texts = batch['text']
        references = batch['reference']
        chapters_list = batch['chapters']
        summary_sources = batch['summary_source']
        deduped_book_entities = batch['deduped_book_entities']
        deduped_book_counts = batch['deduped_book_counts']
        if len(texts) == 0:
            continue
        book_candidates, scores = generate_book_candidates_graph(
            model, 
            chapters_list, 
            deduped_book_entities, 
            deduped_book_counts,
            weights=weights,
            epsilon=epsilon,
            candidate_length=candidate_length
        )
        book_candidates = ['\n'.join(nltk.sent_tokenize(c.strip())) for c in book_candidates]
        references = ['\n'.join(nltk.sent_tokenize(r.strip())) for r in references]
        rouge.update(book_candidates, references)
        m = rouge.compute()
        rouge_1 = m['rouge1_fmeasure'].item()
        rouge_2 = m['rouge2_fmeasure'].item()
        rouge_L = m['rougeL_fmeasure'].item()
        t.set_description(f"ROUGE-1 = {rouge_1 * 100.0:0.4f} – ROUGE-2 = {rouge_2 * 100.0:0.4f} – ROUGE-L = {rouge_L * 100.0:0.4f}")
        for i in range(len(texts)):
            results['candidate'][(book_ids[i], summary_sources[i])] = book_candidates[i]
            results['score'][(book_ids[i], summary_sources[i])] = scores[i]
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--auth_token', type=str, default='')

    parser.add_argument('--strategy', type=str, choices=['perplexity', 'graph'], default='perplexity')
    parser.add_argument('--use_stemmer', type=bool, default=False)
    parser.add_argument('--candidate_length', type=int, default=3)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--progression_weight', type=float, default=1.0)
    parser.add_argument('--diversity_weight', type=float, default=1.0)
    parser.add_argument('--importance_weight', type=float, default=1.0)

    args, _ = parser.parse_known_args()

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName('INFO'),
        handlers=[logging.StreamHandler(sys.stdout)],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Load data
    logger.info('Loading book-level data')
    chapter_results_path = os.path.join(args.root_dir, f'data/chapter/test_t5small_{args.strategy}_k5.csv')
    book_dataset = utils.BookSumBookDataset(args.root_dir, 'test', chapter_results_path)
    book_loader = DataLoader(
        book_dataset, 
        collate_fn=utils.book_collate_fn,
        batch_size=2, 
        pin_memory=True, 
        shuffle=False
    )

    # Chapter-level summary evaluation
    logger.info(f'Evaluating book-level summaries with strategy {args.strategy}')
    if args.strategy == 'perplexity':
        logger.info(f'Loading perplexity model {args.model_name}')
        model = T5ForConditionalGeneration.from_pretrained(args.model_name, use_auth_token=args.auth_token).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=args.auth_token)
        prefix = utils.get_prefix(args.model_name)
        results = evaluate_books_perplexity(
            model, 
            tokenizer, 
            book_loader, 
            args.use_stemmer, 
            args.candidate_length,
        )
    elif args.strategy == 'graph':
        logger.info(f'Loading sentence embedding model {args.model_name}')
        model = SentenceTransformer(args.model_name).to(DEVICE)
        weights = {
            'progression'   : args.progression_weight,
            'diversity'     : args.diversity_weight,
            'importance'    : args.importance_weight
        }
        weights = dict([(k, v / sum(weights.values())) for k, v in weights.items()])
        results = evaluate_books_graph(
            model,
            book_loader,
            weights,
            args.use_stemmer,
            args.epsilon,
            args.candidate_length
        )
        
    book_raw_data = pd.DataFrame(book_dataset.raw_data)
    book_raw_data['candidate'] = book_raw_data.apply(
        lambda row: results['candidate'].get((row['book_id'], row['summary_source']), None), 
        axis=1
    )
    book_raw_data = book_raw_data.dropna(subset=['candidate'])
    book_raw_data['score'] = book_raw_data.apply(lambda row: results['score'][(row['book_id'], row['summary_source'])], axis=1)

    save_path = os.path.join(args.root_dir, f'data/book/test_t5small_{args.strategy}_k{K}.csv')
    logger.info(f'Saving book-level results to {save_path}')
    save_book_results(book_raw_data, save_path)

    logger.info('Finished!')