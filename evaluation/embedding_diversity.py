import argparse
import logging
import nltk
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import torch
import utils

from itertools import combinations
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm

nltk.download('punkt')


# PyTorch device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_average_pairwise_embedding_similarity(model, candidate):
    """Computes average pairwise sentence embedding similarities for a given candidate summary."""
    sentence_embeddings = [model.encode(s, show_progress_bar=False) for s in nltk.sent_tokenize(candidate)]
    pairwise_similarities = []
    for i, j in combinations(range(len(sentence_embeddings)), 2):
        embedding_i = sentence_embeddings[i]
        embedding_j = sentence_embeddings[j]
        pairwise_similarities.append(util.dot_score(embedding_i, embedding_j).item())
    return np.mean(pairwise_similarities)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--results_path', 
        type=str, 
        default='',
        help='Path to a results file'
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='',
        help='Name of the pre-trained SentenceTransformer model to load'
    )

    args, _ = parser.parse_known_args()

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName('INFO'),
        handlers=[logging.StreamHandler(sys.stdout)],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Load results
    logger.info('Loading book-level results')
    book_results = pd.read_csv(args.results_path)

    # Load SentenceTransformer model
    logger.info(f'Loading sentence embedding model {args.model_name}')
    model = SentenceTransformer(args.model_name).to(DEVICE)

    # Compute average pairwise sentence embedding similarities
    average_embedding_similarities = [
        compute_average_pairwise_embedding_similarity(model, candidate) \
            for candidate in book_results['candidate'].values
    ]
    logging.info(f'Average pairwise sentence embedding similarity: {np.mean(average_embedding_similarities):.4f}')

    logging.info('Finished!')