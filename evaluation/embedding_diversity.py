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


def compute_average_pairwise_embedding_similarity(model, candidate):
    """Computes sentiment overlap scores for all pairs of chapters in a graph."""
    sentence_embeddings = [model.encode(s, show_progress_bar=False) for s in nltk.sent_tokenize(candidate)]
    pairwise_similarities = []
    for i, j in combinations(range(len(sentence_embeddings)), 2):
        embedding_i = sentence_embeddings[i]
        embedding_j = sentence_embeddings[j]
        pairwise_similarities.append(util.dot_score(embedding_i, embedding_j).item())
    return np.mean(pairwise_similarities)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--results_path', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')

    args, _ = parser.parse_known_args()

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName('INFO'),
        handlers=[logging.StreamHandler(sys.stdout)],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Load data
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
    print(f'Average pairwise sentence embedding similarity: {np.mean(average_embedding_similarities):.4f}')