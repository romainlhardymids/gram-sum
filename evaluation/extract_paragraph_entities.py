import ast
import argparse
import logging
import os
import pandas as pd
import re
import spacy
import sys
import torch

from datasets import Dataset
from fuzzywuzzy import fuzz, process
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline,
)

ENTITY_TYPES = set(['PER', 'LOC', 'ORG'])
LEVELS = ['paragraph', 'chapter', 'book']
NER_MODEL_NAME = 'romainlhardy/bert-finetuned-ner'
DEVICE = 0 if torch.cuda.is_available() else -1


def load_test_data(root_dir):
    """Loads data for the test split."""
    data = {}
    for level in LEVELS:
        file_path = os.path.join(root_dir, f'data/{level}/test.csv')
        data[level] = pd.read_csv(file_path)
    return data


def save_data_with_entities(data, root_dir):
    """Saves data with identified entities."""
    for level in LEVELS:
        file_path = os.path.join(root_dir, f'data/{level}/test.csv')
        data[level].to_csv(file_path, index=False)


def aggregate_entities(data, target_key, aggregate_key):
    """Aggregates unique entities by a given key."""
    func = lambda group: list(set(sum([g for g in group], [])))
    aggregated = data.groupby(aggregate_key)[target_key].sum()
    aggregated = aggregated.apply(lambda entities: list(set(entities)))
    return aggregated


def deduplicate_entities(entities, top_entities=100, threshold=70, scorer=fuzz.token_set_ratio):
    """Returns a dictionary of deduplicated entities and their occurrence counts."""
    deduplicated = {}
    counts = {}
    for entity in entities:
        matches = process.extract(entity, entities, limit=None, scorer=scorer)
        filtered = [m for m in matches if m[1] > threshold]
        if len(filtered) == 1:
            deduplicated[entity] = entity
            counts.setdefault(entity, 0)
            counts[entity] += 1
        else:
            filtered = sorted(filtered, key=lambda x: x[0])
            filtered = sorted(filtered, key=lambda x: len(x[0]), reverse=True)
            unified_entity = filtered[0][0]
            deduplicated[entity] = unified_entity
            counts.setdefault(unified_entity, 0)
            counts[unified_entity] += 1
    counts = dict(sorted(counts.items(), key=lambda x: -x[1])[:top_entities])
    deduplicated = dict([(k, v) for k, v in deduplicated.items() if v in counts])
    return deduplicated, counts


def map_dictionary(df, d, key, col_name, instance="list"):
    """Maps a dictionary of values to a data frame column by key."""
    df[col_name] = df[key].map(d)
    if instance == "list":
        df[col_name] = df[col_name].apply(lambda e: e if isinstance(e, list) else [])
    else:
        df[col_name] = df[col_name].apply(lambda e: e if isinstance(e, dict) else {})
    return df


def map_deduplicated_entities(df, deduped_entities, deduped_counts):
    """Maps deduplicated book-level entities and counts to a data frame."""
    df = map_dictionary(df, deduped_entities, 'book_id', 'deduped_book_entities', instance="dict")
    df = map_dictionary(df, deduped_counts, 'book_id', 'deduped_book_counts', instance="dict")
    return df
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root_dir', 
        type=str, 
        default='',
        help='Root directory for the project'
    )
    parser.add_argument(
        '--auth_token', 
        type=str, 
        default='',
        help='Authentication token for loading the NER model from the Huggingface hub'
    )
    parser.add_argument(
        '--top_entities',
        type=int,
        default=100,
        help='Only consider the top n entities by frequency'
    )

    args, _ = parser.parse_known_args()

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName('INFO'),
        handlers=[logging.StreamHandler(sys.stdout)],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    logger.info('Loading NER model')
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME, use_auth_token=args.auth_token)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    ner = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='first', device=DEVICE)

    def get_paragraph_entities(batch):
        entities = ner(batch['text'])
        entities = [[e for e in E if e['score'] > 0.99] for E in entities]
        entities = [[re.sub("[^A-Za-z0-9 '.-]+", '', e['word'].strip()) for e in E] for E in entities]
        entities = [[e for e in E if len(e) > 1] for E in entities]
        return {'entities': entities}

    logger.info('Loading test data...')
    test_data = load_test_data(args.root_dir)

    logger.info('Extracting sentence-level entities')
    paragraph_data = test_data['paragraph']
    ner_dataset = Dataset.from_pandas(paragraph_data[['text']])
    ner_dataset = ner_dataset.map(get_paragraph_entities, batched=True)
    paragraph_entities = ner_dataset['entities']
    paragraph_data['paragraph_entities'] = paragraph_entities

    logger.info('Aggregating chapter-level entities')
    chapter_data = test_data['chapter']
    chapter_entities = aggregate_entities(paragraph_data, 'paragraph_entities', 'chapter_id')
    chapter_data = map_dictionary(chapter_data, chapter_entities, 'chapter_id', 'chapter_entities')

    logger.info('Aggregating book-level entities')
    book_data = test_data['book']
    book_entities = aggregate_entities(chapter_data, 'chapter_entities', 'book_id')
    book_data = map_dictionary(book_data, book_entities, 'book_id', 'book_entities')

    logger.info('Deduplicating book-level entities')
    deduped = [(k, deduplicate_entities(v, top_entities=args.top_entities)) for k, v in book_entities.items()]
    deduped_entities = dict([(d[0], d[1][0]) for d in deduped])
    deduped_counts = dict([(d[0], d[1][1]) for d in deduped])

    chapter_data = map_deduplicated_entities(chapter_data, deduped_entities, deduped_counts)
    book_data = map_deduplicated_entities(book_data, deduped_entities, deduped_counts)

    logger.info('Saving data')
    save_data_with_entities(test_data, args.root_dir)

    logger.info('Finished!')