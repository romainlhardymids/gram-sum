import argparse
import logging
import os
import pandas as pd
import sys


# Partial paths in the BookSum directory
PATHS = {
    'book': {
        'train': 'alignments/book-level-summary-alignments/book_summaries_aligned_train.jsonl',
        'valid': 'alignments/book-level-summary-alignments/book_summaries_aligned_val.jsonl',
        'test': 'alignments/book-level-summary-alignments/book_summaries_aligned_test.jsonl'
    },
    'chapter': {
        'train': 'alignments/chapter-level-summary-alignments/chapter_summary_aligned_train_split.jsonl',
        'valid': 'alignments/chapter-level-summary-alignments/chapter_summary_aligned_val_split.jsonl',
        'test': 'alignments/chapter-level-summary-alignments/chapter_summary_aligned_test_split.jsonl'
    },
    'paragraph': {
        'train': 'alignments/paragraph-level-summary-alignments/chapter_summary_aligned_train_split.jsonl.gathered.stable',
        'valid': 'alignments/paragraph-level-summary-alignments/chapter_summary_aligned_val_split.jsonl.gathered.stable',
        'test': 'alignments/paragraph-level-summary-alignments/chapter_summary_aligned_test_split.jsonl.gathered.stable'
    }
}


def load_file(path):
    """Loads a Pandas data frame from a JSON file."""
    return pd.read_json(path, lines=True)


def load_data(booksum_dir):
    """Loads all BookSum data."""
    levels = ['book', 'chapter', 'paragraph']
    splits = ['train', 'valid', 'test']
    data = {}
    metadata = {}
    for l in levels:
        data[l] = {}
        metadata[l] = {}
        for s in splits:
            path = os.path.join(booksum_dir, PATHS[l][s])
            df = load_file(path)
            examples, features = df.shape
            data[l][s] = df
            metadata[l][s] = {'examples': examples, 'features': features}
    return data, metadata


def clean_book_data(df):
    """Cleans book-level data."""
    df = df.rename(columns={'bid': 'book_id', 'source': 'summary_source', 'title': 'book_title'})
    return df


def clean_chapter_data(df):
    """Cleans chapter-level data."""
    df = df.rename(columns={'book_id': 'chapter_id'})
    df = df.rename(columns={'bid': 'book_id', 'source': 'summary_source'})
    df['chapter_id'] = df['chapter_id'].apply(lambda s: s.lower().replace(' ', '_'))
    df['chapter_order'] = df.groupby(['book_id', 'summary_source'])['chapter_id'] \
        .apply(lambda group: pd.Series(range(len(group)), group.index))
    df = df.drop(columns=['summary_id'])
    return df


def clean_paragraph_data(df):
    """Cleans paragraph-level data."""
    df['chapter_id'] = df['title'].apply(lambda s: '.'.join(s.split('.')[:-1]))
    df['summary_source'] = df['title'].apply(lambda s: s.split('.')[-1].split('-')[0])
    df['paragraph_order'] = df.groupby(['chapter_id', 'summary_source'])['chapter_id'] \
        .apply(lambda group: pd.Series(range(len(group)), group.index))
    df = df.drop(columns=['title'])
    return df


def clean_data(data):
    """Master function that cleans all BookSum data."""
    cleaner = {
        'book': clean_book_data, 
        'chapter': clean_chapter_data,
        'paragraph': clean_paragraph_data
    }
    for l in data:
        for s in data[l]:
            data[l][s] = cleaner[l](data[l][s])
    return data


def save_cleaned_data(data, save_dir):
    """Saves cleaned data to a specified directory."""
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for l in data:
        sub_dir = os.path.join(save_dir, f'{l}')
        if not os.path.isdir(sub_dir):
            os.mkdir(sub_dir)
        for s in data[l]:
            data[l][s].to_csv(os.path.join(sub_dir, f'{s}.csv'), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--booksum_dir', 
        type=str, 
        default='', 
        help='Directory containing the BookSum data'
    )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='', 
        help='Directory to save the cleaned data to'
    )

    args, _ = parser.parse_known_args()

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName('INFO'),
        handlers=[logging.StreamHandler(sys.stdout)],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Load data
    logger.info(f'Loading data from {args.booksum_dir}')
    data, metadata = load_data(args.booksum_dir)
    for l in metadata:
        for s in metadata[l]:
            logger.info(f"Dataset {l}-{s} has {metadata[l][s]['examples']} examples and {metadata[l][s]['features']} features")

    # Clean and save data
    logger.info(f'Saving cleaned data to {args.save_dir}')
    data = clean_data(data)
    save_cleaned_data(data, args.save_dir)
    
    logger.info('Finished!')