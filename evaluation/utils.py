import ast
import json
import os
import pandas as pd
import torch

from torch.utils.data import Dataset


class BookSumDataset(Dataset):
    """Base BookSum dataset class."""

    def __init__(self, root_dir, level, split):
        self.root_dir = root_dir
        self.raw_data_path = os.path.join(root_dir, 'data', level, split + '.csv')
        self.raw_data = self.load_raw_data()

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        return None

    def load_raw_data(self):
        df = pd.read_csv(self.raw_data_path)
        return df.to_dict(orient='records')


class BookSumParagraphDataset(BookSumDataset):
    """BookSum paragraph-level dataset class."""

    def __init__(self, root_dir, split):
        super().__init__(root_dir, 'paragraph', split)

    def __getitem__(self, i):
        paragraph = self.raw_data[i]
        return {
            'chapter_id': paragraph['chapter_id'],
            'summary_source': paragraph['summary_source'],
            'text': paragraph['text'],
            'reference': ' '.join(ast.literal_eval(paragraph['summary']))
        }


class BookSumChapterDataset(BookSumDataset):
    """BookSum chapter-level dataset class."""

    def __init__(self, root_dir, split, paragraph_results_path):
        super().__init__(root_dir, 'chapter', split)
        self.paragraph_results_path = paragraph_results_path
        self.paragraph_results = self.load_paragraph_results()
        self.booksum_dir = os.path.join(root_dir, 'booksum')

    def __getitem__(self, i):
        chapter = self.raw_data[i]
        book_id = chapter['book_id']
        chapter_id = chapter['chapter_id']
        summary_source = chapter['summary_source']
        deduped_book_entities = ast.literal_eval(chapter['deduped_book_entities'])
        deduped_book_counts = ast.literal_eval(chapter['deduped_book_counts'])
        try: 
            with open(os.path.join(self.booksum_dir, chapter['chapter_path']), 'r') as f:
                text = f.read()
            with open(os.path.join(self.booksum_dir, chapter['summary_path']), 'r') as f:
                reference = json.loads(f.readlines()[0])['summary']
        except:
            return None
        paragraphs = self.find_paragraphs_by_key(chapter_id, summary_source)
        if len(paragraphs) == 0:
            return None
        return {
            'book_id': book_id,
            'chapter_id': chapter_id,
            'summary_source': summary_source,
            'text': text,
            'reference': reference,
            'paragraphs': paragraphs,
            'deduped_book_entities': deduped_book_entities,
            'deduped_book_counts': deduped_book_counts,
        }

    def load_paragraph_results(self):
        df = pd.read_csv(self.paragraph_results_path)
        return df

    def find_paragraphs_by_key(self, chapter_id, summary_source):
        filter = (self.paragraph_results['chapter_id'] == chapter_id) & \
            (self.paragraph_results['summary_source'] == summary_source)
        paragraphs = self.paragraph_results[filter]
        return [row for row in paragraphs.to_dict(orient='records')]


def chapter_collate_fn(batch):
    """Chapter-level collate function to use in a data loader."""
    batch = [item for item in batch if item is not None]
    return {
        'book_id': [item['book_id'] for item in batch],
        'chapter_id': [item['chapter_id'] for item in batch],
        'summary_source': [item['summary_source'] for item in batch],
        'text': [item['text'] for item in batch],
        'reference': [item['reference'] for item in batch],
        'paragraphs': [item['paragraphs'] for item in batch],
        'deduped_book_entities': [item['deduped_book_entities'] for item in batch],
        'deduped_book_counts': [item['deduped_book_counts'] for item in batch],
    }


class BookSumBookDataset(BookSumDataset):
    """BookSum book-level dataset class."""

    def __init__(self, root_dir, split, chapter_results_path):
        super().__init__(root_dir, 'book', split)
        self.chapter_results_path = chapter_results_path
        self.chapter_results = self.load_chapter_results()
        self.booksum_dir = os.path.join(root_dir, 'booksum')

    def __getitem__(self, i):
        book = self.raw_data[i]
        book_id = book['book_id']
        summary_source = book['summary_source']
        summary = book['summary_path']
        text = book['book_path']
        deduped_book_entities = ast.literal_eval(book['deduped_book_entities'])
        deduped_book_counts = ast.literal_eval(book['deduped_book_counts'])
        try: 
            with open(os.path.join(self.booksum_dir, book['summary_path']), 'r') as f:
                reference = json.loads(f.readlines()[0])['summary']
            with open(os.path.join(self.booksum_dir, book['book_path']), 'r') as f:
                text = f.read()
        except:
            return None
        chapters = self.find_chapters_by_key(book_id, summary_source)
        if len(chapters) == 0:
            return None
        return {
            'book_id': book_id,
            'summary_source': summary_source,
            'text': text,
            'reference': reference,
            'chapters': chapters,
            'deduped_book_entities': deduped_book_entities,
            'deduped_book_counts': deduped_book_counts,
        }

    def load_chapter_results(self):
        df = pd.read_csv(self.chapter_results_path)
        return df

    def find_chapters_by_key(self, book_id, summary_source):
        filter = (self.chapter_results['book_id'] == book_id) & \
            (self.chapter_results['summary_source'] == summary_source)
        chapters = self.chapter_results[filter]
        return [row for row in chapters.to_dict(orient='records')]


def book_collate_fn(batch):
    """Custom book-level collate function to use in a data loader."""
    batch = [item for item in batch if item is not None]
    return {
        'book_id': [item['book_id'] for item in batch],
        'summary_source': [item['summary_source'] for item in batch],
        'text': [item['text'] for item in batch],
        'reference': [item['reference'] for item in batch],
        'chapters': [item['chapters'] for item in batch],
        'deduped_book_entities': [item['deduped_book_entities'] for item in batch],
        'deduped_book_counts': [item['deduped_book_counts'] for item in batch],
    }


def get_prefix(model_name):
    """Returns a prefix to be prepended to the input text before summarization."""
    if 't5' in model_name:
        return 'summarize: '
    else:
        return ''


def perplexity(model, tokens, device, max_length=1024, stride=128):
    """Approximately computes perplexity of an input text."""
    nll_list = []
    for i in range(0, tokens.size(1), stride):
        lower = max(i + stride - max_length, 0)
        upper = min(i + stride, tokens.size(1))
        target_length = upper - i 
        input_ids = tokens[:, lower:upper].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-target_length] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nll = outputs[0] * target_length
        nll_list.append(nll)
    perplexity = torch.exp(torch.stack(nll_list).sum() / upper)
    return perplexity.cpu().detach().numpy()