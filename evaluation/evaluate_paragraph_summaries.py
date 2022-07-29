import ast
import argparse
import logging
import nltk
import numpy as np
import os
import pandas as pd
import sys
import time
import torch

from datasets import Dataset
from tqdm.auto import tqdm
from torchmetrics.text.rouge import ROUGEScore
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    PegasusForConditionalGeneration,
    T5ForConditionalGeneration, 
)
from utils import BookSumParagraphDataset


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_paragraph_results(results, file_path):
    """Saves results to a specified path."""
    results.to_csv(file_path, index=None)


def get_prefix(model_name):
    """Returns a prefix to be prepended to the input text."""
    if 't5' in model_name:
        return 'summarize: '
    else:
        return ''


def perplexity(model, tokens, max_length=1024, stride=128):
    """Computes the (approximate) perplexity of an input text."""
    nll_list = []
    for i in range(0, tokens.size(1), stride):
        lower = max(i + stride - max_length, 0)
        upper = min(i + stride, tokens.size(1))
        target_length = upper - i 
        input_ids = tokens[:, lower:upper].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-target_length] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nll = outputs[0] * target_length
        nll_list.append(nll)
    perplexity = torch.exp(torch.stack(nll_list).sum() / upper)
    return perplexity.cpu().detach().numpy()


def generate_paragraph_candidates(model, tokenizer, texts, **kwargs):
    """Generates candidate summaries and compute their perplexities."""
    text_tokens = tokenizer(texts, truncation=True, padding='longest', return_tensors='pt').to(DEVICE)
    candidate_tokens = model.generate(**text_tokens, **kwargs)
    paragraph_candidates = tokenizer.batch_decode(candidate_tokens, skip_special_tokens=True)
    perplexities = []
    for i in range(candidate_tokens.size(0)):
        perplexities.append(perplexity(model, candidate_tokens[i:i + 1, :]))
    return paragraph_candidates, perplexities


def evaluate_paragraphs(
    model,
    tokenizer,
    paragraph_loader,
    use_stemmer=False,
    prefix='',
    **generate_kwargs
):
    """Generate and evaluate paragraph-level summaries."""
    rouge = ROUGEScore(use_stemmer=use_stemmer)
    t = tqdm(paragraph_loader)
    results = {'candidate': [], 'perplexity': []}
    for batch in t:
        texts = batch['text']
        references = batch['reference']
        if len(texts) == 0:
            continue
        texts = [prefix + text for text in texts]
        paragraph_candidates, perplexities = generate_paragraph_candidates(
            model, 
            tokenizer, 
            texts, 
            **generate_kwargs
        )
        paragraph_candidates = ['\n'.join(nltk.sent_tokenize(c.strip())) for c in paragraph_candidates]
        references = ['\n'.join(nltk.sent_tokenize(r.strip())) for r in references]
        rouge.update(paragraph_candidates, references)
        m = rouge.compute()
        rouge_1 = m['rouge1_fmeasure'].item()
        rouge_2 = m['rouge2_fmeasure'].item()
        rouge_L = m['rougeL_fmeasure'].item()
        t.set_description(f"ROUGE-1 = {rouge_1 * 100.0:0.4f} – ROUGE-2 = {rouge_2 * 100.0:0.4f} – ROUGE-L = {rouge_L * 100.0:0.4f}")
        for i in range(len(texts)):
            results['candidate'].append(paragraph_candidates[i])
            results['perplexity'].append(perplexities[i])
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root_dir', 
        type=str, 
        default='',
        help='Root directory for the project'
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='',
        help='Name of the conditional generation model to use'
    )
    parser.add_argument(
        '--auth_token', 
        type=str, 
        default='',
        help='Authentication token for loading the NER model from the Huggingface hub'
    )
    parser.add_argument(
        '--use_stemmer', 
        type=bool, 
        default=False,
        help='Boolean indicator for whether to use stemming when computing ROUGE scores'
    )
    parser.add_argument(
        '--num_beams', 
        type=int, 
        default=1,
        help='Number of beams to use in the conditional generation step'
    )
    parser.add_argument(
        '--no_repeat_ngram_size', 
        type=int, 
        default=3,
        help='Size of n-grams to block in the generated summaries'
    )
    parser.add_argument(
        '--min_length', 
        type=int, 
        default=20,
        help='Minimum length of the generated summaries'
    )
    parser.add_argument(
        '--max_length', 
        type=int, 
        default=100,
        help='Maximum length of the generated summaries'
    )

    args, _ = parser.parse_known_args()

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName('INFO'),
        handlers=[logging.StreamHandler(sys.stdout)],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    

    # Load data
    logger.info('Loading paragraph-level data')
    paragraph_dataset = BookSumParagraphDataset(args.root_dir, 'test')
    paragraph_loader = DataLoader(
        paragraph_dataset, 
        batch_size=4, 
        pin_memory=True, 
        shuffle=False
    )

    # Load model
    logger.info(f'Loading model {args.model_name}')
    model = T5ForConditionalGeneration.from_pretrained(args.model_name, use_auth_token=args.auth_token).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=args.auth_token)
    
    # Paragraph-level summary evaluation
    logger.info('Evaluating paragraph-level summaries')
    prefix = get_prefix(args.model_name)
    use_stemmer = args.use_stemmer
    generate_kwargs = {
        'num_beams': args.num_beams,
        'no_repeat_ngram_size': args.no_repeat_ngram_size,
        'min_length': args.min_length,
        'max_length': args.max_length,
    }
    results = evaluate_paragraphs(
        model, 
        tokenizer, 
        paragraph_loader, 
        use_stemmer, 
        prefix, 
        **generate_kwargs
    )
    paragraph_df = pd.DataFrame(paragraph_dataset.raw_data)
    paragraph_df['candidate'] = results['candidate']
    paragraph_df['perplexity'] = results['perplexity']

    # Save results
    save_path = os.path.join(args.root_dir, 'data/paragraph/test.csv')
    logger.info(f'Saving paragraph-level results to {save_path}')
    save_paragraph_results(paragraph_df, save_path)

    logger.info('Finished!')