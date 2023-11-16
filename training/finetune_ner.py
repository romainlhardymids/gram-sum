import argparse
import ast
import logging
import nltk
import numpy as np
import os
import pandas as pd
import sys
import torch
import transformers

from datasets import load_dataset, load_metric
from itertools import product
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    pipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)

nltk.download('punkt')


# PyTorch device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def align_labels_with_tokens(labels, word_ids):
    """Aligns NER labels with the corresponding tokens."""
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_dir',
        type=str,
        default='',
        help='Directory to save model checkpoints to'
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='bert-base-cased',
        help='Name of the pre-trained model to load'
    )
    parser.add_argument(
        '--auth_token', 
        type=str, 
        default='',
        help='Authentication token for loading models from the Huggingface hub'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=2,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--train_batch_size', 
        type=int, 
        default=2,
        help='Training batch size'
    )
    parser.add_argument(
        '--valid_batch_size', 
        type=int, 
        default=2,
        help='Validation batch size'
    )
    parser.add_argument(
        '--warmup_steps', 
        type=int, 
        default=500,
        help='Number of warmup steps for the learning rate scheduler'
    )
    parser.add_argument(
        '--learning_rate', 
        type=str, 
        default=2.0e-5,
        help='Learning rate for backpropagation'
    )
    parser.add_argument(
        '--gradient_accumulation_steps', 
        type=int, 
        default=1,
        help='Number of gradient accumulation steps (to reduce memory consumption)'
    )

    args, _ = parser.parse_known_args()

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load the CoNLL 2003 dataset
    raw_datasets = load_dataset('conll2003')
    ner_feature = raw_datasets['train'].features['ner_tags']
    label_names = ner_feature.feature.names
    logging.info(f'NER labels: {label_names}')

    # Load the pre-trained model
    logger.info(f'Loading model {args.model_name}')
    id2label = {str(i): label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        id2label=id2label,
        label2id=label2id
    ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess(batch):
        tokenized_inputs = tokenizer(
            batch['tokens'], 
            truncation=True, 
            is_split_into_words=True
        )
        all_labels = batch['ner_tags']
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))
        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs
    
    # Pre-process inputs for token classification
    logger.info('Preprocessing inputs for token classification')
    tokenized_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        remove_columns=raw_datasets['train'].column_names
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Devine evaluation metrics
    metric = load_metric('seqeval')
    def compute_metrics(outputs):
        logits, labels = outputs
        predictions = np.argmax(logits, axis=-1)
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            'precision': all_metrics['overall_precision'],
            'recall': all_metrics['overall_recall'],
            'f1': all_metrics['overall_f1'],
            'accuracy': all_metrics['overall_accuracy']
        }

    # Define the training arguments
    training_args = TrainingArguments(
        args.output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.valid_batch_size,
        learning_rate=float(args.learning_rate),
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        push_to_hub=True,
        hub_model_id='finetuned-ner',
        push_to_hub_token=args.auth_token
    )

    # Create a trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # Fine-tune the model
    logger.info('Fine-tuning the model')
    trainer.train()

    # Push the model to the Huggingface hub
    logger.info('Saving the model to Huggingface')
    trainer.push_to_hub(commit_message='Training complete')

    logger.info('Finished!')