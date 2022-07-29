import random
import logging
import sys
import argparse
import os
import torch

from datasets import Dataset, load_from_disk, load_metric
from functools import partial
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    PegasusForConditionalGeneration,
    T5ForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)


def load_pretrained_model(model_name):
    """Loads a pre-trained conditional generation model from the Huggingface hub."""
    if 'bart' in model_name:
        return BartForConditionalGeneration.from_pretrained(model_name)
    elif 'pegasus' in model_name:
        return PegasusForConditionalGeneration.from_pretrained(model_name)
    elif 't5' in model_name:
        return T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        raise ValueError(f'Model {model_name} is not supported.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Hyperparameters passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--learning_rate', type=str, default=2.0e-5)
    
    # Huggingface hub arguments
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_strategy", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)
    

    # Data, model, and output directories
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--n_gpus', type=str, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--train_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--valid_dir', type=str, default=os.environ['SM_CHANNEL_VALID'])

    args, _ = parser.parse_known_args()

    # Logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Load datasets
    logger.info('Loading datasets...')
    train_dataset = load_from_disk(args.train_dir)
    valid_dataset = load_from_disk(args.valid_dir)

    # Download model from model hub
    logger.info(f'Loading {args.model_name} model...')
    model = load_pretrained_model(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy='epoch',
        logging_dir=f'{args.output_data_dir}/logs',
        learning_rate=float(args.learning_rate),
        push_to_hub=args.push_to_hub,
        hub_strategy=args.hub_strategy,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
    )
    
    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train model
    logger.info('Fine-tuning the model...')
    trainer.train()

    # Push the model to the Huggingface hub
    logger.info('Saving the model...')
    trainer.create_model_card(model_name=args.hub_model_id)
    trainer.push_to_hub(commit_message='Training complete')
