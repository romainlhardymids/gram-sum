# gram-sum
GraM-SUM: A Graph-Based Model for Book-Level Summarization

# Introduction
Although summarization is a well-studied problem, most related work focuses on short texts: newspaper articles, academic papers, and short stories. By contrast, long-text summarization remains unexplored. In this paper, we introduce GraM-SUM, a graph-based model that produces summaries of book-length texts by combining chains of low-level summaries in a way that considers progression, diversity, and importance. We show that this approach outperforms a perplexity-based approach for small summary chains at the chapter and book levels. Additionally, we demonstrate that the summaries produced by GraM-SUM have more diverse sentence-level embeddings than those generated by a perplexity-based approach.

# Usage
1. [Downloading the BookSum dataset](#booksum)
2. [Preparing the BookSum data](#prepare-data)
2. [Fine-tuning a paragraph-level encoder-decoder model](#finetune-booksum)
3. [Fine-tuning a named entity recognition model](#finetune-ner)
4. [Extracting and resolving entities](#extract-entities)
5. [Generating and evaluating paragraph-level summaries](#evaluate-paragraph)
6. [Generating and evaluating chapter-level summaries](#evaluate-chapter)
7. [Generating and evaluating book-level summaries](#evaluate-book)
8. [Calculating average pairwise sentence embeddings](#pairwise-embeddings)

## Downloading the BookSum dataset<a name="booksum"></a>
To download the BookSum dataset, please follow the instructions at https://github.com/salesforce/booksum.

## Preparing the BookSum data<a name="prepare-data"></a>
```
python3 ./gram-sum/setup/load_booksum_data.py \
    --booksum_dir=/content/drive/MyDrive/W266/Final\ Project/booksum/ \
    --save_dir=/content/drive/MyDrive/W266/Final\ Project/data/
```

## Fine-tuning a paragraph-level encoder-decoder model<a name="finetune-booksum"></a>
At the paragraph level, GraM-SUM uses a `t5-small` encoder-decoder model fine-tuned on the BookSum dataset. To replicate this training, you may use the code found in `training/finetune-booksum.py` and `training/finetune_booksum.ipynb`. Although the notebook was originally run in AWS SageMaker, it can be adapted to run locally or in Google Colab.

## Fine-tuning a named entity recognition model<a name="finetune-ner"></a>
```
python3 ./gram-sum/training/finetune_ner.py \
    --output_dir=./finetuned-ner \
    --model_name=bert-base-cased \
    --auth_token=<HUGGINGFACE AUTHENTICATION TOKEN> \
    --epochs=1 \
    --train_batch_size=8 \
    --valid_batch_size=8 \
    --warmup_steps=100 \
    --learning_rate=2.0e-5 \
    --gradient_accumulation_steps=1
```

## Extracting and resolving entities<a name="extract-entities"></a>
```
python3 ./gram-sum/evaluation/extract_paragraph_entities.py \
    --root_dir=/content/drive/MyDrive/W266/Final\ Project/ \
    --top_entities=100
```

## Generating and evaluating paragraph-level summaries<a name="evaluate-paragraph"></a>
```
python3 ./gram-sum/evaluation/evaluate_paragraph_summaries.py \
    --root_dir=/content/drive/MyDrive/W266/Final\ Project/ \
    --model_name=romainlhardy/t5-small-booksum \
    --auth_token=<HUGGINGFACE AUTHENTICATION TOKEN> \
    --use_stemmer=True \
    --num_beams=3 \
    --no_repeat_ngram_size=3 \
    --min_length=30 \
    --max_length=200
```

## Generating and evaluating chapter-level summaries<a name="evaluate-chapter"></a>
```
python3 ./gram-sum/evaluation/evaluate_chapter_summaries.py \
    --root_dir=/content/drive/MyDrive/W266/Final\ Project \
    --model_name=all-mpnet-base-v2 \
    --auth_token=<HUGGINGFACE AUTHENTICATION TOKEN> \
    --strategy='graph' \
    --use_stemmer=True \
    --chain_length=5 \
    --epsilon=0.1 \
    --progression_weight=1 \
    --diversity_weight=1 \
    --importance_weight=1
```

## Generating and evaluating book-level summaries<a name="evaluate-book"></a>
```
python3 ./gram-sum/evaluation/evaluate_book_summaries.py \
    --root_dir=/content/drive/MyDrive/W266/Final\ Project \
    --model_name=all-mpnet-base-v2 \
    --auth_token=<HUGGINGFACE AUTHENTICATION TOKEN> \
    --strategy='graph' \
    --use_stemmer=True \
    --chain_length=5 \
    --epsilon=0.1 \
    --progression_weight=1 \
    --diversity_weight=1 \
    --importance_weight=1
```

## Calculating average pairwise sentence embeddings<a name="pairwise-embeddings"></a>
```
python3 ./gram-sum/evaluation/embedding_diversity.py \
    --results_path=/content/drive/MyDrive/W266/Final\ Project/data/book/test.csv \
    --model_name=all-mpnet-base-v2
```
