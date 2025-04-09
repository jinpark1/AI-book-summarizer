import unicodedata
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import textwrap
import concurrent.futures
import evaluate
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import KFold
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Using Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"Current CUDA Device: {torch.cuda.current_device()}")
print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Load pre-trained model and tokenizer
model_path = "./models/t5_book_summarizer"
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Set it to use GPU
model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda").half()

# Print to ensure that model is running on GPU
print("Model running on:", next(model.parameters()).device)

# For cross validation
rouge = evaluate.load('rouge')

# For natural language processing
nltk.download('punkt', quiet=True)

def chunk_text(text, max_length=2500):
    # Use chunk for efficiency.
    sentences = sent_tokenize(text)
    text = " ".join(sentences)
    chunks = textwrap.wrap(text, width=max_length, break_long_words=False)

    print(f"Total Chunks: {len(chunks)}")
    return chunks

def summarize_text(text, cross_validate=False):
    if not text or len(text.split()) < 50:
        return "Text too small to summarize."

    # Perform cross-validation
    if cross_validate:
        print("\nPerforming Cross-Validation")
        avg_rouge_scores = cross_validate_summarization(text)
        return avg_rouge_scores

    # Summarization run in parallel for efficiency
    chunks = chunk_text(text)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        summaries = list(executor.map(summarize_chunk, chunks))

    combined_summary = " ".join(summaries)
    print("\nSummarizing the summaries.\n")
    final_summary = summarize_chunk(combined_summary)

    return final_summary

def summarize_chunk(chunk):
    chunk_word_count = len(chunk.split())

    # Adjust max & min summary length
    max_summary_length = min(150, chunk_word_count // 2)
    min_summary_length = min(max(20, max_summary_length // 2), max_summary_length - 1)

    # Tokenization
    inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True).to(device)

    summary_ids = model.generate(
        inputs,
        max_length=max_summary_length,
        min_length=min_summary_length,
        length_penalty=1.2, # Improves structure of the sentences.
        num_beams=5, # Increase for higher quality summary, but may take longer.
        do_sample=True, # Enables sampling
        top_p=0.9,  # Only pick from the top 90% of possible words.
        top_k=50,  # Limit randomness
        repetition_penalty=1.5, # Reduces repetitive phrases
        early_stopping=True, # Stop generation naturally
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = unicodedata.normalize("NFKC", summary)
    summary = re.sub(r"\s+", " ", summary).strip()

    print("Summary of chunk (First 100 chars):", summary[:100])
    return summary

def cross_validate_summarization(text, k=3):
    # K-fold validation with 3 folds.
    chunks = chunk_text(text)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rouge_scores = []
    final_summaries = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(chunks)):
        print(f"\nCross-Validation Fold {fold + 1}/{k}")
        val_chunks = [chunks[i] for i in val_idx]

        # Summarize each validation chunk in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            val_summaries = list(executor.map(summarize_chunk, val_chunks))

        # Get final summaries
        final_summaries.extend(val_summaries)

        individual_scores = []
        # Evaluate each validation chunk summary
        for val_chunk, val_summary in zip(val_chunks, val_summaries):
            result = rouge.compute(predictions=[val_summary], references=[val_chunk])
            individual_scores.append(result)
            print(f"ROUGE for current chunk: {result}")

        # Get ROUGE scores for the fold
        avg_fold_scores = {key: np.mean([score[key] for score in individual_scores]) for key in individual_scores[0].keys()}
        rouge_scores.append(avg_fold_scores)

    # Get average ROUGE scores across all folds
    avg_rouge_scores = {key: np.mean([score[key] for score in rouge_scores]) for key in rouge_scores[0].keys()}

    # Combine all cross-validation summaries
    combined_summary = " ".join(final_summaries)
    return combined_summary, avg_rouge_scores