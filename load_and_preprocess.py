import re
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
import unicodedata

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def preprocess_text(text):
    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Preserve hyphens and fix word splits
    text = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove special characters but preserve punctuation
    text = re.sub(r"[^a-zA-Z0-9.,!?;'\s-]", "", text)

    # Tokenize sentences instead of breaking words apart
    sentences = sent_tokenize(text)

    #  Print how sentences look
    print("\n First 5 sentences after preprocessing:\n", sentences[:5])
    return sentences

def load_and_preprocess_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None, None
    except Exception as e:
        print(f"Error during file opening: {e}")
        return None, None

    processed_tokens = preprocess_text(text)
    word_counts = Counter(processed_tokens)
    return processed_tokens, word_counts
