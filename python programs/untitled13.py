import math
import numpy as np
from collections import Counter
import re
import nltk
from nltk.util import ngrams

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_perplexity(text, n=2):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = nltk.word_tokenize(text)
    
    if len(tokens) < n:
        return 0.0

    n_gram_list = list(ngrams(tokens, n))
    N = len(n_gram_list)
    
    freq_dist = Counter(n_gram_list)
    
    entropy = 0
    for ng in freq_dist:
        probability = freq_dist[ng] / N
        entropy -= probability * math.log2(probability)
    
    perplexity = math.pow(2, entropy)
    
    return perplexity

sample_text = "The cat sat on the mat. The cat sat on the mat."
print(f"Simple Text Perplexity: {calculate_perplexity(sample_text, n=2):.4f}")

complex_text = "The quick brown fox jumps over the lazy dog in a field of green."
print(f"Complex Text Perplexity: {calculate_perplexity(complex_text, n=2):.4f}")