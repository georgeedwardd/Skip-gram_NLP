def generate_pairs(sequence, window_size):
    pairs = []
    
    for i, center in enumerate(sequence):
        for j in range(-window_size, window_size + 1):
            if j == 0 or i + j < 0 or i + j >= len(sequence):
                continue
            context = sequence[i + j]
            pairs.append((center, context))
    return pairs

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_negative_samples_batch(contexts, vocab_size, unigram_dist, k=5):
    """
    contexts: (B,) array of context word indices
    returns:  (B, k) array of negative sample indices
    """
    B = len(contexts)
    # Sample more than needed to account for collisions
    candidates = np.random.choice(vocab_size, size=(B, k * 3), p=unigram_dist)
    
    result = np.zeros((B, k), dtype=np.int32)
    for i in range(B):
        # Filter out the positive context word
        mask = candidates[i] != contexts[i]
        valid = candidates[i][mask][:k]
        # If we didn't get enough (rare), top up
        if len(valid) < k:
            extra = np.random.choice(vocab_size, size=k, p=unigram_dist)
            extra = extra[extra != contexts[i]]
            valid = np.concatenate([valid, extra])[:k]
        result[i] = valid
    return result

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def plot_embeddings_for_words(word_list, W, word_idx, idx_word, seed=17):
    indices = [word_idx[w] for w in word_list if w in word_idx]

    if len(indices) < 2:
        print("Not enough valid words in vocabulary.")
        return

    embeddings = W[indices]
    perplexity = min(30, len(indices) - 1)

    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    for i, idx in enumerate(indices):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
        plt.text(
            embeddings_2d[i, 0] + 0.01,
            embeddings_2d[i, 1] + 0.01,
            idx_word[idx],
            fontsize=7
        )

    plt.title("2D t-SNE of Word Embeddings")
    plt.show()

import numpy as np

def most_similar(word, word_idx, idx_word, embeddings, top_n=110):
    if word not in word_idx:
        print(f"{word} not in vocabulary.")
        return []

    vec = embeddings[word_idx[word]]
    vec = vec / np.linalg.norm(vec)

    embeddings_normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = np.dot(embeddings_normed, vec)

    # exclude the word itself
    best_idxs = np.argsort(-sims)

    results = []
    for i in best_idxs:
        if idx_word[i] == word:
            continue
        results.append((idx_word[i], 1 - sims[i]))
        if len(results) == top_n:
            break

    return results

def analogy(a, b, c, W, word_idx, idx_word, top_n=10):
    if a not in word_idx or b not in word_idx or c not in word_idx:
        print("One or more words not in vocabulary.")
        return []

    # construct analogy vector
    vec = W[word_idx[a]] - W[word_idx[b]] + W[word_idx[c]]
    vec = vec / np.linalg.norm(vec)

    # normalise embedding space
    W_norm = W / np.linalg.norm(W, axis=1, keepdims=True)

    sims = np.dot(W_norm, vec)

    # rank by similarity
    best = np.argsort(-sims)

    results = []
    for i in best:
        word = idx_word[i]
        if word in {a, b, c}:
            continue
        results.append((word, sims[i]))
        if len(results) == top_n:
            break

    return results

def word_similarity(word1, word2, word_idx, W):
    if word1 not in word_idx or word2 not in word_idx:
        return None

    v1 = W[word_idx[word1]]
    v2 = W[word_idx[word2]]

    # cosine similarity
    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    return sim