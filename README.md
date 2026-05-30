# Word2Vec from Scratch — Skip-Gram with Negative Sampling

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4-013243?style=flat-square&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat-square&logo=jupyter&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.9-4B9CD3?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A complete, ground-up implementation of the **Skip-Gram Word2Vec** model with Negative Sampling (SGNS), trained on the Text8 corpus. The project covers the full pipeline from raw text preprocessing and mini-batch gradient descent training, through intrinsic evaluation on the WordSim-353 benchmark, to a browser-based semantic word game powered by the trained embeddings.

---

## Overview

Word embeddings encode semantic meaning as dense vectors in a continuous space, enabling geometric reasoning over language. This project reimplements the original Mikolov et al. (2013) Skip-Gram architecture entirely in NumPy to expose the underlying mathematics clearly.

The trained embeddings are evaluated both qualitatively (nearest-neighbour queries, vector analogy probing, t-SNE visualisation) and quantitatively (Spearman rank correlation against human similarity judgements on WordSim-353). A standalone browser game, **Word Ascent**, ships the resulting embeddings as a playable semantic guessing experience.

---

## Key Features

- **Pure NumPy training** — no PyTorch or TensorFlow; every forward pass and gradient update is written explicitly, making the mathematics transparent and auditable.
- **Negative Sampling** — efficient approximation of the full softmax objective using a smoothed unigram distribution ($f(w)^{0.75}$), consistent with the original paper.
- **Subsampling of frequent words** — probabilistic discard of high-frequency tokens during data preparation to improve downstream embedding quality.
- **WordNet vocabulary filtering** — post-training, the embedding matrix is reduced to morphological base forms validated against WordNet, ensuring clean downstream evaluation.
- **WordSim-353 benchmark** — Spearman and Pearson correlation against 353 human-annotated word pairs as a rigorous intrinsic evaluation.
- **Vector analogy probing** — structured qualitative tests across gender, profession, and comparative categories.
- **Word Ascent game** — a browser-based semantic guessing game (inspired by Contexto) that serves the trained embeddings via a lightweight local HTTP server.

---

## Project Structure

```
project/
├── artefacts/
│   ├── filtered_embeddings.npz     # Trained embedding matrix (24 270 × 256, float32)
│   ├── filtered_word_idx.json      # word → index mapping (WordNet-filtered vocabulary)
│   └── filtered_idx_word.json      # index → word mapping (WordNet-filtered vocabulary)
├── data/
│   └── combined.csv                # WordSim-353 benchmark dataset (353 word pairs)
├── game/
│   └── word_ascent.py              # Browser-based semantic word game
├── notebooks/
│   ├── 01_skipgram.ipynb           # Data preprocessing, model training, artefact export
│   ├── 02_analysis.ipynb           # Evaluation, analogy probing, WordSim-353 benchmark
│   └── utils.py                    # Shared utility functions
└── requirements.txt                # Full dependency list
```

---

## Technologies Used

| Category | Library / Tool | Purpose |
|---|---|---|
| Numerical computing | NumPy 2.4 | Embedding matrices, vectorised gradient updates, cosine similarity |
| NLP preprocessing | NLTK 3.9 | WordNet lemma lookup, morphological filtering |
| Data manipulation | pandas 3.0 | Loading and handling the WordSim-353 CSV |
| Visualisation | Matplotlib 3.10 | Training loss curves, t-SNE scatter plots |
| Dimensionality reduction | scikit-learn 1.8 | t-SNE projection of the embedding space |
| Statistical evaluation | SciPy 1.17 | Pearson and Spearman correlation computation |
| Progress tracking | tqdm 4.67 | Training progress bars |
| Game server | Python `http.server` (stdlib) | Local HTTP server for the Word Ascent game |
| Notebooks | Jupyter / ipykernel | Interactive development and documentation |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- `pip`

```bash
# Install all dependencies
pip install -r requirements.txt
```

The Text8 corpus is downloaded automatically during notebook execution. WordSim-353 benchmark and some pre-trained artefacts are included in the repository.

---

## Usage

### Training the model

Open and run `notebooks/01_skipgram.ipynb` sequentially if you want the full embeddings with all word forms — note that this takes very long to run. If you're happy with the filtered embeddings that are already included, this step is not necessary. The notebook will:

1. Download and tokenise the Text8 corpus.
2. Build a frequency-filtered vocabulary and generate Skip-gram training pairs.
3. Train the SGNS model for 20 epochs with learning rate decay.
4. Filter the resulting vocabulary to WordNet base forms.
5. Serialise the trained embeddings and vocabulary mappings to `artefacts/`.

### Evaluating the embeddings

Open and run `notebooks/02_analysis.ipynb`. It will use the full embeddings if they were produced in notebook 01, otherwise it defaults to the filtered embeddings already included. The notebook loads the available artefacts and provides:

- t-SNE visualisation of word clusters
- Nearest-neighbour queries using cosine similarity
- Vector analogy probing (`king − man + woman ≈ ?`)
- WordSim-353 Spearman correlation benchmark

### Playing Word Ascent

From the project root, with artefacts present:

```bash
python game/word_ascent.py
```

This starts a local HTTP server on port `8765` and opens the game automatically in your default browser. If the browser does not open, navigate to `http://localhost:8765` manually.

---

## Methodology and Technical Details

### Skip-Gram with Negative Sampling (SGNS)

The model learns to predict surrounding context words from a given centre word. For a centre word $w_c$ and a true context word $w_o$, the objective maximises:

$$\mathcal{L} = \log \sigma\!\left(u_{w_o}^\top v_{w_c}\right) + \sum_{i=1}^{k} \log \sigma\!\left(-u_{w_{n_i}}^\top v_{w_c}\right)$$

where $k = 12$ negative samples are drawn per positive pair from the smoothed unigram distribution $P_n(w) \propto f(w)^{0.75}$.

### Data Preprocessing Pipeline

| Step | Detail |
|---|---|
| Corpus | Text8 — a cleaned Wikipedia extract of ~17 million tokens |
| Minimum frequency | Words appearing fewer than 5 times are discarded |
| Subsampling | Frequent words are probabilistically discarded using the Word2Vec formula |
| Window size | ±5 tokens (symmetric context window) |
| Batch size | 2 048 centre–context pairs |

### Model Configuration

| Hyperparameter | Value |
|---|---|
| Embedding dimension | 256 |
| Epochs | 20 |
| Initial learning rate | 0.0375 |
| Negative samples ($k$) | 12 |
| Learning rate decay | Linear decay to 0.005 |

Gradients are computed analytically. The best-performing epoch (lowest total loss) is retained across training.

### Vocabulary Filtering

After training, the raw vocabulary is reduced to canonical base forms by intersecting with WordNet lemmas and applying the morphological check `wn.morphy(w) == w`. This removes inflected forms, proper nouns, and spurious tokens, yielding a clean 24 270-word vocabulary used for all evaluation and game play.

---

## Results and Findings

### WordSim-353 Intrinsic Evaluation

Cosine similarities are rescaled to the [0, 10] range using a power transform ($\text{sim}^{0.4} \times 10$) before correlation is measured against human annotations.

| Metric | Score |
|---|---|
| Pearson correlation ($r$) | **0.6732** |
| Spearman correlation ($\rho$) | **0.6705** |

### Analogy Probing

Vector arithmetic of the form $\vec{a} - \vec{b} + \vec{c}$ was evaluated across three relational categories:

| Category | Analogy | Top-1 Result | Correct? |
|---|---|---|---|
| Gender | queen − woman + man | ii | ❌ (king at rank 2) |
| Gender | actor − man + woman | actress | ✅ |
| Gender | uncle − man + woman | aunt | ✅ |
| Gender | son − man + woman | daughter | ✅ |
| Profession | scientist − science + politics | politician | ✅ |
| Profession | lawyer − law + medicine | physician | ✅\* |
| Profession | painter − painting + music | composer | ✅\* |
| Profession | soldier − army + navy | gunsmith | ❌ (admiral at rank 2) |
| Comparative | bigger − big + small | smaller | ✅ |
| Comparative | better − good + bad | worse | ✅ |

\*Semantically correct but not the exact expected token.

### Benchmark comparison

Benchmarked against published corpus-based methods on WordSim-353 (Spearman's $\rho$, sourced from the [ACL Wiki leaderboard](https://www.aclweb.org/aclwiki/WordSimilarity-353_Test_Collection_(State_of_the_art))):

| Model | Type | Spearman's $\rho$ |
|---|---|---|
| C&W — Collobert & Weston (2008) | Corpus-based | 0.50 |
| LSA — Landauer et al. (1997) | Corpus-based | 0.58 |
| HSMN+csmRNN — Luong et al. (2013) | Corpus-based | 0.65 |
| **This project (SGNS, Text8, 256-d)** | **Corpus-based** | **0.6705** |
| GloVe — Pennington et al. (2014) | Corpus-based | 0.706 |
| Multi-prototype — Huang et al. (2012) | Corpus-based | 0.71 |
| ESA — Gabrilovich & Markovitch (2007) | Corpus-based | 0.748 |
| ConceptNet Numberbatch — Speer et al. (2017) | Hybrid | 0.828 |

> The result places this from-scratch NumPy implementation ahead of LSA, C&W, and the morphology-aware recursive neural network of Luong et al. (2013), and within ~0.035 of GloVe trained on much larger corpora. The gap relative to GloVe and higher models is expected: those systems use larger training corpora, higher-dimensional embeddings, or hybrid knowledge sources. The score is strong for a single-corpus, pure-NumPy SGNS implementation.
---

## Word Ascent — Semantic Word Game

Word Ascent is a single-page browser game inspired by the game Contexto. Every word in the filtered vocabulary is **ranked** by its cosine similarity to a secret target word. Players receive a rank after each guess and must use that signal to converge on the target within 10 attempts.

**Game mechanics:**

- The vocabulary is ordered by decreasing similarity to the target; rank 1 means the guess *is* the target.
- An opening hint (the word at similarity rank 50) is provided at the start of each round.
- Progressive hints are revealed at steps 3, 6, and 9, each drawing from increasingly close neighbours.
- Only morphological base forms are valid guesses (e.g. `run`, not `running`).
- The UI is served by a stdlib `http.server` (no external web framework required).

The game doubles as a qualitative probe of the embedding space: a well-structured vector space makes the rank signal informative and the game solvable; a poorly trained model produces erratic ranks that feel arbitrary to the player.

---

## Dataset Information

### Text8 Corpus (training)

| Property | Value |
|---|---|
| Source | [mattmahoney.net/dc/text8.zip](http://mattmahoney.net/dc/text8.zip) |
| Content | Cleaned English Wikipedia text |
| Approximate tokens | 17 million |
| Preprocessing | Lowercased, short tokens removed, rare words filtered |

### WordSim-353 (evaluation)

| Property | Value |
|---|---|
| Source | [Gabrilovich & Markovitch (2002)](https://gabrilovich.com/resources/data/wordsim353/wordsim353.html) |
| File | `data/combined.csv` |
| Pairs | 353 |
| Annotation | Human mean similarity score, scale 0–10 |
| Coverage | Scores range from 0.0 (unrelated) to 10.00 (identical) |


---

## Future Improvements

- **Evaluation coverage** — run against additional intrinsic benchmarks such as SimLex-999 or the Google Analogy Test Set for a more comprehensive picture of embedding quality.
- **GloVe comparison** — implement GloVe (global co-occurrence matrix factorisation) alongside SGNS to compare representations on the same corpus and evaluation suite.
- **Subword embeddings** — extend the vocabulary with fastText-style character n-gram representations to handle out-of-vocabulary words, which the current implementation cannot.
- **Contextualised embeddings baseline** — compare the static SGNS vectors against a frozen BERT or similar model on WordSim-353 to quantify the cost of sense conflation.

---

## References

- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS 2013.
- Finkelstein, L., et al. (2002). *Placing Search in Context: The Concept Revisited.* ACM TOIS — WordSim-353 dataset.
- Mahoney, M. *Text8 corpus.* [mattmahoney.net/dc/text8.zip](http://mattmahoney.net/dc/text8.zip)