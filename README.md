# Word2Vec: Word Embedding Implementation & Analysis

**Course Assignment**: MATHS FOUNDATION for COMPUTER SCIENCE (CO5097)  
**Institution**: HCM University of Technology - Vietnam National University (School of Computer Science and Engineering)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AqaSKrozlrNKx-6tEISOHcSbsOGas38S?usp=sharing)

A comprehensive implementation and analysis of Word2Vec models (Skip-Gram and CBOW) using both PyTorch from scratch and Gensim library, with theoretical explanations and practical applications.

## 👥 Contributors

**Group 7 - Word Embedding Project**

### Instructor's section
- **PhD. Nguyen An Khuong**

### Team Members:
- **Ly Minh Trung**
- **Ngo Le Khoa**
- **Bui Minh Hieu**
- **Tran Duy Quang**
- **Nguyen Anh Tai**

---

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Theoretical Background](#theoretical-background)
- [Implementation Details](#implementation-details)
- [Use Cases](#use-cases)
- [Exercises & Solutions](#exercises--solutions)
- [Results](#results)
- [Usage](#usage)
- [References](#references)

## 🎯 Overview

This project provides a complete exploration of word embeddings through the Word2Vec framework, covering:

- **Theoretical foundations** of Skip-Gram and CBOW models
- **Manual implementation** using PyTorch neural networks
- **Production-ready implementation** using Gensim
- **Real-world applications** in e-commerce recommendation systems
- **Detailed solutions** to computational and linguistic challenges

## ✨ Features

### 1. **Manual Implementation (PyTorch)**
- Custom Skip-Gram model with center and context embeddings
- Custom CBOW model with context averaging
- Training loop with cross-entropy loss
- 2D visualization of learned embeddings

### 2. **Gensim Implementation**
- Skip-Gram model for e-commerce product recommendations
- CBOW model for context-based predictions
- Phrase detection for multi-word expressions
- Vector arithmetic and similarity analysis

### 3. **Comprehensive Theory**
- Mathematical derivations of Skip-Gram and CBOW
- Gradient calculations and optimization details
- Comparison with one-hot encoding
- Self-supervised learning explanations

### 4. **Exercise Solutions**
- Computational complexity analysis and optimization strategies
- Multi-word phrase detection and training
- Relationship between dot product and cosine similarity
- Semantic similarity emergence proof

## 🔧 Installation

### Prerequisites

```bash
Python 3.7+
```

### Required Packages

```bash
pip install torch
pip install gensim
pip install matplotlib
pip install numpy
pip install pandas
```

### Quick Install

```bash
pip install torch gensim matplotlib numpy pandas
```

## 📁 Project Structure

```
Group7-WordEmbedding/
│
├── Group7-WordEmbedding-src.ipynb    # Main notebook with all implementations
└── README.md                          # This file
```

## 📖 Theoretical Background

### Word2Vec Overview

Word2Vec maps words to fixed-length vectors that capture semantic and syntactic relationships. It consists of two main architectures:

#### 1. **Skip-Gram Model**

Predicts context words given a center word:

$$P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}$$

- **Input**: Center word (e.g., "loves")
- **Output**: Context words (e.g., "the", "man", "his", "son")
- **Best for**: Rare words, larger datasets

#### 2. **CBOW (Continuous Bag of Words)**

Predicts center word given context words:

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}$$

- **Input**: Context words (e.g., "the", "man", "his", "son")
- **Output**: Center word (e.g., "loves")
- **Best for**: Small datasets, frequent words

### Why Not One-Hot Encoding?

One-hot vectors have two major issues:

1. **No similarity information**: Cosine similarity between any two one-hot vectors is 0
2. **High dimensionality**: Vocabulary size = vector dimension (inefficient)

Word2Vec solves this by learning dense, low-dimensional representations where semantically similar words have similar vectors.

## 🛠️ Implementation Details

### PyTorch Manual Implementation

```python
class SkipGram(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(SkipGram, self).__init__()
        self.center_embeddings = nn.Linear(vocab_size, emb_dim, bias=False)
        self.context_embeddings = nn.Linear(emb_dim, vocab_size, bias=False)
```

**Key components:**
- Two embedding matrices: center words and context words
- Cross-entropy loss optimization
- 1000 epochs training on simple corpus

### Gensim Implementation

```python
model = Word2Vec(
    sentences=transactions,
    vector_size=10,
    window=3,
    min_count=1,
    workers=4,
    sg=1  # 1 for Skip-Gram, 0 for CBOW
)
```

**Features:**
- Automatic negative sampling
- Efficient C implementation
- Built-in phrase detection
- Similarity queries and analogies

## 💼 Use Cases

### E-Commerce Product Recommendation

Using transaction history as "sentences" and products as "words":

```python
transactions = [
    ['iPhone', 'Ốp_lưng', 'Sạc_nhanh', 'Tai_nghe_AirPods'],
    ['Samsung_S24', 'Ốp_lưng', 'Sạc_nhanh', 'Tai_nghe_GalaxyBuds'],
    ['Laptop_Dell', 'Chuột_Logitech', 'Bàn_phím_Cơ', 'Lót_chuột']
]
```

**Applications:**
1. **Similar Product Discovery**: Find products frequently bought together
2. **Product Substitution**: Recommend alternatives (iPhone ↔ Samsung_S24)
3. **Ecosystem Detection**: Identify brand ecosystems (Apple products cluster together)
4. **Cross-sell Recommendations**: Suggest complementary products

## 📝 Exercises & Solutions

### Exercise 1: Computational Complexity

**Question**: What is the computational complexity for calculating each gradient?

**Answer**: 
- **Complexity**: O(|V| × d) per gradient
  - |V| = vocabulary size
  - d = embedding dimension
- **Problem**: Expensive for large vocabularies (100K+ words)
- **Solutions**:
  - **Negative Sampling**: Reduces to O(K) where K ≈ 5-20
  - **Hierarchical Softmax**: Reduces to O(log |V|)

### Exercise 2: Multi-Word Phrases

**Question**: How to train word vectors for phrases like "New York"?

**Answer**: Statistical phrase detection using co-occurrence scoring:

$$\text{score}(w_i, w_j) = \frac{\text{count}(w_i w_j) - \delta}{\text{count}(w_i) \times \text{count}(w_j)}$$

**Implementation**:
```python
from gensim.models.phrases import Phrases

bigram_transformer = Phrases(sentences, min_count=5, threshold=10)
trigram_transformer = Phrases(bigram_transformer[sentences])
```

### Exercise 3: Dot Product & Cosine Similarity

**Question**: Why do semantically similar words have high cosine similarity?

**Answer**: 

**Mathematical relationship**:
$$\text{cosine-similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

**Why similar words cluster**:
1. Words with similar contexts must maximize dot products with the **same** context words
2. This forces their vectors to align in the same direction
3. High directional alignment = high cosine similarity

**Example**: "king" and "queen" both appear with {crown, throne, royal} → their vectors must align with these context vectors → vectors become similar

## 📊 Results

### Embedding Visualizations

The project includes 2D visualizations showing:

- **Gender relationships**: man-woman, king-queen, boy-girl
- **Hierarchical relationships**: king-prince, queen-princess
- **Vector arithmetic**: king - man + woman ≈ queen

### Performance Metrics

**Manual PyTorch Implementation**:
- Training corpus: 8 sentences (artificially repeated)
- Embedding dimension: 2D (for visualization)
- Final loss: Converges after ~800 epochs

**Gensim Implementation**:
- Transaction data: 10 patterns × 100 repetitions
- Embedding dimension: 10D
- Similarity examples:
  - iPhone ↔ Sạc_nhanh: High (frequently bought together)
  - iPhone ↔ Samsung_S24: Medium (substitutes)
  - iPhone ↔ Bàn_phím_Cơ: Low (unrelated categories)

## 🚀 Usage

### 1. Run the Complete Notebook

```bash
jupyter notebook Group7-WordEmbedding-src.ipynb
```

### 2. Train Your Own Model

```python
from gensim.models import Word2Vec

# Prepare your corpus (list of tokenized sentences)
sentences = [
    ['word1', 'word2', 'word3'],
    ['word2', 'word4', 'word5']
]

# Train model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Find similar words
similar = model.wv.most_similar('word1', topn=5)

# Vector arithmetic
result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
```

### 3. Visualize Embeddings

```python
import matplotlib.pyplot as plt

# Get embeddings
words = list(model.wv.key_to_index.keys())
vectors = [model.wv[word] for word in words]

# Plot (if 2D)
for i, word in enumerate(words):
    x, y = vectors[i]
    plt.scatter(x, y)
    plt.annotate(word, (x, y))
plt.show()
```

## 📚 References

### Academic Papers

1. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). **Distributed Representations of Words and Phrases and their Compositionality**. NIPS 2013.
2. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). **Efficient Estimation of Word Representations in Vector Space**. ICLR 2013.

### Learning Resources

- [D2L.ai - Word Embedding (word2vec)](https://d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html)
- [Gensim Word2Vec Documentation](https://radimrehurek.com/gensim/models/word2vec.html)
- [Original Word2Vec Google Code](https://code.google.com/archive/p/word2vec/)

### Libraries Used

- **PyTorch**: Neural network framework for manual implementation
- **Gensim**: Production-ready NLP library with optimized Word2Vec
- **Matplotlib**: Visualization and plotting
- **NumPy**: Numerical computations

## 📄 License

This project is created for educational purposes as part of a Master's course in Computer Science Mathematics.

## 🙏 Acknowledgments

- D2L.ai team for excellent theoretical explanations
- Gensim developers for the robust implementation
- Course instructors for guidance and feedback

---

## 🔗 Quick Links

- **Main Notebook**: `Group7-WordEmbedding-src.ipynb`
- **Theory Section**: Cells 1-26 (Mathematical foundations)
- **PyTorch Implementation**: Cells 27-32 (Manual skip-gram/CBOW)
- **Gensim Implementation**: Cells 33-52 (Production use cases)
- **Exercise Solutions**: Cells 53-63 (Detailed answers with proofs)

---

**Last Updated**: December 2025

For questions or contributions, please open an issue or contact the project maintainers.
