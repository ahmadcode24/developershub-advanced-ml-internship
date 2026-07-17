"""
corpus_loader.py
----------------
Knowledge corpus creation and text chunking for RAG pipeline.
"""

import pandas as pd
from typing import List, Dict


KNOWLEDGE_CORPUS = [
    {
        'title': 'Machine Learning Overview',
        'content': 'Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.'
    },
    {
        'title': 'Supervised Learning',
        'content': 'Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. Common algorithms include linear regression, logistic regression, support vector machines, decision trees, and neural networks.'
    },
    {
        'title': 'Deep Learning',
        'content': 'Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, and natural language processing.'
    },
    {
        'title': 'Convolutional Neural Networks',
        'content': 'A Convolutional Neural Network (CNN) is a class of deep neural networks, most commonly applied to analyzing visual imagery. CNNs use a variation of multilayer perceptrons designed to require minimal preprocessing. They are also known as shift invariant or space invariant artificial neural networks.'
    },
    {
        'title': 'Natural Language Processing',
        'content': 'Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.'
    },
    {
        'title': 'Transformers and Attention',
        'content': 'The Transformer is a deep learning model introduced in 2017, used primarily in the field of natural language processing. The attention mechanism allows the model to focus on different parts of the input sequence at each step of the output sequence. The original Transformer model was introduced in the paper Attention Is All You Need by Vaswani et al. in 2017.'
    },
    {
        'title': 'BERT Model',
        'content': 'BERT, which stands for Bidirectional Encoder Representations from Transformers, is a language model based on the transformer architecture. It was created by Google in 2018 and has since become one of the most influential models in NLP. BERT is designed to pre-train deep bidirectional representations from unlabeled text.'
    },
    {
        'title': 'Computer Vision',
        'content': 'Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. Applications include facial recognition, autonomous vehicles, and medical imaging.'
    },
    {
        'title': 'Reinforcement Learning',
        'content': 'Reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Key algorithms include Q-learning, Deep Q-Networks (DQN), and Policy Gradient methods.'
    },
    {
        'title': 'Generative AI',
        'content': 'Generative artificial intelligence (AI) describes algorithms that can be used to create new content, including audio, code, images, text, simulations, and videos. Popular generative models include Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and diffusion models.'
    },
    {
        'title': 'Model Evaluation Metrics',
        'content': 'Model evaluation is the process of using different evaluation metrics to understand a machine learning model's performance. Common classification metrics include accuracy, precision, recall, F1-score, and ROC-AUC. For regression tasks, MAE, MSE, RMSE, and R-squared are commonly used.'
    },
    {
        'title': 'Overfitting and Underfitting',
        'content': 'Overfitting occurs when a statistical model captures the noise of the data. Underfitting occurs when a model cannot capture the underlying trend of the data. Techniques like cross-validation, regularization, and early stopping help prevent these issues.'
    },
    {
        'title': 'Feature Engineering',
        'content': 'Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. Feature selection is the process of selecting a subset of relevant features for use in model construction.'
    },
    {
        'title': 'Unsupervised Learning',
        'content': 'Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses. The most common unsupervised learning method is cluster analysis, which is used for exploratory data analysis.'
    },
    {
        'title': 'Recurrent Neural Networks',
        'content': 'A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. Long Short-Term Memory (LSTM) networks are a special kind of RNN, capable of learning long-term dependencies.'
    }
]


def load_corpus() -> pd.DataFrame:
    """Load the knowledge corpus as a DataFrame."""
    return pd.DataFrame(KNOWLEDGE_CORPUS)


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval granularity.

    Parameters
    ----------
    text : str
        Input text to chunk.
    chunk_size : int
        Maximum words per chunk.
    overlap : int
        Number of overlapping words between chunks.

    Returns
    -------
    List[str]
        List of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
        if start >= len(words):
            break

    return chunks


def create_chunks(corpus_df: pd.DataFrame, chunk_size: int = 80, overlap: int = 15) -> tuple:
    """
    Create chunks from the corpus and return chunks with metadata.

    Returns
    -------
    tuple
        (all_chunks, chunk_metadata)
    """
    all_chunks = []
    chunk_metadata = []

    for idx, row in corpus_df.iterrows():
        chunks = chunk_text(row['content'], chunk_size=chunk_size, overlap=overlap)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                'doc_title': row['title'],
                'chunk_index': chunk_idx,
                'chunk_text': chunk
            })

    return all_chunks, chunk_metadata
