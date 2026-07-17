"""
classifier.py
-------------
Zero-shot and few-shot classification utilities using Hugging Face transformers.
"""

import pandas as pd
from transformers import pipeline
from typing import Dict, List, Tuple


def load_zero_shot_classifier(model_name: str = 'facebook/bart-large-mnli',
                               device: int = -1):
    """
    Load a zero-shot classification pipeline.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.
    device : int
        -1 for CPU, 0 for GPU.

    Returns
    -------
    pipeline
        Hugging Face zero-shot classification pipeline.
    """
    return pipeline(
        'zero-shot-classification',
        model=model_name,
        device=device
    )


def classify_ticket(classifier, text: str, candidate_labels: List[str],
                    top_k: int = 3) -> Dict:
    """
    Classify a single ticket and return top-k predictions.

    Parameters
    ----------
    classifier : pipeline
        Loaded classification pipeline.
    text : str
        Ticket text to classify.
    candidate_labels : List[str]
        List of possible labels.
    top_k : int
        Number of top predictions to return.

    Returns
    -------
    Dict
        Dictionary with top_k labels and scores.
    """
    result = classifier(text, candidate_labels, multi_label=False)
    return {
        'labels': result['labels'][:top_k],
        'scores': result['scores'][:top_k]
    }


def batch_classify(classifier, df: pd.DataFrame, candidate_labels: List[str],
                   text_column: str = 'text', top_k: int = 3) -> List[Dict]:
    """
    Classify multiple tickets in batch.

    Parameters
    ----------
    classifier : pipeline
        Loaded classification pipeline.
    df : pd.DataFrame
        DataFrame containing ticket texts.
    candidate_labels : List[str]
        List of possible labels.
    text_column : str
        Column name containing text.
    top_k : int
        Number of top predictions to return.

    Returns
    -------
    List[Dict]
        List of classification results.
    """
    results = []
    for _, row in df.iterrows():
        result = classify_ticket(classifier, row[text_column], candidate_labels, top_k)
        results.append({
            'ticket_id': row.get('ticket_id', ''),
            'text': row[text_column],
            'true_tag': row.get('true_tag', ''),
            'pred_tag': result['labels'][0],
            'pred_score': result['scores'][0],
            'top_3_labels': result['labels'],
            'top_3_scores': result['scores']
        })
    return results
