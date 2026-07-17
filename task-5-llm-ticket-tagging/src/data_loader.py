"""
data_loader.py
--------------
Dataset generation and loading for support ticket classification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


TICKET_TEMPLATES = {
    'Billing': [
        'I was charged twice for my subscription this month. Please fix this urgently.',
        'My credit card was charged $99 but I only subscribed to the basic plan.',
        'Why is there an extra $25 charge on my invoice? I did not authorize this.',
        'My automatic payment failed and now my account shows a past due balance.',
        'I need a copy of my last 3 months invoices for tax purposes.',
        'The pricing on my bill does not match what was advertised on your website.',
        'I was promised a 20% discount but my bill shows full price.',
        'Can you explain the prorated charges on my latest statement?',
        'My bank statement shows multiple small charges from your company.',
        'I need to update my billing address and payment method.'
    ],
    'Technical': [
        'The app keeps crashing every time I try to upload a file larger than 50MB.',
        'I keep getting a 500 Internal Server Error when accessing the dashboard.',
        'The API integration stopped working after the latest update. Error code 403.',
        'My login credentials are correct but I cannot access my account.',
        'The mobile app freezes on the loading screen after the recent update.',
        'Data synchronization between devices is not working properly.',
        'I am unable to export my reports to PDF format. The button does nothing.',
        'The search functionality returns no results even for common queries.',
        'Two-factor authentication codes are not being sent to my phone.',
        'The website loads extremely slowly and images are not displaying.'
    ],
    'Account': [
        'I forgot my password and the reset link is not arriving in my inbox.',
        'I need to change the email address associated with my account.',
        'My account was locked after too many failed login attempts. Please unlock it.',
        'I want to close my account and delete all my personal data permanently.',
        'Can I merge two accounts into one? I have duplicate profiles.',
        'I need to transfer account ownership to my colleague.',
        'My account shows the wrong name and profile picture.',
        'I am unable to add team members to my organization account.',
        'My subscription was downgraded without my consent.',
        'I need to upgrade my plan but the payment page gives an error.'
    ],
    'Refund': [
        'I would like a full refund for my purchase made last week. The product is defective.',
        'Please process a refund for the duplicate charge on my account.',
        'I cancelled within the trial period but was still charged. I need my money back.',
        'The refund I was promised 5 days ago still has not appeared in my account.',
        'I want to return the item I received. It does not match the description.',
        'Can I get a partial refund since I only used the service for 10 days?',
        'My refund request was denied but I believe I am eligible. Please review.',
        'How long does it take for refunds to process? It has been 2 weeks.',
        'I was charged after cancelling my subscription. Need immediate refund.',
        'The product arrived damaged. I need a replacement or full refund.'
    ],
    'Feature Request': [
        'It would be great if you could add dark mode to the dashboard.',
        'Can you implement an auto-save feature? I lost my work twice today.',
        'Please add support for importing CSV files directly from Google Drive.',
        'I would love to see a mobile widget for quick status updates.',
        'Can you add keyboard shortcuts for common actions? It would save a lot of time.',
        'It would be helpful to have real-time collaboration features like Google Docs.',
        'Please consider adding a calendar integration with Outlook and Google Calendar.',
        'I need the ability to schedule recurring reports automatically.',
        'Can you add multi-language support? We have a global team.',
        'It would be useful to have custom notification rules for different projects.'
    ],
    'Complaint': [
        'Your customer service has been terrible. I have been waiting 3 days for a response.',
        'I am extremely disappointed with the quality of support I received.',
        'This is the third time I am reporting the same issue and nothing has been done.',
        'Your product is completely unreliable and I am considering switching to a competitor.',
        'I was promised a callback within 24 hours but it has been a week.',
        'The service downtime this month has been unacceptable for a paid product.',
        'Your team closed my ticket without resolving the actual problem.',
        'I am frustrated that basic features are locked behind expensive paywalls.',
        'The onboarding process was confusing and documentation is outdated.',
        'I expect better communication about scheduled maintenance windows.'
    ],
    'General Inquiry': [
        'What are your business hours for phone support?',
        'Can you tell me more about the enterprise plan pricing?',
        'How do I get started with the API? Is there a tutorial?',
        'What is the difference between the Pro and Business plans?',
        'Do you offer discounts for non-profit organizations?',
        'Where can I find the latest product changelog and release notes?',
        'Is there a community forum where I can ask questions to other users?',
        'What security certifications does your platform have?',
        'Can I schedule a demo with your sales team?',
        'How do I contact the data protection officer for GDPR requests?'
    ]
}


def generate_ticket_dataset() -> pd.DataFrame:
    """
    Generate the synthetic support ticket dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ticket_id, text, true_tag
    """
    tickets = []
    ticket_id = 1
    for tag, texts in TICKET_TEMPLATES.items():
        for text in texts:
            tickets.append({
                'ticket_id': f'TKT-{ticket_id:04d}',
                'text': text,
                'true_tag': tag
            })
            ticket_id += 1

    return pd.DataFrame(tickets)


def get_candidate_labels() -> List[str]:
    """Return the list of candidate classification labels."""
    return list(TICKET_TEMPLATES.keys())


def create_few_shot_examples(df: pd.DataFrame, shots_per_class: int = 2,
                              random_state: int = 42) -> Dict[str, List[str]]:
    """
    Create few-shot examples by sampling from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The ticket dataset.
    shots_per_class : int
        Number of examples per class.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping each tag to a list of example texts.
    """
    examples = {}
    for tag in get_candidate_labels():
        tag_df = df[df['true_tag'] == tag]
        sampled = tag_df.sample(min(shots_per_class, len(tag_df)), 
                                random_state=random_state)
        examples[tag] = sampled['text'].tolist()
    return examples


def create_few_shot_prompt(text: str, examples: Dict[str, List[str]], 
                           candidate_labels: List[str]) -> str:
    """
    Create a few-shot prompt with examples.

    Parameters
    ----------
    text : str
        The ticket text to classify.
    examples : Dict[str, List[str]]
        Few-shot examples per class.
    candidate_labels : List[str]
        List of possible labels.

    Returns
    -------
    str
        The constructed prompt.
    """
    prompt = 'Classify the following support tickets into one of these categories: '
    prompt += ', '.join(candidate_labels) + '

'
    prompt += 'Here are some examples:
'

    for tag, ex_list in examples.items():
        for ex in ex_list:
            prompt += f'
Ticket: {ex}
Category: {tag}
'

    prompt += f'
Now classify this ticket:
Ticket: {text}
Category:'
    return prompt
