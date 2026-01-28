from ...smp import *
import numpy as np
import re

FAIL_MSG = 'Failed to obtain answer via API.'

# LVBench question types (from actual data)
QUESTION_TYPES = [
    'entity recognition',
    'event understanding',
    'key information retrieval',
    'temporal grounding',
    'reasoning',
    'summarization'
]

# LVBench video types (from actual data)
VIDEO_TYPES = [
    'selfmedia',
    'cartoon',
    'live',
    'tv',
    'sport',
    'documentary'
]


def get_dimension_rating(data_path):
    """
    Calculate accuracy ratings by different dimensions for LVBench.
    """
    data = load(data_path)
    
    results = {
        'overall': {'correct': 0, 'total': 0, 'accuracy': 0.0},
        'by_question_type': {},
        'by_video_type': {}
    }
    
    # Initialize counters
    for qt in QUESTION_TYPES:
        results['by_question_type'][qt] = {'correct': 0, 'total': 0, 'accuracy': 0.0}
    for vt in VIDEO_TYPES:
        results['by_video_type'][vt] = {'correct': 0, 'total': 0, 'accuracy': 0.0}
    
    for i in range(len(data)):
        score = data.iloc[i].get('score', -1)
        if score < 0:
            continue
        
        # Overall
        results['overall']['total'] += 1
        results['overall']['correct'] += score
        
        # By question type (question_type can be numpy array, list, or string)
        question_types = data.iloc[i].get('question_type', [])
        # Convert numpy array or other types to list
        if hasattr(question_types, 'tolist'):
            question_types = question_types.tolist()
        elif isinstance(question_types, str):
            try:
                question_types = eval(question_types)
            except:
                question_types = [question_types]
        elif not isinstance(question_types, list):
            question_types = [str(question_types)]
        
        for qt in question_types:
            qt_lower = str(qt).lower().strip('[]"\' ')
            for known_qt in QUESTION_TYPES:
                if known_qt == qt_lower:
                    results['by_question_type'][known_qt]['total'] += 1
                    results['by_question_type'][known_qt]['correct'] += score
                    break
        
        # By video type
        video_type = str(data.iloc[i].get('type', '')).lower().strip()
        if video_type in VIDEO_TYPES:
            results['by_video_type'][video_type]['total'] += 1
            results['by_video_type'][video_type]['correct'] += score
    
    # Calculate accuracies
    if results['overall']['total'] > 0:
        results['overall']['accuracy'] = results['overall']['correct'] / results['overall']['total'] * 100
    
    for qt in QUESTION_TYPES:
        if results['by_question_type'][qt]['total'] > 0:
            results['by_question_type'][qt]['accuracy'] = (
                results['by_question_type'][qt]['correct'] / 
                results['by_question_type'][qt]['total'] * 100
            )
    
    for vt in VIDEO_TYPES:
        if results['by_video_type'][vt]['total'] > 0:
            results['by_video_type'][vt]['accuracy'] = (
                results['by_video_type'][vt]['correct'] / 
                results['by_video_type'][vt]['total'] * 100
            )
    
    # Format output
    output = {
        'Overall Accuracy': f"{results['overall']['accuracy']:.2f}%",
        'Total Questions': results['overall']['total'],
        'Correct': results['overall']['correct'],
    }
    
    # Add question type breakdown
    for qt in QUESTION_TYPES:
        if results['by_question_type'][qt]['total'] > 0:
            output[f'QType:{qt}'] = f"{results['by_question_type'][qt]['accuracy']:.2f}%"
    
    # Add video type breakdown
    for vt in VIDEO_TYPES:
        if results['by_video_type'][vt]['total'] > 0:
            output[f'VType:{vt}'] = f"{results['by_video_type'][vt]['accuracy']:.2f}%"
    
    return output


def extract_characters_regex(s):
    """
    Extract the answer choice (A/B/C/D) from a prediction string.
    """
    if not isinstance(s, str):
        s = str(s)
    
    s = s.strip()
    
    # Common answer prefixes to remove
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is',
        'The correct option is',
        'Best answer:',
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')
    
    # If the response is too long and doesn't contain A/B/C/D, return empty
    if len(s.split()) > 10 and not re.search('[ABCD]', s):
        return ''
    
    # Try to find A, B, C, or D
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ''
    return matches[0]

