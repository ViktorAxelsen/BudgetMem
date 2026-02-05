import regex
import string
import numpy as np
from collections import Counter
from json_repair import repair_json
import json
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def normalize_answer(s):
    s = s.replace(',', "")
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    Compute the token-level F1 score between `prediction` and `ground_truth`.

    Steps:
        1. Return 0.0 immediately if prediction indicates an API request error.
        2. Normalize both strings (lowercase, remove punctuation, trim spaces, etc.).
        3. Tokenize and apply stemming (Porter Stemmer) to improve matching robustness.
        4. Count overlapping tokens using Counter intersection.
        5. If no overlap exists, return 0.
        6. Otherwise compute:
            precision = overlap_count / number_of_prediction_tokens
            recall    = overlap_count / number_of_ground_truth_tokens
            F1        = 2 * precision * recall / (precision + recall)
    """
    if prediction.lower() == "API Request Error".lower():
        return 0.0
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def f1_max(prediction, ground_truth):
    """
    Compute the multi-answer F1 score when both prediction and ground_truth
    may contain multiple comma-separated answers.

    For example:
        prediction     = "A, B, C"
        ground_truth   = "B, D"

    For each ground-truth answer `gt`:
        - Compute F1(pred_i, gt) for all predicted answers pred_i
        - Take the maximum F1 (best match)

    Final score:
        - Average the best-match F1 scores across all ground-truth entries

    In summary:
        score = mean_over_gt( max_over_pred( F1(pred, gt) ) )
    """
    if prediction.lower() == "API Request Error".lower():
        return 0.0
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]

    return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])



def parse_judge_response(response):
    # Check if response is None or empty
    if response is None:
        print("Judge score parse failed: Response is None")
        return 0.0

    response = response.strip()
    if not response:
        print("Judge score parse failed: Response is empty")
        return 0.0

    try:
        return float(json.loads(response)["score"])
    except:
        try:
            fixed = repair_json(response)
            return float(json.loads(fixed)["score"])
        except:
            # Only show first 500 characters to avoid log being too long
            print("Judge score parse failed. Raw response:", response[:500] if len(response) > 500 else response)
            return 0.0


def compute_bleu(prediction, ground_truth):
    if not isinstance(ground_truth, list):
        ground_truths = [ground_truth]
    else:
        ground_truths = ground_truth
    pred_tokens = normalize_answer(str(prediction or "")).split()
    if not pred_tokens:
        return 0.0
    smooth = SmoothingFunction().method1
    scores = []
    for gt in ground_truths:
        ref_tokens = normalize_answer(str(gt or "")).split()
        if not ref_tokens:
            continue
        scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth))
    return max(scores) if scores else 0.0


def compute_rouge_l(prediction, ground_truth):
    if not isinstance(ground_truth, list):
        ground_truths = [ground_truth]
    else:
        ground_truths = ground_truth

    pred_text = str(prediction or "")
    scores = []
    for gt in ground_truths:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        result = scorer.score(prediction=pred_text, target=str(gt or ""))
        scores.append(result["rougeL"].fmeasure)
    return max(scores) if scores else 0.0

if __name__ == '__main__':
    pass

