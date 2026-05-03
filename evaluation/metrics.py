import re
from collections import Counter


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(ngrams(ref_tokens, n))
        hyp_ngrams = Counter(ngrams(hyp_tokens, n))

        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(p for p in precisions) / len(precisions)

    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = len(hyp_tokens) / len(ref_tokens)

    import math
    log_precisions = [math.log(p) for p in precisions]
    geo_mean = math.exp(sum(log_precisions) / len(log_precisions))

    return bp * geo_mean


def rouge_l_score(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    m = len(ref_tokens)
    n = len(hyp_tokens)
    lcs = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])

    lcs_len = lcs[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / n
    recall = lcs_len / m
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def normalize_docstring(docstring: str) -> str:
    text = docstring.strip()
    text = re.sub(r'"""', '', text)
    text = re.sub(r"'''", '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def exact_match(reference: str, hypothesis: str) -> bool:
    return normalize_docstring(reference) == normalize_docstring(hypothesis)


def compute_metrics(reference: str, hypothesis: str) -> dict:
    ref_norm = normalize_docstring(reference)
    hyp_norm = normalize_docstring(hypothesis)

    return {
        "bleu": bleu_score(ref_norm, hyp_norm),
        "rouge_l": rouge_l_score(ref_norm, hyp_norm),
        "exact_match": exact_match(ref_norm, hyp_norm),
        "ref_length": len(ref_norm.split()),
        "hyp_length": len(hyp_norm.split()),
    }
