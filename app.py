import os
import re
import string
import pickle
import difflib
from pathlib import Path
from typing import List, Dict

import torch
import warnings
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
import torch.nn.functional as F

# ------------------ CONFIG ------------------
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()  # hide transformers noisy warnings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths / names
LaBSE_PICKLE = "LaBSE.pkl"  # where LaBSE will be saved/loaded
NLI_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
NLI_CACHE_DIR = "xlm_roberta_xnli_cache"  # transformers cache dir for NLI

# NLI thresholds
NLI_CONTRADICTION_THRESHOLD = 0.65  # per-answer contradiction probability considered "contradiction"
NLI_CONTRADICTION_PROP = 0.5  # proportion of accepted answers contradicted -> reject
NLI_ENTAILMENT_PROP = 0.5  # proportion entailment -> accept
NLI_ENTAILMENT_AVG_PROB = 0.70  # avg entailment prob threshold to force accept

# similarity / scores
SIMILARITY_ACCEPT_THRESHOLD = 0.66
KEYWORD_PENALTY = 0.25  # milder penalty
NEGATION_PENALTY = 0.45

# ------------------ UTIL: text normalization & fuzzy keyword match ------------------
PUNCT_TABLE = str.maketrans({p: " " for p in (string.punctuation + "،؟؛«»“”…—ـ«»ٔ؟")})


def normalize_text(text: str) -> str:
    if not text:
        return ""
    txt = text.strip().lower()
    txt = txt.translate(PUNCT_TABLE)
    txt = re.sub(r"\s+", " ", txt)  # collapse spaces
    return txt.strip()


def fuzzy_keyword_match(student_text: str, keyword: str, ratio_threshold: float = 0.70) -> bool:
    s = normalize_text(student_text)
    k = normalize_text(keyword)
    if not k:
        return False
    # direct substring
    if k in s:
        return True
    # fuzzy ratio
    r = difflib.SequenceMatcher(None, s, k).ratio()
    if r >= ratio_threshold:
        return True
    # also check token-level partial matches (in case keyword shorter)
    tokens = s.split()
    for t in tokens:
        if difflib.SequenceMatcher(None, t, k).ratio() >= ratio_threshold:
            return True
    return False


# ------------------ NEGATION WORDS ------------------
NEGATION_WORDS = [
    # english
    "not", "never", "no", "none", "n't", "cannot", "can't", "don't", "doesn't",
    "didn't", "won't", "wouldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
    # persian (common)
    "نیست", "نیستم", "نیستی", "نیستیم", "نیستند",
    "نبود", "نبوده", "نبودیم", "نبودند",
    "نشد", "نشده", "نشدنی", "نمیشه",
    "نمی", "نمید", "نمیک", "نمیخ",
    "نخواهد", "نخواهم", "نخواهند",
    "نه", "ندارد", "نداریم", "ندانست", "نکرده",
    "نا", "بی", "بدترین"
                "افتضحاح", "بد", "بدتر"

]


def contains_negation(text: str) -> bool:
    if not text:
        return False
    s = " " + normalize_text(text) + " "
    for w in NEGATION_WORDS:
        if f" {w} " in s or s.startswith(f"{w} "):
            return True
    return False


# ------------------ LOAD / CACHE MODELS ------------------
# LaBSE
if Path(LaBSE_PICKLE).exists():
    with open(LaBSE_PICKLE, "rb") as f:
        sbert_model = pickle.load(f)
    print(f"✅ Loaded LaBSE from {LaBSE_PICKLE}")
else:
    sbert_model = SentenceTransformer("sentence-transformers/LaBSE")
    with open(LaBSE_PICKLE, "wb") as f:
        pickle.dump(sbert_model, f)
        print(f"✅ LaBSE Loaded {LaBSE_PICKLE}")

# NLI
os.makedirs(NLI_CACHE_DIR, exist_ok=True)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, cache_dir=NLI_CACHE_DIR)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME, cache_dir=NLI_CACHE_DIR)
nli_model.to(DEVICE)
nli_model.eval()
# label mapping (robust)
label2id = {v.lower(): int(k) for k, v in nli_model.config.id2label.items()}

# ------------------ CACHE for embeddings (optional speedup) ------------------
_EMBED_CACHE: Dict[str, torch.Tensor] = {}


def cached_encode(sentences: List[str]):
    """
    Encode with caching to speed up repeated texts.
    Returns tensor on DEVICE.
    """
    # we will encode new sentences, but use cache per exact normalized text
    results = []
    to_encode = []
    idxs = []
    for i, s in enumerate(sentences):
        key = normalize_text(s)
        if key in _EMBED_CACHE:
            results.append(_EMBED_CACHE[key])
        else:
            results.append(None)
            to_encode.append(s)
            idxs.append(i)
    if to_encode:
        emb = sbert_model.encode(to_encode, convert_to_tensor=True, device=DEVICE)
        j = 0
        for i in idxs:
            results[i] = emb[j]
            _EMBED_CACHE[normalize_text(to_encode[j])] = emb[j]
            j += 1
    # ensure all tensors on DEVICE
    results = [r.to(DEVICE) if isinstance(r, torch.Tensor) else torch.tensor([], device=DEVICE) for r in results]
    return results


# ------------------ NLI helpers ------------------
def nli_summary(student: str, accepted_answers: List[str]):
    """
    Returns dict with counts and average probs for entailment & contradiction.
    """
    if not accepted_answers:
        return {"entail_count": 0, "contra_count": 0, "entail_avg_prob": 0.0, "contra_avg_prob": 0.0, "n": 0}

    entail_count = 0
    contra_count = 0
    entail_probs = []
    contra_probs = []
    with torch.no_grad():
        for ans in accepted_answers:
            enc = nli_tokenizer(student, ans, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            out = nli_model(**enc)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
            # get label indices robustly
            # some models map differently, use config.id2label inversion
            # label2id maps e.g. 'contradiction' -> index
            ent_idx = label2id.get("entailment", None)
            contra_idx = label2id.get("contradiction", None)
            if ent_idx is None or contra_idx is None:
                # fallback to common ordering if map missing
                # try to infer: if 'entailment' not found, assume index 2 is entailment as common in many models
                ent_idx = ent_idx if ent_idx is not None else 2
                contra_idx = contra_idx if contra_idx is not None else 0
            ent_p = float(probs[ent_idx])
            contra_p = float(probs[contra_idx])
            entail_probs.append(ent_p)
            contra_probs.append(contra_p)
            if ent_p >= 0.5:
                entail_count += 1
            if contra_p >= 0.5:
                contra_count += 1
    n = len(accepted_answers)
    return {
        "entail_count": entail_count,
        "contra_count": contra_count,
        "entail_avg_prob": sum(entail_probs) / n,
        "contra_avg_prob": sum(contra_probs) / n,
        "n": n
    }


# ------------------ CORE scoring functions ------------------
def get_sbert_similarity(student_answer: str, accepted_answers: List[str]) -> float:
    if not accepted_answers:
        return 0.0
    texts = [student_answer] + accepted_answers
    embs = cached_encode(texts)  # list of tensors on DEVICE
    student_emb = embs[0].unsqueeze(0)  # shape (1, dim)
    accepted_embs = torch.stack([e for e in embs[1:]], dim=0)  # shape (n, dim)
    # cosine similarity along dim=1 between student_emb repeated and each accepted_emb
    sims = F.cosine_similarity(student_emb.repeat(accepted_embs.size(0), 1), accepted_embs, dim=1)
    # sims are in [-1,1], keep as float in [0,1] approx (Roberta embeddings typically positive)
    sims = sims.cpu().tolist()
    return float(max(sims))


def keywords_present(student_answer: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    for kw in keywords:
        if fuzzy_keyword_match(student_answer, kw, ratio_threshold=0.66):
            return True
    return False


def accepted_answers_are_negative(accepted_answers: List[str]) -> bool:
    if not accepted_answers:
        return False
    neg_count = sum(1 for a in accepted_answers if contains_negation(a))
    return (neg_count / len(accepted_answers)) > 0.5


def compute_final_score(student_answer: str, accepted_answers: List[str], keywords: List[str]) -> float:
    # Normalize inputs quickly
    student_norm = normalize_text(student_answer)
    accepted_norm = [normalize_text(a) for a in accepted_answers]

    # 1) NLI summary first (fast decision if strong contradiction or entailment)
    nli = nli_summary(student_norm, accepted_norm)
    contra_prop = nli["contra_count"] / nli["n"] if nli["n"] > 0 else 0.0
    entail_prop = nli["entail_count"] / nli["n"] if nli["n"] > 0 else 0.0
    # strong contradiction -> reject
    if contra_prop >= NLI_CONTRADICTION_PROP or nli["contra_avg_prob"] >= NLI_CONTRADICTION_THRESHOLD:
        return 0.0

    # compute LaBSE similarity
    sbert_sim = get_sbert_similarity(student_answer, accepted_answers)

    # If NLI indicates majority entailment or avg entail prob high -> boost score
    if entail_prop >= NLI_ENTAILMENT_PROP or nli["entail_avg_prob"] >= NLI_ENTAILMENT_AVG_PROB:
        # accept strongly: keep maximum between sbert_sim and high value
        return max(sbert_sim, 0.9)

    score = sbert_sim

    # Keyword penalty (mild). If keywords provided but none fuzzy-match -> small deduction
    if keywords:
        if not keywords_present(student_answer, keywords):
            score -= KEYWORD_PENALTY

    # Negation handling: if student negative but accepted not negative -> penalty
    student_neg = contains_negation(student_answer)
    answers_neg = accepted_answers_are_negative(accepted_answers)
    if student_neg and not answers_neg and score >= SIMILARITY_ACCEPT_THRESHOLD:
        score -= NEGATION_PENALTY

    return max(0.0, score)


# ------------------ FASTAPI ------------------
app = FastAPI(title="Semantic Answer Checker v5",
              description="LaBSE (LaBSE) + XLM-RoBERTa NLI optimized; fuzzy keywords; negation; local caching.",
              version="5.0")


class CheckRequest(BaseModel):
    student_answer: str
    accepted_answers: List[str]
    keywords: List[str] = []


class CheckResponse(BaseModel):
    similarity_score: float
    accepted: bool
    details: dict = {}


@app.post("/check_answer", response_model=CheckResponse)
def check_answer_endpoint(data: CheckRequest):
    score = compute_final_score(data.student_answer, data.accepted_answers, data.keywords)
    accepted = score >= SIMILARITY_ACCEPT_THRESHOLD
    # produce a helpful details dict (optional)
    # include NLI summary for debugging:
    nli = nli_summary(normalize_text(data.student_answer), [normalize_text(a) for a in data.accepted_answers])
    details = {
        "sbert_similarity": round(get_sbert_similarity(data.student_answer, data.accepted_answers), 4),
        "nli_entail_count": nli["entail_count"],
        "nli_contra_count": nli["contra_count"],
        "nli_entail_avg_prob": round(nli["entail_avg_prob"], 4),
        "nli_contra_avg_prob": round(nli["contra_avg_prob"], 4),
        "keywords_present": keywords_present(data.student_answer, data.keywords) if data.keywords else True,
        "student_negation": contains_negation(data.student_answer),
    }
    return CheckResponse(similarity_score=round(score, 4), accepted=accepted, details=details)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8020, reload=True)
