# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


INTERNAL_LABELS = [
    "positive",   # support
    "negative",   # disappointment
    "neutral",
    "meme",
    "boycott",
    "fanwar",
]

LABELS = INTERNAL_LABELS
LABEL_SET = set(LABELS)

OUTPUT_LABEL_MAP = {
    "positive": "support",
    "negative": "disappointment",
    "neutral": "neutral",
    "meme": "meme",
    "boycott": "boycott",
    "fanwar": "fanwar",
}

DEFAULT_SENTIMENT_MAP = {
    "Positive": "positive",
    "Negative": "negative",
    "Neutral": "neutral",
    "Meme": "meme",
    "Boycott": "boycott",
    "Fanwar": "fanwar",
    "Context": "neutral",
}

DEFAULT_TRIGGER_MAP = {
    "meme": "meme",
    "boycott": "boycott",
    "fanwar": "fanwar",
    "action": "action",
    "emotion": "emotion",
    "context": "context",
}


@dataclass
class SentimentAnalysisResult:
    dominant_sentiment: str
    secondary_sentiment: Optional[str]
    has_mixed_sentiment: bool
    sentiment_distribution: Dict[str, float]
    confidence: float
    lexicon_matches: List[Dict[str, Any]]
    rationale: List[str]


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.maximum(p, 0)
    s = float(p.sum())
    if s <= 0:
        return np.ones_like(p) / len(p)
    return p / s


def _load_csv_with_fallback(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def _ensure_dir(p: str) -> None:
    d = os.path.dirname(os.path.abspath(p))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


class SentimentAgent:
    def __init__(
        self,
        model,
        tokenizer,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        lexicon_path: str,
        device: str = "cpu",
        meme_weight: float = 0.15,
        fanwar_weight: float = 0.20,
        boycott_weight: float = 0.25,
        mixed_threshold: float = 0.15,
        sentiment_map: Optional[Dict[str, str]] = None,
        trigger_map: Optional[Dict[str, str]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label
        self.device = device

        self.meme_weight = meme_weight
        self.fanwar_weight = fanwar_weight
        self.boycott_weight = boycott_weight
        self.mixed_threshold = mixed_threshold

        self.sentiment_map = sentiment_map or DEFAULT_SENTIMENT_MAP
        self.trigger_map = trigger_map or DEFAULT_TRIGGER_MAP

        lex = _load_csv_with_fallback(lexicon_path)

        rename_map = {
            "term": "term",
            "normalized_form": "normalized_form",
            "type": "type",
            "sentiment_label": "sentiment_label",
            "trigger_type": "trigger_type",
            "action_strength": "action_strength",
            "fandom_scope": "fandom_scope",
            "target_entity": "target_entity",
            "polarity": "polarity",
            "intensity": "intensity",
            "risk_flag": "risk_flag",
            "example_text": "example_text",
            "usage_mode": "usage_mode",
            "notes": "notes",
            "created_at": "created_at",
            "updated_at": "updated_at",
        }
        lex = lex.rename(columns={k: v for k, v in rename_map.items() if k in lex.columns})

        required = {"term", "sentiment_label"}
        if not required.issubset(set(lex.columns)):
            raise ValueError(f"Lexicon CSV must include {required}, got {set(lex.columns)}")

        if "trigger_type" not in lex.columns:
            lex["trigger_type"] = ""

        if "normalized_form" not in lex.columns:
            lex["normalized_form"] = lex["term"]

        self.lexicon = lex

        missing_labels = [l for l in LABELS if l not in self.label2id]
        if missing_labels:
            raise ValueError(f"label2id missing: {missing_labels}")

    def preprocess(self, text: str) -> str:
        text = text.encode("utf-8", "ignore").decode("utf-8")
        text = (text or "").strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(ㅋ)\1{2,}", r"\1\1", text)
        text = re.sub(r"(ㅠ)\1{2,}", r"\1\1", text)
        text = re.sub(r"(ㅎ)\1{2,}", r"\1\1", text)


        return text

    def model_predict(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        probs = probs.squeeze().detach().cpu().numpy()
        
        if probs.ndim == 0:
            probs = np.array([float(probs)])

        if probs.shape[-1] != 6:
            raise ValueError(f"Model output labels must be 6, got {probs.shape[-1]}")

        probs = _normalize_probs(probs)
        return probs

    def _lexicon_match(self, text: str) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
        matches: List[Dict[str, Any]] = []
        sentiment_counts = {k: 0 for k in LABELS}
        trigger_counts = {"meme": 0, "boycott": 0, "fanwar": 0}

        for _, row in self.lexicon.iterrows():
            term = str(row["term"])
            norm = str(row.get("normalized_form", term))
            sraw = str(row.get("sentiment_label", ""))
            raw_type = str(row.get("type", ""))

            # type → 6개 감정 라벨 직접 매핑 (sentiment_label 대신)
            # sentiment_label은 의미적 역할일 뿐, 6개 감정 분류와는 다른 차원
            TYPE_TO_SENTIMENT = {
                # 보이콧 (감정: boycott)
                "boycott_action": "boycott",
                # 팬웨어 (감정: fanwar)
                "fanwar_action": "fanwar",
                "fanwar_target": "fanwar",
                # 밈 (감정: meme)
                "meme_positive": "meme",
                "meme_negative": "meme",
                "meme_slang": "meme",
                # 중립/컨텍스트 (감정: neutral)
                "context_marker": "neutral",
                "fandom_slang": "neutral",
                "irony_cue": "neutral",          # irony는 추후 별도 처리
                "search_evasion": "neutral",     # reference는 중립
                # 지지 (감정: positive)
                "support_action": "positive",
            }

            if not term:
                continue

            if term in text or (norm and norm in text):
                # type으로 직접 감정 매핑 (sentiment_label 무시)
                mapped_sent = TYPE_TO_SENTIMENT.get(raw_type, None)

                if mapped_sent is not None and mapped_sent in LABEL_SET:
                    sentiment_counts[mapped_sent] += 1

                matches.append(
                    {
                        "term": term,
                        "normalized_form": norm,
                        "sentiment_label": sraw,                    # 참고용만
                        "mapped_sentiment": mapped_sent,            # type 기반
                        "type": raw_type,                           # 원본 type
                        "polarity": row.get("polarity", None),
                        "intensity": row.get("intensity", None),
                        "risk_flag": row.get("risk_flag", None),
                    }
                )
        
        # trigger_counts 계산 (matches에서 감정별로 집계)
        # router_second_stage에서 사용하는 boycott/fanwar/meme 신호
        trigger_counts = {"meme": 0, "boycott": 0, "fanwar": 0}
        for match in matches:
            sent = match.get("mapped_sentiment")
            if sent in trigger_counts:
                trigger_counts[sent] += 1

        return matches, sentiment_counts, trigger_counts

    def lexicon_adjust(self, probs: np.ndarray, trigger_counts: Dict[str, int]) -> np.ndarray:
        p = probs.copy()

        if trigger_counts.get("meme", 0) > 0:
            p[self.label2id["meme"]] += self.meme_weight

        if trigger_counts.get("fanwar", 0) > 0:
            p[self.label2id["fanwar"]] += self.fanwar_weight

        if trigger_counts.get("boycott", 0) > 0:
            p[self.label2id["boycott"]] += self.boycott_weight

        return _normalize_probs(p)

    def router_second_stage(
        self,
        text: str,
        probs: np.ndarray,
        matches: List[Dict[str, Any]],
        trigger_counts: Dict[str, int],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        matched_count = len(matches)

        meta = {
            "route": "model_only",
            "trigger_counts": dict(trigger_counts),
            "matched_count": matched_count,
        }
        
        has_any_trigger = any(
            trigger_counts.get(k, 0) > 0
            for k in ["boycott", "fanwar", "meme"]
        )

        # ===============================
        # 0. disappointment semantic boost (NO trigger case)
        # ===============================
        if not has_any_trigger:
            if any(k in text for k in ["실망", "아쉽"]):
                neg_idx = self.label2id["negative"]
                neu_idx = self.label2id["neutral"]
                pos_idx = self.label2id["positive"]

                # negative가 너무 낮으면 최소 보정
                probs = probs.copy()
                probs[neg_idx] = max(float(probs[neg_idx]), 0.35)

                # neutral / positive보다 약간 우위 확보
                probs[neg_idx] = max(
                    probs[neg_idx],
                    probs[neu_idx] + 0.02,
                    probs[pos_idx] + 0.02,
                )

                probs = _normalize_probs(probs)
        
        # ===============================
        # 0.5 implicit disengagement override
        # ===============================
        if any(k in text for k in ["지지 못하겠", "응원 못하겠", "더 이상 응원", "정 떨어졌"]):
            disappoint_idx = self.label2id["negative"]

            forced = np.zeros_like(probs)
            forced[disappoint_idx] = 1.0

            meta["route"] = "implicit_disengagement_override"
            return forced, meta

        # ===============================
        # 1. trigger 없는 경우: 행동 라벨 dominance 차단
        # ===============================
        if not has_any_trigger:
            p = probs.copy()

            boycott_idx = self.label2id["boycott"]
            fanwar_idx  = self.label2id["fanwar"]

            # 행동 라벨 임시로 매우 낮게 설정
            p[boycott_idx] = 0.0
            p[fanwar_idx]  = 0.0

            # 남은 라벨 중 최고값 계산
            max_other = max(
                p[i] for i in range(len(p))
                if i not in (boycott_idx, fanwar_idx)
            )

            # 행동 라벨은 1등보다 항상 낮게
            p[boycott_idx] = 0.0
            p[fanwar_idx]  = 0.0

            p = _normalize_probs(p)
            meta["route"] = "cheer_guard_no_trigger_rank_block"
            return p, meta

        # ===============================
        # 2️. trigger 있는 경우: lexicon 보정
        # ===============================
        adjusted = self.lexicon_adjust(probs, trigger_counts)
        meta["route"] = "model+lexicon"

        if trigger_counts.get("boycott", 0) > 0:
                adjusted[self.label2id["boycott"]] = max(
                    float(adjusted[self.label2id["boycott"]]), 0.60
                )
                adjusted = _normalize_probs(adjusted)
                meta["route"] = "lexicon_override(boycott)"

        if trigger_counts.get("fanwar", 0) > 0:
            adjusted[self.label2id["fanwar"]] = max(
                float(adjusted[self.label2id["fanwar"]]), 0.55
            )
            adjusted = _normalize_probs(adjusted)
            meta["route"] = "lexicon_override(fanwar)"

        if trigger_counts.get("meme", 0) > 0:
            adjusted[self.label2id["meme"]] = max(
                float(adjusted[self.label2id["meme"]]), 0.45
            )
            adjusted = _normalize_probs(adjusted)
            meta["route"] = "lexicon_override(meme)"
        
        # ===============================
        # 3. disappointment soft guard
        # ===============================
        disappoint_idx = self.label2id["negative"]
        boycott_idx    = self.label2id["boycott"]
        fanwar_idx     = self.label2id["fanwar"]

        top_idx  = int(np.argmax(adjusted))
        top_prob = float(adjusted[top_idx])

        if top_idx not in (boycott_idx, fanwar_idx):
            if ("실망" in text or "아쉽" in text):
                if top_prob < 0.35:
                    adjusted[disappoint_idx] = max(
                        float(adjusted[disappoint_idx]), 0.40
                    )
                    adjusted = _normalize_probs(adjusted)
                    meta["route"] = "disappointment_soft_guard"

        return adjusted, meta

    def compute_confidence(self, probs: np.ndarray, analyzed_count: int) -> float:
        model_conf = float(np.max(probs))

        if analyzed_count >= 100:
            factor = 1.0
        elif analyzed_count >= 50:
            factor = 0.9
        else:
            factor = 0.8

        c = model_conf * factor
        if math.isnan(c) or math.isinf(c):
            c = 0.0
        return max(0.0, min(1.0, float(c)))

    def format_output(self, text: str, probs: np.ndarray, matches: List[Dict[str, Any]], trigger_counts: Dict[str, int], confidence: float) -> Dict[str, Any]:
        dist = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}
        dist = dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))

        labels_sorted = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        dominant_label, dominant_p = labels_sorted[0]
        secondary_label, secondary_p = (labels_sorted[1] if len(labels_sorted) > 1 else (None, 0.0))

        # ===============================
        # disappointment semantic boost
        # ===============================
        if any(k in text for k in ["실망", "아쉽"]):
            if dominant_label == "neutral" and dominant_p < 0.35:
                dominant_label = "negative"
                secondary_label = None


        has_mixed = False
        if secondary_label is not None:
            has_mixed = (dominant_p - secondary_p) < self.mixed_threshold

        # ===============================
        # Disappointment should be single-label
        # ===============================
        if dominant_label == "negative":
            has_mixed = False
            secondary_label = None

        # 1. dominant가 행동 라벨인데 trigger 없으면 mixed 아님
        if dominant_label in ["boycott", "fanwar"] and len(matches) == 0:
            has_mixed = False

        # 2. secondary가 행동 라벨인데 trigger 없으면 mixed 아님
        if secondary_label in ["boycott", "fanwar"] and len(matches) == 0:
            has_mixed = False

        # ===============================
        # FINAL OVERRIDE: Cheer is always single-label when no action trigger
        # ===============================
        if dominant_label == "support" and (
            trigger_counts.get("meme", 0) == 0
            and trigger_counts.get("boycott", 0) == 0
            and trigger_counts.get("fanwar", 0) == 0
        ):
            has_mixed = False
            secondary_label = None

        secondary_sentiment = secondary_label if has_mixed else None

        lexicon_matches = []
        rationale = []
        for m in matches:
            term = m.get("term")
            if term:
                rationale.append(str(term))

            lexicon_matches.append({
                "term": m.get("term"),
                "normalized_form": m.get("normalized_form"),
                "type": m.get("type"),
                "sentiment_label": m.get("sentiment_label"),
                "trigger_type": m.get("trigger_type"),
                # "action_strength": m.get("action_strength"),
                # "fandom_scope": m.get("fandom_scope"),
                # "target_entity": m.get("target_entity"),
                "polarity": m.get("polarity"),
                "intensity": m.get("intensity"),
                "risk_flag": m.get("risk_flag"),
                # "usage_mode": m.get("usage_mode"),
            })

        rationale = list(dict.fromkeys([r for r in rationale if r]))
        if len(rationale) > 10:
            rationale = rationale[:10]

        out = {
            "dominant_sentiment": dominant_label,
            "secondary_sentiment": secondary_sentiment,
            "has_mixed_sentiment": bool(has_mixed),
            "sentiment_distribution": dist,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "lexicon_matches": lexicon_matches,
            "rationale": rationale,
        }

        out["dominant_sentiment"] = OUTPUT_LABEL_MAP.get(out["dominant_sentiment"], out["dominant_sentiment"])
        if out["secondary_sentiment"] is not None:
            out["secondary_sentiment"] = OUTPUT_LABEL_MAP.get(out["secondary_sentiment"], out["secondary_sentiment"])

        out["sentiment_distribution"] = {
            OUTPUT_LABEL_MAP.get(k, k): v for k, v in out["sentiment_distribution"].items()
        }

        return out


    def analyze(self, text: str, analyzed_count: int = 100) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        text = self.preprocess(text)
        probs = self.model_predict(text)
        matches, _, trigger_counts = self._lexicon_match(text)
        probs2, route_meta = self.router_second_stage(text, probs, matches, trigger_counts)
        confidence = self.compute_confidence(probs2, analyzed_count)
        out = self.format_output(text, probs2, matches, trigger_counts, confidence)
        return out, route_meta


def load_model_and_tokenizer(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    cfg = model.config

    label2id_raw = {}
    if getattr(cfg, "label2id", None):
        label2id_raw = dict(cfg.label2id)

    norm = {}
    for k, v in label2id_raw.items():
        kk = str(k).strip()
        if kk in LABEL_SET:
            norm[kk] = int(v)

    if set(norm.keys()) != LABEL_SET:
        norm = {l: i for i, l in enumerate(LABELS)}

    id2label = {v: k for k, v in norm.items()}

    device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, norm, id2label


def build_agent(model_path: str, lexicon_path: str, device: str) -> SentimentAgent:
    model, tokenizer, label2id, id2label = load_model_and_tokenizer(model_path, device)
    return SentimentAgent(
        model=model,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        lexicon_path=lexicon_path,
        device=device,
    )


def mock_inputs() -> List[Dict[str, Any]]:
    return [
        {"name": "cheer", "text": "우리 애들 오늘 무대 미쳤다 진짜 최고야 ㅋㅋ", "analyzed_count": 120},
        {"name": "boycott", "text": "이번 활동은 불매한다. 이제 안 본다", "analyzed_count": 80},
        {"name": "meme+cheer", "text": "덕통사고 이게 말이 돼? 진짜 웃기다 ㅋㅋ", "analyzed_count": 60},
        {"name": "fanwar", "text": "쟤네 팬덤 또 시비 거네 수준 보소", "analyzed_count": 30},
        {"name": "cheer_slang", "text": "미친 무대였다 진짜 개잘함ㅋㅋ", "analyzed_count": 100},
        {"name": "cheer_strong", "text": "와 이건 찢었다 레전드임", "analyzed_count": 100},
        {"name": "sarcasm_meme", "text": "또 시작이네 ㅋㅋㅋ 이번엔 얼마나 잘하나 보자", "analyzed_count": 70},
        {"name": "dry_meme", "text": "아 네네 또 레전드라구요 ㅋㅋ", "analyzed_count": 70},
        {"name": "ambiguous_1", "text": "솔직히 이번 활동 좀 실망스럽다", "analyzed_count": 50},
        {"name": "ambiguous_2", "text": "기대 많이 했는데 아쉽네", "analyzed_count": 50},
        {"name": "implicit_action", "text": "이젠 진짜 지지 못하겠다", "analyzed_count": 40},
        {"name": "implicit_leave", "text": "더 이상 응원은 안 할 듯", "analyzed_count": 40},
        {"name": "fanwar_strong", "text": "쟤네 팬들은 왜 맨날 시비냐", "analyzed_count": 30},
        {"name": "fanwar_mock", "text": "또 저 팬덤이네 ㅋㅋ 수준 보소", "analyzed_count": 30},
    ]


def run_tests(agent: SentimentAgent) -> List[Dict[str, Any]]:
    results = []
    for sample in mock_inputs():
        out, meta = agent.analyze(sample["text"], analyzed_count=sample["analyzed_count"])
        s = out["sentiment_distribution"]
        results.append(
            {
                "name": sample["name"],
                "text": sample["text"],
                "route": meta.get("route"),
                "dominant": out["dominant_sentiment"],
                "secondary": out.get("secondary_sentiment"),
                "mixed": out.get("has_mixed_sentiment"),
                "confidence": out["confidence"],
                "sum_probs": float(sum(s.values())),
                "top3": sorted(s.items(), key=lambda x: x[1], reverse=True)[:3],
                "matched_count": len(out.get("lexicon_matches",[])),
                "trigger_counts": meta.get("trigger_counts"),
                "rationale": out.get("rationale",[]),
            }
        )
    return results


def main():

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--lexicon_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save_json", type=str, default="")
    args = p.parse_args()

    agent = build_agent(
        model_path=args.model_path,
        lexicon_path=args.lexicon_path,
        device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )



    test_results = run_tests(agent)

    for r in test_results:
        print("=" * 80)
        print(r["name"])
        print("route:", r["route"])
        print("dominant:", r["dominant"], "secondary:", r["secondary"], "mixed:", r["mixed"])
        print("confidence:", r["confidence"])
        print("sum_probs:", r["sum_probs"])
        print("top3:", r["top3"])
        print("matched_count:", r["matched_count"])
        print("trigger_counts:", r["trigger_counts"])
        print("rationale:", r["rationale"])
        print("text:", r["text"])

    if args.save_json:
        _ensure_dir(args.save_json)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        print(f"\nsaved: {args.save_json}")


if __name__ == "__main__":
    main()


