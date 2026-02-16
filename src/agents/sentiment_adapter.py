class SentimentAgentAdapter:
    def __init__(self, client):
        self.client = client

    def run(self, text: str, analyzed_count: int = 100) -> dict:
        result, meta = self.client.analyze(text, analyzed_count=analyzed_count)

        types = []

        dominant = result.get("dominant_sentiment")
        secondary = result.get("secondary_sentiment")
        dist = result.get("sentiment_distribution", {})

        trigger_counts = meta.get("trigger_counts", {})

        # ===============================
        # 1. Meme 계열 세분화
        # ===============================
        if dominant == "meme":
            neg_score = dist.get("disappointment", 0.0)
            pos_score = dist.get("support", 0.0)

            if neg_score >= pos_score:
                types.append("meme_negative")
            else:
                types.append("meme_positive")

        # ===============================
        # 2. Boycott / Fanwar (행동 트리거 기반 강화)
        # ===============================
        if dominant == "boycott" or trigger_counts.get("boycott", 0) > 0:
            types.append("boycott_action")

        if dominant == "fanwar" or trigger_counts.get("fanwar", 0) > 0:
            types.append("fanwar_action")

        # ===============================
        # 3. Support (순수 지지일 때만)
        # ===============================
        if (
            dominant == "support"
            and trigger_counts.get("boycott", 0) == 0
            and trigger_counts.get("fanwar", 0) == 0
        ):
            types.append("support_action")

        return {
            "sentiment": result,
            "sentiment_meta": meta,
            "types": types,
        }
