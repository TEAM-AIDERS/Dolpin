class SentimentAgentAdapter:
    def __init__(self, client):
        self.client = client

    def run(self, text: str) -> dict:
        result, meta = self.client.analyze(text)

        types = []

        dominant = result.get("dominant_sentiment")
        dist = result.get("sentiment_distribution", {})

        # 1. meme 계열 합성
        if dominant == "meme":
            neg_score = dist.get("disappointment", 0.0)
            pos_score = dist.get("support", 0.0)

            if neg_score >= pos_score:
                types.append("meme_negative")
            else:
                types.append("meme_positive")

        # 2. action 계열
        if dominant == "boycott":
            types.append("boycott_action")

        if dominant == "fanwar":
            types.append("fanwar_action")

        # 3. 순수 지지 (선택)
        if dominant == "support":
            types.append("support_action")

        return {
            "sentiment": result,
            "sentiment_meta": meta,
            "types": types,
        }
