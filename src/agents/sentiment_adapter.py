class SentimentAgentAdapter:
    def __init__(self, client):
        self.client = client  # local import, API, or callable

    def run(self, text: str) -> dict:
        """
        Dolphin에서 사용하는 표준 형태로 SentimentAgent 결과를 변환
        """
        result, meta = self.client.analyze(text)

        return {
            "sentiment": result,
            "sentiment_meta": meta,
        }
