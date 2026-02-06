"""
Demo script for testing SentimentAgentAdapter locally.
Not used in Dolphin production pipeline.
"""

from sentiment_agent import build_agent
from sentiment_adapter import SentimentAgentAdapter


if __name__ == "__main__":
    agent = build_agent(
        model_path="./sentiment_ft",
        lexicon_path="lexicon_master_final.csv",
        device="cpu",
    )

    adapter = SentimentAgentAdapter(agent)

    text = "아 네네 또 레전드라구요 ㅋㅋ"
    out = adapter.run(text)

    print(out)
