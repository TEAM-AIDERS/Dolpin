# tests/test_slack_integration.py
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integrations.slack import format_to_slack


def test_format_to_slack():
    """Slack 포맷팅 테스트"""
    
    print("="*60)
    print("🧪 DOLPIN Slack 포맷팅 테스트")
    print("="*60)
    
    # Mock state
    state = {
        "trace_id": "test-123",
        "executive_brief": {
            "summary": "표절 - negative 이슈 (crisis)",
            "severity_score": 8,
            "trend_direction": "escalating",
            "issue_polarity": "negative",
            "spike_summary": "5.0배 급등, organic 타입",
            "sentiment_summary": "disappointment 60%",
            "legal_summary": "법적 검토 필요 (high 리스크)",
            "action_summary": "공식 입장문 발표 (urgent)",
            "opportunity_summary": None,
            "generated_at": "2026-02-07T15:30:00Z",
            "analysis_duration_seconds": 3.5
        },
        "playbook": {
            "priority": "urgent",
            "recommended_actions": [
                {
                    "action": "issue_statement",
                    "urgency": "urgent",
                    "description": "공식 사과문 발표"
                }
            ]
        }
    }
    
    # Format
    print("\n📋 Slack 메시지 생성 중...")
    message = format_to_slack(state)
    
    # 검증
    assert "blocks" in message
    assert "text" in message
    assert len(message["blocks"]) > 0
    
    print("✅ 포맷팅 성공!")
    print(f"\n📊 생성된 메시지 정보:")
    print(f"   - Blocks 개수: {len(message['blocks'])}")
    print(f"   - Fallback text: {message['text']}")
    
    # Block 구조 출력
    print(f"\n📝 Block 구조:")
    print("-"*60)
    for i, block in enumerate(message["blocks"], 1):
        block_type = block.get("type", "unknown")
        
        if block_type == "header":
            text = block["text"]["text"]
            print(f"{i:2d}. [Header] {text}")
        
        elif block_type == "section":
            if "fields" in block:
                print(f"{i:2d}. [Section] 요약 정보 (4개 필드)")
                for field in block["fields"]:
                    # Markdown에서 텍스트만 추출
                    text = field["text"].replace("*", "").replace("\n", " ")
                    print(f"      - {text}")
            
            elif "text" in block:
                text = block["text"]["text"]
                # 첫 줄만 출력
                first_line = text.split("\n")[0]
                print(f"{i:2d}. [Section] {first_line}")
        
        elif block_type == "divider":
            print(f"{i:2d}. [Divider] ───────────────")
        
        elif block_type == "context":
            if "elements" in block and len(block["elements"]) > 0:
                text = block["elements"][0]["text"]
                print(f"{i:2d}. [Context] {text}")
    
    print("\n" + "="*60)
    print("✅ 테스트 완료!")
    print("="*60)
    
    # 전체 JSON 보고 싶으면 주석 해제
    # print("\n📄 전체 JSON:")
    # print(json.dumps(message, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_format_to_slack()