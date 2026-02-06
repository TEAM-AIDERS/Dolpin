from typing import List, TypedDict, Optional


class CausalityItem(TypedDict):
    id: str
    referenced_id: Optional[str]
    created_at: int
    text: str


class CausalityInput(TypedDict):
    issue_id: str
    items: List[CausalityItem]


class CausalityChain(TypedDict):
    root_id: str
    chain: List[str]
    length: int


class CausalityOutput(TypedDict):
    issue_id: str
    chains: List[CausalityChain]

