"""
SERP format analysis: what KIND of page is Google rewarding for this query?

Pure Python (no LLM, no extra API calls). Classifies the titles of the top
organic results into page formats ("what is", how-to, comparison, list,
guide, product/solution) and turns the dominant format into explicit
instructions for the title, H1 and outline.
"""
import re
from collections import Counter
from typing import Dict, List, Any

FORMATS = [
    ("what_is", r"^\s*what(?:'s| is| are)\b|\bdefinition\b|\bexplained\b|\bwhat is\b|\bwhat are\b"),
    ("how_to", r"^\s*how (?:to|do|does|can|should)\b|\bstep[- ]by[- ]step\b|\btutorial\b|\bhow to\b"),
    ("comparison", r"\bvs\.?\b|\bversus\b|\balternatives?\b|\bcompar(?:e|ison|ing)\b"),
    ("list", r"^\s*\d+\s|^\s*(?:top|best)\b|\bbest\b|\btop \d+\b"),
    ("guide", r"\b(?:complete|ultimate|beginner'?s?|definitive)?\s*guide\b|\bhandbook\b|\bcheat ?sheet\b"),
]

LABELS = {
    "what_is": "'What is…' explainer pages",
    "how_to": "'How to…' instructional pages",
    "comparison": "comparison / alternatives pages",
    "list": "'best / top N' list pages",
    "guide": "long-form guides",
    "product": "product / solution / vendor pages",
}

RULES = {
    "what_is": ("Open the title and H1 with a 'What is {kw}?' style question or definition. "
                "The first H2 must be a plain-English definition. Then cover how it works, "
                "types/components, benefits, use cases and FAQs."),
    "how_to": ("Make the title and H1 a 'How to…' promise. Structure the outline as ordered steps, "
               "with prerequisites first and troubleshooting last."),
    "comparison": ("Frame the title and H1 as a comparison/alternatives page. Include a "
                   "side-by-side comparison section and clear selection criteria."),
    "list": ("Frame the title and H1 as a ranked or numbered list ('Best…', 'Top N…'). "
             "Each H2 should be one item with a consistent mini-structure."),
    "guide": ("Frame the title and H1 as a comprehensive guide. Cover the full topic "
              "from fundamentals to implementation."),
    "product": ("The SERP is dominated by vendor/product pages: a commercial title and H1 are "
                "appropriate. Lead with the solution and proof points."),
}


def classify_title(title: str) -> str:
    t = (title or "").lower()
    for name, pattern in FORMATS:
        if re.search(pattern, t):
            return name
    return "product"


def analyze_serp_patterns(titles: List[str], keyword: str = "") -> Dict[str, Any]:
    """Return dominant format, counts and ready-to-paste prompt guidance."""
    titles = [t for t in (titles or []) if t and t.strip()]
    if not titles:
        return {}
    labelled = [(t, classify_title(t)) for t in titles]
    counts = Counter(f for _, f in labelled)
    dominant, n = counts.most_common(1)[0]
    share = n / len(titles)
    result: Dict[str, Any] = {
        "counts": dict(counts),
        "total": len(titles),
        "dominant": dominant if share >= 0.4 else "mixed",
        "dominant_share": round(share, 2),
        "examples": [t for t, f in labelled if f == dominant][:4],
    }
    summary = ", ".join(f"{c} {LABELS[f]}" for f, c in counts.most_common())
    if result["dominant"] != "mixed":
        rule = RULES[dominant].replace("{kw}", keyword or "the topic")
        result["guidance"] = (
            f"SERP FORMAT ANALYSIS for \"{keyword}\": of the top {len(titles)} results: {summary}.\n"
            f"Google is clearly rewarding {LABELS[dominant]} ({n}/{len(titles)}). "
            f"Example ranking titles: " + " | ".join(result["examples"]) + "\n"
            f"RULE: {rule}\n"
            f"The proposed Title, H1 and first H2 MUST match this dominant format. "
            f"Do not use a different page format unless the audit explicitly requires it."
        )
    else:
        result["guidance"] = (
            f"SERP FORMAT ANALYSIS for \"{keyword}\": the top {len(titles)} results are mixed: {summary}. "
            f"No single format dominates, so choose the format that best fits the search intent and "
            f"the audit, and consider covering the top two formats' elements."
        )
    return result
