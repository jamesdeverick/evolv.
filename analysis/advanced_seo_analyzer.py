# --------------------------------------------
# Advanced SEO Features Using Claude Effectively
# analysis/advanced_seo_analyzer.py
# --------------------------------------------

"""
New capabilities to add to your SEO tool that leverage Claude better:
1. Topical authority mapping
2. Content gap analysis (deeper than current)
3. E-E-A-T signal detection
4. Search intent refinement
5. Featured snippet optimization
6. Semantic content coverage scoring
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from api.llm_client import get_llm_client


class TopicalAuthorityAnalyzer:
    """Analyze topical authority and content coverage."""
    
    def __init__(self):
        self.llm = get_llm_client()
    
    def analyze_topical_coverage(
        self,
        main_topic: str,
        existing_content: str,
        competitor_titles: List[str],
        target_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze how well content covers the topic comprehensively.
        
        Returns authority score, missing subtopics, and recommendations.
        """
        if not self.llm.available:
            return {"error": "LLM unavailable"}
        
        prompt = f"""Analyze topical authority and content coverage for SEO.

MAIN TOPIC: {main_topic}

EXISTING CONTENT (first 2000 chars):
{existing_content[:2000] if existing_content else "[No existing content - new page]"}

COMPETITOR ARTICLE TITLES:
{chr(10).join(f"- {title}" for title in competitor_titles[:10])}

TARGET KEYWORDS:
{', '.join(target_keywords[:20])}

Analyze:
1. **Topical Authority Score** (0-100): How comprehensively does/will this content cover the topic?
2. **Missing Subtopics**: What important aspects are competitors covering that we're missing?
3. **Content Depth Issues**: Where is content too shallow or missing detail?
4. **Semantic Coverage Gaps**: What related concepts/entities should be mentioned?
5. **Topical Cluster Opportunities**: What supporting content pieces should be created?

Return ONLY valid JSON:
{{
  "authority_score": 75,
  "missing_subtopics": [
    "subtopic 1 (mentioned by 7/10 competitors)",
    "subtopic 2 (mentioned by 5/10 competitors)"
  ],
  "depth_issues": [
    "Section X needs more detail on Y",
    "Missing practical examples for Z"
  ],
  "semantic_gaps": [
    "Should mention entity/concept A",
    "Missing coverage of related topic B"
  ],
  "cluster_opportunities": [
    {{
      "topic": "Supporting topic 1",
      "intent": "Informational",
      "priority": "High",
      "rationale": "Why this supports main content"
    }}
  ],
  "recommendations": [
    "Specific actionable improvement 1",
    "Specific actionable improvement 2"
  ]
}}"""

        try:
            raw = self.llm.complete(prompt, temperature=0.3, max_tokens=2000)
            
            # Extract JSON
            match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL | re.IGNORECASE)
            json_str = match.group(1) if match else raw[raw.find('{'):raw.rfind('}')+1]
            
            result = json.loads(json_str)
            return result
            
        except Exception as e:
            st.error(f"Topical analysis failed: {e}")
            return {"error": str(e)}


class EEATAnalyzer:
    """Analyze E-E-A-T (Experience, Expertise, Authoritativeness, Trust) signals."""
    
    def __init__(self):
        self.llm = get_llm_client()
    
    def analyze_eeat_signals(
        self,
        topic: str,
        existing_content: str,
        competitor_snippets: List[str]
    ) -> Dict[str, Any]:
        """
        Identify E-E-A-T signals needed for the content.
        Critical for YMYL (Your Money Your Life) topics.
        """
        if not self.llm.available:
            return {"error": "LLM unavailable"}
        
        prompt = f"""Analyze E-E-A-T signals for Google SEO quality.

TOPIC: {topic}

EXISTING CONTENT:
{existing_content[:2000] if existing_content else "[New content]"}

COMPETITOR EXAMPLES:
{chr(10).join(competitor_snippets[:5])}

Identify what E-E-A-T signals this content needs:

1. **Experience Signals**: First-hand experience, case studies, real examples
2. **Expertise Signals**: Author credentials, data/research citations, technical depth
3. **Authoritativeness Signals**: Industry recognition, authoritative sources cited
4. **Trust Signals**: Transparency, author bio, fact-checking, recent updates

Return ONLY JSON:
{{
  "is_ymyl": true,
  "ymyl_category": "Financial/Medical/Legal/etc or None",
  "experience_needed": [
    "Specific example: Real case study of X",
    "First-hand account of Y"
  ],
  "expertise_needed": [
    "Author should have: [credentials]",
    "Should cite: [specific types of sources]",
    "Technical depth needed in: [areas]"
  ],
  "authority_needed": [
    "Should reference: [authoritative sources]",
    "Industry recognition: [how to demonstrate]"
  ],
  "trust_signals": [
    "Add author bio with credentials",
    "Include last updated date",
    "Cite peer-reviewed sources for claims"
  ],
  "competitor_eeat_strengths": [
    "Competitor A: Uses certified expert authors",
    "Competitor B: Cites 10+ authoritative sources"
  ],
  "priority_actions": [
    "Highest priority E-E-A-T improvement 1",
    "High priority improvement 2"
  ]
}}"""

        try:
            raw = self.llm.complete(prompt, temperature=0.3, max_tokens=1500)
            match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL | re.IGNORECASE)
            json_str = match.group(1) if match else raw[raw.find('{'):raw.rfind('}')+1]
            return json.loads(json_str)
        except Exception as e:
            st.error(f"E-E-A-T analysis failed: {e}")
            return {"error": str(e)}


class FeaturedSnippetOptimizer:
    """Optimize content for featured snippets."""
    
    def __init__(self):
        self.llm = get_llm_client()
    
    def identify_snippet_opportunities(
        self,
        target_keyword: str,
        paa_questions: List[str],
        serp_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Identify featured snippet opportunities and provide optimization guidance.
        """
        if not self.llm.available:
            return {"error": "LLM unavailable"}
        
        prompt = f"""Analyze featured snippet opportunities for SEO.

TARGET KEYWORD: {target_keyword}

PEOPLE ALSO ASK QUESTIONS:
{chr(10).join(f"- {q}" for q in paa_questions[:15])}

CURRENT SERP FEATURES:
{json.dumps(serp_features, indent=2)}

Identify:
1. **Snippet Opportunities**: Which queries could win featured snippets?
2. **Snippet Types**: What format (paragraph, list, table, etc.)?
3. **Content Structure**: How to format content to win snippets?
4. **PAA Optimization**: Which PAA questions to target?

Return ONLY JSON:
{{
  "high_opportunity_queries": [
    {{
      "query": "exact question from PAA",
      "snippet_type": "paragraph|list|table|steps",
      "current_snippet_holder": "competitor.com or None",
      "win_probability": "High|Medium|Low",
      "optimization_guide": "Specific formatting instructions"
    }}
  ],
  "content_formatting": {{
    "recommended_sections": [
      {{
        "heading": "H2 or H3 heading",
        "format": "Description of format (e.g., numbered list, comparison table)",
        "content_guideline": "What to include, length, structure"
      }}
    ],
    "paa_answer_format": "How to structure PAA answers (length, format, placement)"
  }},
  "quick_wins": [
    "Easiest snippet to win: [query] with [format]",
    "Add FAQ section for: [questions]"
  ]
}}"""

        try:
            raw = self.llm.complete(prompt, temperature=0.3, max_tokens=1800)
            match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL | re.IGNORECASE)
            json_str = match.group(1) if match else raw[raw.find('{'):raw.rfind('}')+1]
            return json.loads(json_str)
        except Exception as e:
            st.error(f"Snippet analysis failed: {e}")
            return {"error": str(e)}


class SearchIntentRefiner:
    """Deep search intent analysis beyond basic classification."""
    
    def __init__(self):
        self.llm = get_llm_client()
    
    def analyze_user_intent_deeply(
        self,
        keyword: str,
        serp_titles: List[str],
        paa_questions: List[str]
    ) -> Dict[str, Any]:
        """
        Go beyond "informational/commercial" to understand what users REALLY want.
        """
        if not self.llm.available:
            return {"error": "LLM unavailable"}
        
        prompt = f"""Deep search intent analysis for: "{keyword}"

SERP TITLES (what currently ranks):
{chr(10).join(f"{i+1}. {title}" for i, title in enumerate(serp_titles[:10]))}

PAA QUESTIONS:
{chr(10).join(f"- {q}" for q in paa_questions[:10])}

Analyze:
1. **Primary User Need**: What problem are users trying to solve?
2. **User Journey Stage**: Awareness/Consideration/Decision?
3. **User Sophistication**: Beginner/Intermediate/Expert?
4. **Expected Content Type**: Tutorial/Guide/Comparison/Tool/Directory?
5. **Pain Points**: What frustrations/challenges do users have?
6. **Success Metrics**: What does a successful result look like for the user?
7. **Content Tone Match**: What tone/style do users expect?

Return ONLY JSON:
{{
  "primary_user_need": "Specific problem users want solved",
  "user_journey_stage": "Awareness|Consideration|Decision",
  "sophistication_level": "Beginner|Intermediate|Expert|Mixed",
  "expected_content_format": "Long-form guide|Quick tutorial|Comparison table|Tool/Calculator|etc",
  "user_pain_points": [
    "Pain point 1",
    "Pain point 2"
  ],
  "success_criteria": "What user considers successful",
  "content_tone": "Professional|Casual|Technical|Beginner-friendly|etc",
  "must_include_elements": [
    "Element users expect (e.g., examples, pricing, how-to steps)",
    "Another expected element"
  ],
  "content_length_recommendation": "500-1000|1000-2000|2000-3000|3000+ words with rationale",
  "serp_intent_signals": [
    "Signal from SERP: 8/10 results are comparisons",
    "Signal: PAA questions focus on 'how to' not 'what is'"
  ]
}}"""

        try:
            raw = self.llm.complete(prompt, temperature=0.3, max_tokens=1500)
            match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL | re.IGNORECASE)
            json_str = match.group(1) if match else raw[raw.find('{'):raw.rfind('}')+1]
            return json.loads(json_str)
        except Exception as e:
            st.error(f"Intent analysis failed: {e}")
            return {"error": str(e)}


class ContentGapAnalyzer:
    """Advanced content gap analysis."""
    
    def __init__(self):
        self.llm = get_llm_client()
    
    def find_content_gaps(
        self,
        topic: str,
        your_content_outline: str,
        competitor_outlines: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Find what competitors cover that you don't, and vice versa.
        """
        if not self.llm.available:
            return {"error": "LLM unavailable"}
        
        comp_summary = "\n\n".join([
            f"COMPETITOR {i+1} ({c.get('url', 'unknown')}):\n" + 
            "\n".join(f"- {h}" for h in c.get('headings', [])[:15])
            for i, c in enumerate(competitor_outlines[:5])
        ])
        
        prompt = f"""Content gap analysis for: {topic}

YOUR OUTLINE:
{your_content_outline}

COMPETITOR OUTLINES:
{comp_summary}

Identify:
1. **Coverage Gaps**: What do competitors cover that you don't?
2. **Unique Angles**: What do you cover that competitors don't (your advantage)?
3. **Depth Gaps**: Where are competitors more detailed?
4. **Opportunity Gaps**: What should be covered but NO ONE is covering well?

Return ONLY JSON:
{{
  "critical_gaps": [
    {{
      "missing_topic": "Topic/section missing",
      "covered_by": "X out of 5 competitors",
      "priority": "High|Medium|Low",
      "recommended_placement": "Where in outline to add",
      "content_suggestion": "What to cover in this section"
    }}
  ],
  "your_unique_angles": [
    "Unique aspect 1 (not in competitor content)",
    "Unique aspect 2"
  ],
  "depth_improvements": [
    {{
      "section": "Your section name",
      "issue": "Too shallow compared to competitors",
      "competitor_approach": "How competitors handle it better",
      "recommendation": "Specific improvement"
    }}
  ],
  "blue_ocean_opportunities": [
    "Subtopic NO competitor covers well (opportunity to dominate)",
    "Another uncovered angle"
  ],
  "content_quality_score": 75,
  "competitiveness_assessment": "Can compete|Needs more work|Strong position"
}}"""

        try:
            raw = self.llm.complete(prompt, temperature=0.3, max_tokens=2000)
            match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL | re.IGNORECASE)
            json_str = match.group(1) if match else raw[raw.find('{'):raw.rfind('}')+1]
            return json.loads(json_str)
        except Exception as e:
            st.error(f"Gap analysis failed: {e}")
            return {"error": str(e)}


# =============================================================================
# Integration Helper
# =============================================================================

def run_advanced_seo_analysis(
    main_topic: str,
    target_keywords: List[str],
    existing_content: str,
    competitor_data: Dict[str, Any],
    serp_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run all advanced SEO analyses and return comprehensive results.
    
    This is what you'd call from your main app to get deeper insights.
    """
    results = {
        "topical_authority": None,
        "eeat_signals": None,
        "featured_snippets": None,
        "search_intent": None,
        "content_gaps": None
    }
    
    # Topical Authority
    with st.spinner("Analyzing topical authority..."):
        ta_analyzer = TopicalAuthorityAnalyzer()
        results["topical_authority"] = ta_analyzer.analyze_topical_coverage(
            main_topic=main_topic,
            existing_content=existing_content,
            competitor_titles=[c.get('title', '') for c in competitor_data.get('competitors', [])],
            target_keywords=target_keywords
        )
    
    # E-E-A-T
    with st.spinner("Analyzing E-E-A-T signals..."):
        eeat_analyzer = EEATAnalyzer()
        results["eeat_signals"] = eeat_analyzer.analyze_eeat_signals(
            topic=main_topic,
            existing_content=existing_content,
            competitor_snippets=[c.get('snippet', '') for c in competitor_data.get('competitors', [])]
        )
    
    # Featured Snippets
    with st.spinner("Identifying featured snippet opportunities..."):
        snippet_analyzer = FeaturedSnippetOptimizer()
        results["featured_snippets"] = snippet_analyzer.identify_snippet_opportunities(
            target_keyword=main_topic,
            paa_questions=serp_data.get('paa_questions', []),
            serp_features=serp_data.get('features', {})
        )
    
    # Deep Intent
    with st.spinner("Performing deep search intent analysis..."):
        intent_analyzer = SearchIntentRefiner()
        results["search_intent"] = intent_analyzer.analyze_user_intent_deeply(
            keyword=main_topic,
            serp_titles=[c.get('title', '') for c in competitor_data.get('competitors', [])],
            paa_questions=serp_data.get('paa_questions', [])
        )
    
    # Content Gaps
    if competitor_data.get('competitors'):
        with st.spinner("Finding content gaps..."):
            gap_analyzer = ContentGapAnalyzer()
            results["content_gaps"] = gap_analyzer.find_content_gaps(
                topic=main_topic,
                your_content_outline=existing_content[:1000] if existing_content else "[New content]",
                competitor_outlines=competitor_data.get('competitors', [])
            )
    
    return results
