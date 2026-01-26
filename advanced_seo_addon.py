# --------------------------------------------
# Advanced SEO Add-on - Simple Integration
# advanced_seo_addon.py
# --------------------------------------------

"""
Place this file in the root directory (same level as app.py).
Then add ONE line to your app.py to enable advanced features.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from analysis.advanced_seo_analyzer import (
    TopicalAuthorityAnalyzer,
    EEATAnalyzer,
    FeaturedSnippetOptimizer,
    SearchIntentRefiner,
    ContentGapAnalyzer
)


def show_advanced_analysis_section():
    """
    Call this function from your app.py at the end of Step 2.
    Just add: from advanced_seo_addon import show_advanced_analysis_section
    Then call: show_advanced_analysis_section()
    """
    
    # Only show if we have analyzed keywords
    if st.session_state.get("analyzed_keywords_df") is None:
        return
    
    st.markdown("---")
    st.subheader("🚀 Advanced SEO Analysis (Optional)")
    
    with st.expander("💎 Run Advanced Analysis (5 deep insights)", expanded=False):
        st.info("""
        Get professional-grade insights:
        - **E-E-A-T Requirements**: What trust signals Google expects
        - **Topical Authority Score**: How comprehensive your coverage is
        - **Featured Snippet Opportunities**: Easy wins for top-of-page placement
        - **Deep Search Intent**: What users really want (beyond "informational")
        - **Content Gaps**: Exact topics competitors have that you're missing
        """)
        
        if st.button("🔍 Run Advanced Analysis", type="primary"):
            run_advanced_analysis()
        
        # Display results if they exist
        if st.session_state.get("advanced_analysis"):
            display_advanced_results()


def run_advanced_analysis():
    """Execute all 5 advanced analyses."""
    
    # Gather data
    target_keywords = st.session_state.analyzed_keywords_df["Keyword"].head(20).tolist()
    
    competitor_data = {
        "competitors": st.session_state.competitor_analysis.get("competitors", [])
        if st.session_state.get("competitor_analysis") else []
    }
    
    serp_data = {
        "paa_questions": [],  # We'll get these from related searches
        "features": {}
    }
    
    results = {}
    
    # 1. Topical Authority
    with st.spinner("📊 Analyzing topical authority..."):
        ta_analyzer = TopicalAuthorityAnalyzer()
        competitor_titles = [c.get('title', '') for c in competitor_data.get('competitors', [])]
        
        results["topical_authority"] = ta_analyzer.analyze_topical_coverage(
            main_topic=st.session_state.query_topic,
            existing_content=st.session_state.get("fetched_webpage_content", ""),
            competitor_titles=competitor_titles[:10],
            target_keywords=target_keywords
        )
    
    # 2. E-E-A-T Analysis
    with st.spinner("🏆 Analyzing E-E-A-T requirements..."):
        eeat_analyzer = EEATAnalyzer()
        competitor_snippets = [c.get('snippet', '') for c in competitor_data.get('competitors', [])]
        
        results["eeat_signals"] = eeat_analyzer.analyze_eeat_signals(
            topic=st.session_state.query_topic,
            existing_content=st.session_state.get("fetched_webpage_content", ""),
            competitor_snippets=competitor_snippets[:5]
        )
    
    # 3. Featured Snippet Opportunities
    with st.spinner("⭐ Finding featured snippet opportunities..."):
        snippet_analyzer = FeaturedSnippetOptimizer()
        paa_questions = [k for k in target_keywords if "?" in k or any(k.lower().startswith(q) for q in ["what", "how", "why", "when"])]
        
        results["featured_snippets"] = snippet_analyzer.identify_snippet_opportunities(
            target_keyword=st.session_state.query_topic,
            paa_questions=paa_questions[:15],
            serp_features=serp_data.get("features", {})
        )
    
    # 4. Deep Search Intent
    with st.spinner("🎯 Analyzing deep search intent..."):
        intent_analyzer = SearchIntentRefiner()
        competitor_titles = [c.get('title', '') for c in competitor_data.get('competitors', [])]
        
        results["search_intent"] = intent_analyzer.analyze_user_intent_deeply(
            keyword=st.session_state.query_topic,
            serp_titles=competitor_titles[:10],
            paa_questions=paa_questions[:10]
        )
    
    # 5. Content Gap Analysis
    if competitor_data.get('competitors'):
        with st.spinner("🔍 Finding content gaps..."):
            gap_analyzer = ContentGapAnalyzer()
            
            # Build competitor outlines
            competitor_outlines = []
            for comp in competitor_data.get('competitors', [])[:5]:
                competitor_outlines.append({
                    "url": comp.get("url", ""),
                    "headings": comp.get("headings", [])
                })
            
            results["content_gaps"] = gap_analyzer.find_content_gaps(
                topic=st.session_state.query_topic,
                your_content_outline=st.session_state.get("fetched_webpage_content", "")[:1000],
                competitor_outlines=competitor_outlines
            )
    
    # Save results
    st.session_state.advanced_analysis = results
    st.success("✅ Advanced analysis complete!")
    st.rerun()


def display_advanced_results():
    """Display the advanced analysis results."""
    
    results = st.session_state.advanced_analysis
    
    # Topical Authority
    if results.get("topical_authority") and not results["topical_authority"].get("error"):
        st.markdown("### 📊 Topical Authority")
        ta = results["topical_authority"]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Authority Score", f"{ta.get('authority_score', 0)}/100")
        col2.metric("Missing Topics", len(ta.get('missing_subtopics', [])))
        col3.metric("Opportunities", len(ta.get('cluster_opportunities', [])))
        
        with st.expander("📋 Missing Subtopics (Add These to Your Content)"):
            for topic in ta.get('missing_subtopics', []):
                st.write(f"- {topic}")
        
        with st.expander("🎯 Content Cluster Opportunities"):
            for opp in ta.get('cluster_opportunities', []):
                st.write(f"**{opp.get('topic')}** ({opp.get('priority')} priority)")
                st.write(f"_{opp.get('rationale')}_")
                st.write("")
    
    # E-E-A-T
    if results.get("eeat_signals") and not results["eeat_signals"].get("error"):
        st.markdown("### 🏆 E-E-A-T Requirements")
        eeat = results["eeat_signals"]
        
        if eeat.get("is_ymyl"):
            st.warning(f"⚠️ This is a YMYL ({eeat.get('ymyl_category')}) topic - E-E-A-T is CRITICAL for rankings!")
        
        with st.expander("✅ Required Trust Signals"):
            for signal in eeat.get('trust_signals', []):
                st.write(f"- {signal}")
        
        with st.expander("🎓 Expertise Requirements"):
            for req in eeat.get('expertise_needed', []):
                st.write(f"- {req}")
        
        with st.expander("⚡ Priority Actions"):
            for action in eeat.get('priority_actions', []):
                st.write(f"**{action}**")
    
    # Featured Snippets
    if results.get("featured_snippets") and not results["featured_snippets"].get("error"):
        st.markdown("### ⭐ Featured Snippet Opportunities")
        snippets = results["featured_snippets"]
        
        high_opps = [o for o in snippets.get('high_opportunity_queries', []) if o.get('win_probability') == 'High']
        
        if high_opps:
            st.success(f"Found {len(high_opps)} HIGH probability snippet opportunities!")
        
        for opp in snippets.get('high_opportunity_queries', [])[:5]:
            with st.expander(f"📌 {opp.get('query')} ({opp.get('win_probability')} probability)"):
                st.write(f"**Format:** {opp.get('snippet_type')}")
                st.write(f"**How to win:** {opp.get('optimization_guide')}")
        
        if snippets.get('quick_wins'):
            st.info("**Quick Wins:**\n" + "\n".join(f"- {w}" for w in snippets['quick_wins']))
    
    # Search Intent
    if results.get("search_intent") and not results["search_intent"].get("error"):
        st.markdown("### 🎯 Deep Search Intent Analysis")
        intent = results["search_intent"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**User Need:** {intent.get('primary_user_need')}")
            st.write(f"**Journey Stage:** {intent.get('user_journey_stage')}")
            st.write(f"**User Level:** {intent.get('sophistication_level')}")
        with col2:
            st.write(f"**Expected Format:** {intent.get('expected_content_format')}")
            st.write(f"**Tone:** {intent.get('content_tone')}")
            st.write(f"**Recommended Length:** {intent.get('content_length_recommendation')}")
        
        with st.expander("😰 User Pain Points"):
            for pain in intent.get('user_pain_points', []):
                st.write(f"- {pain}")
        
        with st.expander("✅ Must Include in Content"):
            for element in intent.get('must_include_elements', []):
                st.write(f"- {element}")
    
    # Content Gaps
    if results.get("content_gaps") and not results["content_gaps"].get("error"):
        st.markdown("### 🔍 Content Gap Analysis")
        gaps = results["content_gaps"]
        
        col1, col2 = st.columns(2)
        col1.metric("Quality Score", f"{gaps.get('content_quality_score', 0)}/100")
        col2.metric("Critical Gaps", len(gaps.get('critical_gaps', [])))
        
        with st.expander("❌ Critical Missing Topics (HIGH PRIORITY)"):
            for gap in gaps.get('critical_gaps', []):
                st.write(f"**{gap.get('missing_topic')}**")
                st.write(f"- Covered by: {gap.get('covered_by')}")
                st.write(f"- Priority: {gap.get('priority')}")
                st.write(f"- What to add: {gap.get('content_suggestion')}")
                st.write("")
        
        with st.expander("🎁 Your Unique Angles (Competitive Advantages)"):
            for angle in gaps.get('your_unique_angles', []):
                st.write(f"✅ {angle}")
        
        with st.expander("🌊 Blue Ocean Opportunities"):
            st.info("These are topics NO competitor covers well - easy to dominate!")
            for opp in gaps.get('blue_ocean_opportunities', []):
                st.write(f"💎 {opp}")


# Function to add to content brief
def enhance_content_brief(brief_content: str, advanced_analysis: Dict) -> str:
    """
    Enhance the content brief with advanced analysis insights.
    Call this from content_brief_creator.py
    """
    
    if not advanced_analysis:
        return brief_content
    
    enhanced = brief_content
    
    # Add E-E-A-T section
    if advanced_analysis.get('eeat_signals') and not advanced_analysis['eeat_signals'].get('error'):
        eeat = advanced_analysis['eeat_signals']
        enhanced += "\n\n## E-E-A-T Requirements\n\n"
        
        if eeat.get('is_ymyl'):
            enhanced += f"⚠️ **YMYL Topic ({eeat.get('ymyl_category')})** - E-E-A-T is critical!\n\n"
        
        enhanced += "### Trust Signals Required:\n"
        for signal in eeat.get('trust_signals', [])[:5]:
            enhanced += f"- {signal}\n"
        
        enhanced += "\n### Expertise Requirements:\n"
        for req in eeat.get('expertise_needed', [])[:3]:
            enhanced += f"- {req}\n"
    
    # Add Featured Snippet section
    if advanced_analysis.get('featured_snippets') and not advanced_analysis['featured_snippets'].get('error'):
        snippets = advanced_analysis['featured_snippets']
        high_opps = [o for o in snippets.get('high_opportunity_queries', []) if o.get('win_probability') == 'High'][:3]
        
        if high_opps:
            enhanced += "\n\n## Featured Snippet Opportunities\n\n"
            for opp in high_opps:
                enhanced += f"\n### Target Query: {opp.get('query')}\n"
                enhanced += f"**Format:** {opp.get('snippet_type')}\n"
                enhanced += f"**How to optimize:** {opp.get('optimization_guide')}\n"
    
    # Add Content Gaps section
    if advanced_analysis.get('content_gaps') and not advanced_analysis['content_gaps'].get('error'):
        gaps = advanced_analysis['content_gaps']
        critical = gaps.get('critical_gaps', [])[:5]
        
        if critical:
            enhanced += "\n\n## Critical Topics to Cover\n\n"
            for gap in critical:
                enhanced += f"\n### {gap.get('missing_topic')}\n"
                enhanced += f"**Priority:** {gap.get('priority')}\n"
                enhanced += f"{gap.get('content_suggestion')}\n"
    
    # Add User Intent section
    if advanced_analysis.get('search_intent') and not advanced_analysis['search_intent'].get('error'):
        intent = advanced_analysis['search_intent']
        enhanced += "\n\n## User Intent Deep Dive\n\n"
        enhanced += f"**What users need:** {intent.get('primary_user_need')}\n"
        enhanced += f"**Journey stage:** {intent.get('user_journey_stage')}\n"
        enhanced += f"**User level:** {intent.get('sophistication_level')}\n"
        enhanced += f"**Expected format:** {intent.get('expected_content_format')}\n"
        enhanced += f"**Recommended length:** {intent.get('content_length_recommendation')}\n\n"
        
        if intent.get('must_include_elements'):
            enhanced += "### Must Include:\n"
            for element in intent.get('must_include_elements', [])[:5]:
                enhanced += f"- {element}\n"
    
    return enhanced
