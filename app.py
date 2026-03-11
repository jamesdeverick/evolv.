# --------------------------------------------
# Advanced LLM SEO Assistant - Refactored & Cloud-Ready
# --------------------------------------------

import os
import urllib.parse
from datetime import datetime

import pandas as pd
import streamlit as st

# Import modules
from config import (
    BRIEF_TONES,
    CONTENT_TYPES,
    SCRAPINGDOG_API_KEY,
    DEFAULT_LLM_PROVIDER,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL
)
from api.llm_client import get_llm_client
from api.scrapingdog_client import ScrapingdogClient, probe_scrapingdog_status
from utils.file_processing import read_tov_upload
from utils.web_scraping import fetch_and_parse_url
from analysis.keyword_analyzer import KeywordAnalyzer, KeywordClusterer
from analysis.keyword_extraction import extract_and_filter_keywords
from analysis.competitive_analyzer import CompetitiveAnalyzer
from analysis.content_brief_creator import ContentBriefCreator
from advanced_seo_addon import show_advanced_analysis_section
from page_optimizer import PageTypeManager, WireframeGenerator

# Optional plotting
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ========== App Config ==========
st.set_page_config(layout="wide", page_title="Advanced LLM SEO Assistant")
st.title("💡 Advanced LLM SEO Assistant")
st.markdown("Combines LLM audit analysis with real-time keyword research, SERP insights, and content brief generation.")


# ========== Get API Keys ==========
# Scrapingdog API key (ENV > secrets > UI override)
def _sanitize_key(k):
    """Strip quotes, whitespace, and validate key."""
    if not k:
        return None
    k = str(k).strip()
    # Strip accidental wrapping quotes
    if (k.startswith('"') and k.endswith('"')) or (k.startswith("'") and k.endswith("'")):
        k = k[1:-1].strip()
    return k or None

def _mask(k):
    """Mask API key for display."""
    if not k:
        return "[none]"
    if len(k) <= 6:
        return k[0] + "…"
    return k[:2] + "…" + k[-4:]

# Prefer ENV first (secrets might be stale/blank)
env_key = _sanitize_key(os.getenv("SCRAPINGDOG_API_KEY"))

secret_key = None
try:
    secret_key = _sanitize_key(st.secrets.get("scrapingdog_api_key", None))
except Exception:
    pass

scrapingdog_api_key = env_key or secret_key


# ========== Sidebar: LLM & Settings ==========
st.sidebar.header("LLM & Settings")

# Get LLM client
llm_client = get_llm_client()
llm_status = llm_client.get_status()

if llm_status["available"]:
    st.sidebar.success(f"✓ LLM Ready: {llm_status['provider']} ({llm_status['model']})")
else:
    st.sidebar.error(f"✗ LLM Unavailable: {llm_status.get('error', 'Unknown error')}")
    st.sidebar.info(f"Current provider: {llm_status['provider']}")

# Tone selection
if "brief_tone" not in st.session_state:
    st.session_state.brief_tone = BRIEF_TONES[0]

st.session_state.brief_tone = st.sidebar.selectbox(
    "Base tone (high-level style)",
    BRIEF_TONES,
    index=BRIEF_TONES.index(st.session_state.brief_tone)
)

# Tone of Voice Document Upload
st.sidebar.subheader("Tone of Voice (document upload)")
tov_file = st.sidebar.file_uploader(
    "Upload ToV (DOCX / PDF / TXT / MD)",
    type=["docx", "pdf", "txt", "md"]
)

if "tov_text" not in st.session_state:
    st.session_state.tov_text = ""

if tov_file is not None:
    st.session_state.tov_text = read_tov_upload(tov_file)
    if st.session_state.tov_text.strip():
        st.sidebar.success("Tone of Voice file loaded.")
    else:
        st.sidebar.warning("Uploaded file is empty or unreadable.")

# Additional ToV notes
tov_notes = st.sidebar.text_area(
    "Additional ToV notes (optional)",
    value=os.getenv("CLIENT_TONE_NOTES", ""),
    height=120,
    help="Extra constraints (e.g., banned phrases, industry terms, UK/US spelling)."
)

# Preview ToV
with st.sidebar.expander("Preview Tone of Voice text", expanded=False):
    preview = (st.session_state.tov_text or "").strip()
    if tov_notes.strip():
        preview = (preview + "\n\n" + tov_notes).strip()
    st.code(preview[:2000] + ("…" if len(preview) > 2000 else ""))


# ========== Scrapingdog Status ==========
st.sidebar.header("Scrapingdog API Key")
# UI override (handy on RunPod)
ui_key = _sanitize_key(
    st.sidebar.text_input(
        "Override API key (optional)",
        type="password",
        help="Paste to override env/secrets for this session"
    )
)
if ui_key:
    scrapingdog_api_key = ui_key

st.sidebar.caption(f"Using key: {_mask(scrapingdog_api_key)}")

if not scrapingdog_api_key:
    st.error("Scrapingdog API key not found. Set SCRAPINGDOG_API_KEY env, put it in .streamlit/secrets.toml, or paste above.")
    st.stop()

st.sidebar.header("Scrapingdog Status")
refresh = st.sidebar.button("Refresh Scrapingdog Check", use_container_width=True)
if refresh:
    probe_scrapingdog_status.clear()  # Clear cache
    st.rerun()  # Ensure the new key flows everywhere

# Always use the current key
status_info = probe_scrapingdog_status(scrapingdog_api_key)

if status_info["ok"]:
    if status_info["http_status"] == 200:
        st.sidebar.success("✓ Connected (HTTP 200)")
    else:
        st.sidebar.warning(f"⚠ Data present (HTTP {status_info['http_status']})")
    st.sidebar.write(f"- Related: **{status_info['related_count']}**")
    st.sidebar.write(f"- People Also Ask: **{status_info['paa_count']}**")
    st.sidebar.write(f"- Organic results: **{status_info['organic_count']}**")
else:
    st.sidebar.error(f"✗ Not OK (HTTP {status_info['http_status']})")
    with st.sidebar.expander("Response body (first 300 chars)"):
        st.sidebar.code(status_info["body_sample"])


# ========== Session State Initialization ==========
def init_state():
    """Initialize session state variables."""
    defaults = {
        "current_step": 1,
        "query_topic": "",
        "client_url": "",
        "report_content": "",
        "llm_analysis_output": "",
        "fetched_webpage_content": "",
        "analyzed_keywords_df": None,
        "selected_brief_keyword": None,
        "related_keywords_for_brief": None,
        "serp_insights": None,
        "desired_content_intent": "Any",
        "competitor_analysis": None,
        "keyword_clusters": {},
        "generated_brief_content": "",
        "drafted_content": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ========== Initialize API Clients ==========
# Note: Clients are re-instantiated with the current API key each time they're used
# to ensure any UI-entered key is picked up
def get_scrapingdog_client():
    """Get Scrapingdog client with current API key."""
    return ScrapingdogClient(scrapingdog_api_key)

# Initialize non-API-dependent clients
brief_creator = ContentBriefCreator()
clusterer = KeywordClusterer()


# ========== Step 1: Topic & Client Info ==========
def show_step1():
    """Step 1: Enter audit topic and client info."""
    st.header("Step 1: Enter Audit Topic & Client Info")
    st.session_state.query_topic = st.text_input(
        "Main Topic / Target Query (e.g., 'corporate budgeting'):",
        st.session_state.query_topic
    )
    st.session_state.client_url = st.text_input(
        "Client URL (optional):",
        st.session_state.client_url,
        help="Leave blank if this is net-new content."
    )
    st.session_state.desired_content_intent = st.selectbox(
        "Desired Content Intent for Supporting Keywords:",
        CONTENT_TYPES,
        index=CONTENT_TYPES.index(st.session_state.desired_content_intent)
    )
    
    # ADD THIS SECTION - Page Type Selection
    st.divider()
    st.subheader("📄 Page Type")
    st.info("Select the type of page you're creating to get tailored content requirements")
    
    page_types = PageTypeManager.get_all_page_types()
    st.session_state.page_type = st.selectbox(
        "What type of page are you creating?",
        options=list(page_types.keys()),
        format_func=lambda x: page_types[x],
        index=0
    )
    
    # Show page type info
    page_info = PageTypeManager.get_page_type_info(st.session_state.page_type)
    with st.expander("ℹ️ About this page type"):
        st.markdown(f"**Purpose:** {page_info['description']}")
        st.markdown(f"**Typical Length:** {page_info['typical_length']}")
        st.markdown(f"**SEO Focus:** {page_info['seo_focus']}")
    st.divider()
    # END OF NEW SECTION
    
    uploaded_file = st.file_uploader(
        "Upload Audit Findings (Markdown/Text/PDF)",
        type=["md", "txt", "pdf"]
    )
    if uploaded_file is not None:
        try:
            # Handle PDF
            if uploaded_file.type == "application/pdf":
                try:
                    import PyPDF2
                    import io
                    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                    st.session_state.report_content = text
                except ImportError:
                    st.error("PyPDF2 not available. Install it or upload text/markdown.")
                except Exception as e:
                    st.error(f"PDF read error: {e}")
            else:
                st.session_state.report_content = uploaded_file.getvalue().decode("utf-8", errors="replace")
            st.success(f"File '{uploaded_file.name}' uploaded.")
            with st.expander("Preview Uploaded Content"):
                preview = st.session_state.report_content
                st.code(preview[:1000] + "..." if len(preview) > 1000 else preview, language="markdown")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    # Display Scrapingdog status
    st.subheader("Scrapingdog Connection")
    si = status_info
    if si["ok"]:
        st.success(
            f"Usable payload (HTTP {si['http_status']}). "
            f"Related={si['related_count']}, PAA={si['paa_count']}, Organic={si['organic_count']}"
        )
    else:
        st.error(f"Not usable (HTTP {si['http_status']}).")
        with st.expander("Server body (first 300 chars)"):
            st.code(si["body_sample"])
    if st.button("Proceed to Keyword Research (Step 2)", type="primary"):
        if not st.session_state.query_topic:
            st.warning("Please enter a main topic/target query.")
            return
        with st.spinner("Performing initial keyword research and SERP analysis..."):
            scrapingdog = get_scrapingdog_client()
            keyword_analyzer = KeywordAnalyzer(scrapingdog)
            related, paa, organics = scrapingdog.get_keywords(st.session_state.query_topic)
            df, selected_kw, serp_insights = keyword_analyzer.analyze_and_identify_keywords(
                st.session_state.query_topic,
                related + paa,
                organics,
                st.session_state.desired_content_intent
            )
            st.session_state.analyzed_keywords_df = df
            st.session_state.selected_brief_keyword = selected_kw
            st.session_state.serp_insights = serp_insights
            st.session_state.current_step = 2
        st.rerun()


# KEYWORD CLUSTERING INTEGRATION GUIDE

## What This Does

**Problem:** You get 50-100 keywords but many aren't relevant to your main topic - they're for different subtopics, intents, or tangential queries.

**Solution:** Automatically groups keywords into semantic clusters so you can:
- See which keyword groups exist (e.g., "How-To Guides", "Pricing", "Comparisons")
- Select only relevant clusters for your content
- Get focused keyword lists instead of keyword soup

## Example Output

**Main Keyword:** "corporate budgeting"

**Clusters Found:**
1. ✅ **Budget Planning** (15 keywords, 85% relevant) - Main topic
   - corporate budget planning, budget planning process, annual budget planning...
2. ⚡ **Software/Tools** (12 keywords, 72% relevant)
   - budgeting software, budget tools, corporate budget template...
3. 📊 **How-To Guides** (8 keywords, 68% relevant)
   - how to create corporate budget, how to manage budget...
4. 💰 **Cost Management** (6 keywords, 45% relevant)
   - cost reduction strategies, expense management...
5. ❓ **Definitions** (4 keywords, 35% relevant)
   - what is corporate budgeting, budgeting definition...

**User selects:** Clusters 1, 2, 3 → Gets 35 focused keywords instead of 50+ mixed ones

## Files to Add

1. **keyword_clustering.py** → `analysis/` folder (NEW)

## Integration in app.py - Step 2

### After keyword analysis completes, add clustering:

```python
# In show_step2(), after keyword analysis is done:

if st.session_state.analyzed_keywords_df is not None and not st.session_state.analyzed_keywords_df.empty:
    st.divider()
    st.subheader("🎯 Keyword Clustering")
    st.info("Group keywords by topic to focus on relevant clusters")
    
    # Import clusterer
    from analysis.keyword_clustering import KeywordClusterer
    
    # Run clustering
    if st.button("Cluster Keywords by Topic", type="secondary"):
        with st.spinner("Clustering keywords..."):
            clusterer = KeywordClusterer()
            clustering_result = clusterer.cluster_keywords(
                keywords_df=st.session_state.analyzed_keywords_df,
                main_keyword=st.session_state.query_topic,
                max_clusters=8
            )
            st.session_state.keyword_clusters = clustering_result
    
    # Display clusters
    if st.session_state.get('keyword_clusters'):
        clusters = st.session_state.keyword_clusters
        
        st.markdown(f"### Found {clusters['total_clusters']} Keyword Groups")
        st.markdown(f"*Total keywords: {clusters['total_keywords']}*")
        
        # Show each cluster
        cluster_selections = []
        
        for i, cluster in enumerate(clusters['clusters']):
            is_main = (i == clusters.get('main_topic_cluster'))
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Checkbox for selection
                default_selected = is_main or cluster.get('relevance_score', 0) > 50
                selected = st.checkbox(
                    f"{'✅ ' if is_main else ''}{cluster['name']}",
                    value=default_selected,
                    key=f"cluster_{i}"
                )
                
                if selected:
                    cluster_selections.append(cluster['name'])
            
            with col2:
                # Metrics
                st.metric("Keywords", cluster['size'])
                st.metric("Relevance", f"{cluster.get('relevance_score', 0)}%")
            
            # Expandable details
            with st.expander(f"View {cluster['size']} keywords in this group"):
                st.markdown(f"**Average Score:** {cluster.get('avg_keyword_score', 0)}")
                st.markdown(f"**PAA Questions:** {cluster.get('paa_percentage', 0)}%")
                
                if cluster.get('theme_words'):
                    st.markdown(f"**Theme:** {', '.join(cluster['theme_words'])}")
                
                # Show keywords
                for kw in cluster['keywords'][:10]:
                    st.write(f"- {kw}")
                
                if len(cluster['keywords']) > 10:
                    st.write(f"*...and {len(cluster['keywords']) - 10} more*")
        
        # Apply filter button
        st.divider()
        if st.button("Apply Selected Clusters", type="primary"):
            if cluster_selections:
                # Filter DataFrame to selected clusters
                filtered_df = clusterer.filter_keywords_by_clusters(
                    keywords_df=st.session_state.analyzed_keywords_df,
                    selected_cluster_names=cluster_selections,
                    clustering_result=st.session_state.keyword_clusters
                )
                
                st.session_state.analyzed_keywords_df = filtered_df
                st.success(f"✅ Filtered to {len(filtered_df)} keywords from {len(cluster_selections)} clusters")
                st.rerun()
            else:
                st.warning("Select at least one cluster")

# ========== Step 3: LLM Audit Analysis ==========
def show_step3():
    """Step 3: Get LLM audit analysis."""
    st.header("Step 3: Get LLM Audit Analysis")
    st.info(
        f"The LLM will analyze your audit report and/or SERP + keyword context for "
        f"**{st.session_state.query_topic}**. This works even when there is **no existing "
        f"client URL** (net-new content)."
    )
    # Build context
    kw_df = st.session_state.analyzed_keywords_df if st.session_state.analyzed_keywords_df is not None else pd.DataFrame()
    top_kw_list = []
    if not kw_df.empty and "Keyword" in kw_df.columns:
        top_kw_list = kw_df["Keyword"].dropna().astype(str).head(25).tolist()
    serp_summary = st.session_state.serp_insights or {
        "common_themes": "", "gaps_to_exploit": "", "unique_angles": ""
    }
    default_review_prompt = f"""
You are an SEO content strategist.
Goal: produce a crisp gap/opportunity analysis and a **proposed content outline** for a NEW PAGE targeting:
- Main topic: **{st.session_state.query_topic}**
- If an existing client page is unavailable, base recommendations on SERP insights and the keyword list.
Client URL: {st.session_state.client_url or '[No existing page — new content required]'}
---
Client Webpage Content (extracted):
{st.session_state.fetched_webpage_content or '[Not available]'}
---
Audit Findings (uploaded):
{st.session_state.report_content or '[Not provided]'}
---
SERP Insights (summary):
- Common themes: {serp_summary.get('common_themes','')}
- Gaps to exploit: {serp_summary.get('gaps_to_exploit','')}
- Unique angles: {serp_summary.get('unique_angles','')}
---
Candidate Keywords (top ~25):
{top_kw_list or '[No table — rely on topic + SERP]'}
---
Please return a concise analysis with:
1) Primary search intent & user needs
2) Key gaps/opportunities (bullet points)
3) A recommended **H2/H3 outline** with one-line summaries
4) On-page SEO elements (Title/H1/Meta/URL)
5) 3–5 FAQs with concise answers
""".strip()
    review_prompt = st.text_area(
        "Enter your prompt for the LLM review:",
        value=default_review_prompt,
        height=400
    )
    if st.button("Run LLM Analysis", type="primary"):
        with st.spinner(f"Sending content to {llm_client.model} for analysis..."):
            result = llm_client.complete(review_prompt, temperature=0.7, max_tokens=2000)
            st.session_state.llm_analysis_output = result
            st.subheader("💡 LLM's Analysis")
            st.markdown(result)
            if not result.startswith("Error"):
                st.session_state.current_step = 4
    
    # ADD THIS SECTION - Wireframe Generator
    if st.session_state.client_url and st.session_state.fetched_webpage_content:
        st.divider()
        st.subheader("🎨 Page Optimization Wireframe")
        st.info("Analyze the actual HTML structure to show optimization opportunities")
        
        if st.button("Generate Wireframe", type="secondary"):
            with st.spinner("Parsing HTML and generating wireframe..."):
                wireframe_gen = WireframeGenerator()
                wireframe = wireframe_gen.generate_wireframe(
                    url=st.session_state.client_url,
                    html_content=st.session_state.fetched_webpage_content,
                    page_type=st.session_state.get('page_type', 'blog_post'),
                    keyword=st.session_state.get('selected_brief_keyword', st.session_state.query_topic)
                )
                st.session_state.wireframe = wireframe
        
        # Display wireframe if generated
        if st.session_state.get('wireframe'):
            wf = st.session_state.wireframe
            
            # Show HTML structure score
            if wf.get('html_structure_score') is not None:
                score = wf['html_structure_score']
                color = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
                st.metric("HTML Structure Score", f"{color} {score}/100")
            
            # Show current issues
            st.markdown("### 📊 Current Page Issues")
            for issue in wf.get('current_issues', []):
                st.warning(f"❌ {issue}")
            
            # Show recommended structure
            st.markdown("### ✨ Recommended Page Structure")
            for i, section in enumerate(wf.get('recommended_sections', []), 1):
                status_emoji = "✅" if section.get('current_status') == 'present' else "⚠️" if section.get('current_status') == 'needs_improvement' else "❌"
                
                with st.expander(f"{status_emoji} {i}. {section.get('name', 'Section')}"):
                    st.markdown(f"**Purpose:** {section.get('purpose', 'N/A')}")
                    if section.get('current_status'):
                        st.markdown(f"**Status:** {section.get('current_status', 'unknown').replace('_', ' ').title()}")
                    st.markdown("**Elements to include:**")
                    for elem in section.get('elements', []):
                        st.markdown(f"- {elem}")
            
            # Show priority fixes
            if wf.get('priority_fixes'):
                st.markdown("### 🎯 Priority Fixes")
                for fix in wf['priority_fixes']:
                    impact_emoji = "🔥" if fix.get('impact') == 'high' else "⚡" if fix.get('impact') == 'medium' else "✓"
                    effort_emoji = "🟢" if fix.get('effort') == 'low' else "🟡" if fix.get('effort') == 'medium' else "🔴"
                    st.write(f"{impact_emoji} **{fix.get('fix')}** (Impact: {fix.get('impact')}, Effort: {effort_emoji} {fix.get('effort')})")
    # END OF NEW SECTION
    
    # Navigation
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back to Keyword Selection (Step 2)"):
            st.session_state.current_step = 2
            st.rerun()
    with c2:
        if st.button("Proceed to Brief Generation (Step 4)"):
            st.session_state.current_step = 4
            st.rerun()

# ========== Step 4: Generate Content Brief ==========
def show_step4():
    """Step 4: Generate and export content brief."""
    st.header("Step 4: Generate Content Brief")
    st.info(
        f"Generating a comprehensive content brief for "
        f"'{st.session_state.selected_brief_keyword}' based on collected data."
    )

    # Build strict ToV
    strict_tov_text = (st.session_state.tov_text or "").strip()
    if tov_notes.strip():
        strict_tov_text = (strict_tov_text + "\n\n" + tov_notes).strip()

    # Display context
    serp_block = ""
    if st.session_state.serp_insights:
        si = st.session_state.serp_insights
        serp_block += "\n### Competitive Differentiators (Based on SERP Insights):"
        serp_block += f"\n* **Common Themes:** {si.get('common_themes','N/A')}"
        serp_block += f"\n* **Gaps to Exploit:** {si.get('gaps_to_exploit','N/A')}"
        serp_block += f"\n* **Unique Angles:** {si.get('unique_angles','N/A')}"

    related_str = ""
    rel = st.session_state.related_keywords_for_brief or []
    if rel:
        related_str = f"**Additional Important Keywords:** {', '.join([k for k in rel if isinstance(k,str)][:15])}"

    display_lines = [
        f"**Main Topic:** {st.session_state.query_topic or '[Not specified]'}",
        f"**Target Keyword:** {st.session_state.selected_brief_keyword or '[Not selected]'}",
        f"**Client URL:** {st.session_state.client_url or '[Not provided]'}",
        f"**Base Tone:** {st.session_state.brief_tone}",
        "",
        "**Client Webpage Content (excerpt):**",
        (st.session_state.fetched_webpage_content[:800] + "…") if st.session_state.fetched_webpage_content else "[Not available]",
        "",
        "**Audit Findings (excerpt):**",
        (st.session_state.report_content[:800] + "…") if st.session_state.report_content else "[Not provided]",
        "",
        "**LLM Previous Analysis (excerpt):**",
        (st.session_state.llm_analysis_output[:800] + "…") if st.session_state.llm_analysis_output else "[No previous analysis]",
        "",
        "**SERP Insights:**",
        serp_block or "[Not available]",
        "",
        related_str or "",
        "",
        "**Tone of Voice (STRICT):**",
        (strict_tov_text[:800] + "…") if strict_tov_text else "[No ToV uploaded]"
    ]

    st.text_area(
        "Prompt context (read-only, for reference)",
        value="\n".join(display_lines),
        height=420
    )

    if st.button("Generate Content Brief", type="primary"):
        if not st.session_state.selected_brief_keyword:
            st.error("Select a primary keyword in Step 2 to generate a brief.")
        else:
            with st.spinner(f"Generating content brief with {llm_client.model}..."):
                st.session_state.generated_brief_content = brief_creator.create_brief(
                    keyword=st.session_state.selected_brief_keyword,
                    related_keywords=st.session_state.related_keywords_for_brief,
                    main_topic=st.session_state.query_topic,
                    client_url=st.session_state.client_url,
                    webpage_content=st.session_state.fetched_webpage_content,
                    audit_findings=st.session_state.report_content,
                    llm_analysis=st.session_state.llm_analysis_output,
                    serp_insights=st.session_state.serp_insights,
                    tone_style=st.session_state.brief_tone,
                    tone_guidelines=strict_tov_text,
                    competitor_analysis=st.session_state.competitor_analysis,
                    advanced_analysis=st.session_state.get('advanced_analysis'),
                    page_type=st.session_state.get('page_type', 'blog_post')
                )

            if st.session_state.generated_brief_content and not st.session_state.generated_brief_content.startswith("Error"):
                # Enhance with page type requirements
                st.session_state.generated_brief_content = PageTypeManager.enhance_brief_for_page_type(
                    st.session_state.generated_brief_content,
                    st.session_state.get('page_type', 'blog_post')
                )
                
                st.subheader("📝 Generated Content Brief")
                st.markdown(st.session_state.generated_brief_content)
            else:
                st.error("Failed to generate content brief.")

    # Edit and download
    if st.session_state.generated_brief_content and not st.session_state.generated_brief_content.startswith("Error"):
        st.markdown("#### Edit the brief before exporting")
        edited = st.text_area(
            "Edit Brief",
            value=st.session_state.generated_brief_content,
            height=500
        )
        st.session_state.generated_brief_content = edited

        st.download_button(
            "Download Brief (Markdown)",
            data=edited.encode("utf-8"),
            file_name=f"content_brief_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )

    # Navigation
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Back to Audit Analysis (Step 3)"):
            st.session_state.current_step = 3
            st.rerun()
    with c2:
        if st.session_state.generated_brief_content and not st.session_state.generated_brief_content.startswith("Error"):
            if st.button("✨ Draft Content with Claude (Step 5)", type="primary"):
                st.session_state.current_step = 5
                st.rerun()
    with c3:
        if st.button("Reset All and Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_state()
            st.rerun()

def show_step5():
    """Step 5: Draft content using Claude API."""
    st.header("Step 5: Draft Content with Claude")

    # Check if we have a brief to work with
    if not st.session_state.generated_brief_content or st.session_state.generated_brief_content.startswith("Error"):
        st.error("No content brief available. Please complete Step 4 first.")
        if st.button("Back to Step 4"):
            st.session_state.current_step = 4
            st.rerun()
        return

    st.info(
        f"Using Claude to draft full content based on your brief for "
        f"'{st.session_state.selected_brief_keyword}'"
    )

    # Display brief summary
    with st.expander("📋 Content Brief (Click to expand)", expanded=False):
        st.markdown(st.session_state.generated_brief_content)

    # Claude configuration
    st.subheader("Claude Configuration")

    # Get or create Claude client
    claude_available = bool(ANTHROPIC_API_KEY)

    if claude_available:
        st.success(f"✓ Claude API configured: {ANTHROPIC_MODEL}")
    else:
        st.warning("⚠ Claude API key not found in environment")
        api_key_input = st.text_input(
            "Enter your Anthropic API Key:",
            type="password",
            help="Get your API key from https://console.anthropic.com/"
        )
        if api_key_input:
            # Temporarily set the API key
            import os
            os.environ["ANTHROPIC_API_KEY"] = api_key_input
            from config import ANTHROPIC_API_KEY as temp_key
            claude_available = bool(api_key_input)
        else:
            st.stop()

    # Content length selection
    word_count = st.slider(
        "Target word count:",
        min_value=500,
        max_value=3000,
        value=1500,
        step=100,
        help="Approximate number of words for the drafted content"
    )

    # Additional instructions
    additional_instructions = st.text_area(
        "Additional instructions for Claude (optional):",
        height=100,
        placeholder="e.g., Include specific examples, focus on practical tips, add statistics, etc."
    )

    # Generate button
    if st.button("🚀 Generate Content Draft", type="primary"):
        with st.spinner(f"Claude is drafting your content (~{word_count} words)..."):
            # Create the prompt for Claude
            additional_instr = f"\n\nADDITIONAL INSTRUCTIONS:\n{additional_instructions}" if additional_instructions else ""

            prompt = f"""You are an expert SEO content writer. Based on the following content brief, write a complete, publication-ready article.

CONTENT BRIEF:
{st.session_state.generated_brief_content}

TARGET WORD COUNT: ~{word_count} words{additional_instr}

Please write a complete, engaging article that:
1. Follows the structure outlined in the brief
2. Incorporates all target keywords naturally
3. Addresses the user intent and SERP insights
4. Maintains the specified tone of voice
5. Is well-formatted with proper headings (## and ###)
6. Includes engaging introduction and conclusion
7. Uses clear, scannable paragraphs

Write the article in Markdown format. Begin now:"""

            # Use Anthropic client for drafting
            claude_client = get_llm_client(provider="anthropic")

            if not claude_client.available:
                st.error(f"❌ Claude API not available: {claude_client.get_status().get('error', 'Unknown error')}")
                st.info("Make sure ANTHROPIC_API_KEY is set in your environment or enter it above.")
            else:
                # Generate the content
                st.session_state.drafted_content = claude_client.complete(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=word_count * 2  # Rough token estimate
                )

                if st.session_state.drafted_content and not st.session_state.drafted_content.startswith("Error"):
                    st.success("✅ Content draft generated successfully!")
                else:
                    st.error("❌ Failed to generate content draft.")
                    # Show the actual error for debugging
                    with st.expander("Error Details"):
                        st.code(st.session_state.drafted_content if st.session_state.drafted_content else "No response received")
                        st.info("**Troubleshooting:**\n"
                               "1. Check that ANTHROPIC_API_KEY is set correctly\n"
                               "2. Verify your API key is valid at https://console.anthropic.com/\n"
                               "3. Check you have API credits available\n"
                               f"4. Current API key starts with: {ANTHROPIC_API_KEY[:10] if ANTHROPIC_API_KEY else 'NOT SET'}...")

    # Display and edit drafted content
    if st.session_state.drafted_content and not st.session_state.drafted_content.startswith("Error"):
        st.markdown("---")
        st.subheader("📝 Generated Content Draft")

        # Show preview
        with st.expander("Preview (rendered)", expanded=True):
            st.markdown(st.session_state.drafted_content)

        # Edit area
        st.markdown("#### Edit the content before exporting")
        edited_content = st.text_area(
            "Edit Content",
            value=st.session_state.drafted_content,
            height=600,
            key="content_editor"
        )
        st.session_state.drafted_content = edited_content

        # Word count
        word_count_actual = len(edited_content.split())
        st.caption(f"📊 Word count: {word_count_actual} words")

        # Download button
        st.download_button(
            "⬇ Download Content (Markdown)",
            data=edited_content.encode("utf-8"),
            file_name=f"content_{st.session_state.selected_brief_keyword.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )

    # Navigation
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back to Content Brief (Step 4)"):
            st.session_state.current_step = 4
            st.rerun()
    with c2:
        if st.button("Reset All and Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_state()
            st.rerun()


# ========== Main Router ==========
if st.session_state.current_step == 1:
    show_step1()
elif st.session_state.current_step == 2:
    show_step2()
elif st.session_state.current_step == 3:
    show_step3()
elif st.session_state.current_step == 4:
    show_step4()
elif st.session_state.current_step == 5:
    show_step5()

# Footer
st.markdown("---")
st.info(
    f"💡 **LLM Provider:** {DEFAULT_LLM_PROVIDER} | "
    f"For cloud deployment (HuggingFace Spaces), set environment variable `LLM_PROVIDER=deepseek` "
    f"and configure `DEEPSEEK_API_KEY`."
)
