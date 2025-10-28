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
    DEFAULT_LLM_PROVIDER
)
from api.llm_client import get_llm_client
from api.scrapingdog_client import ScrapingdogClient, probe_scrapingdog_status
from utils.file_processing import read_tov_upload
from utils.web_scraping import fetch_and_parse_url
from analysis.keyword_analyzer import KeywordAnalyzer, KeywordClusterer
from analysis.keyword_extraction import extract_and_filter_keywords
from analysis.competitive_analyzer import CompetitiveAnalyzer
from analysis.content_brief_creator import ContentBriefCreator

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
# Initialize API key in session state to persist across reruns
if "scrapingdog_api_key" not in st.session_state:
    st.session_state.scrapingdog_api_key = SCRAPINGDOG_API_KEY
    try:
        st.session_state.scrapingdog_api_key = st.session_state.scrapingdog_api_key or st.secrets.get("scrapingdog_api_key")
    except Exception:
        pass

# Use session state value
scrapingdog_api_key = st.session_state.scrapingdog_api_key


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
if not scrapingdog_api_key:
    st.error("Scrapingdog API key not found. Enter it below or set SCRAPINGDOG_API_KEY.")
    entered_key = st.text_input("Enter Scrapingdog API Key:", type="password", key="scrapingdog_key_input")
    if entered_key:
        st.session_state.scrapingdog_api_key = entered_key
        scrapingdog_api_key = entered_key
        st.rerun()  # Rerun to use the new key
    else:
        st.stop()

st.sidebar.header("Scrapingdog Status")
refresh = st.sidebar.button("Refresh Scrapingdog Check", use_container_width=True)
if refresh:
    probe_scrapingdog_status.clear()
    st.rerun()  # Force rerun after clearing cache

# Always use the session state key
status_info = probe_scrapingdog_status(st.session_state.scrapingdog_api_key)

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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ========== Initialize API Clients ==========
# Note: Clients are re-instantiated with the current API key each time they're used
# to ensure any UI-entered key is picked up
def get_scrapingdog_client():
    """Get Scrapingdog client with current API key from session state."""
    return ScrapingdogClient(st.session_state.scrapingdog_api_key)

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


# ========== Step 2: Review & Select Keywords ==========
def show_step2():
    """Step 2: Review and select keywords."""
    st.header("Step 2: Review & Select Keywords")
    st.info("Review brainstormed keywords and SERP insights. Select the ones to target in your content.")

    # Display SERP insights
    if st.session_state.serp_insights:
        st.subheader("🌐 Key SERP Insights")
        si = st.session_state.serp_insights
        st.markdown(f"**Common Themes:** {si.get('common_themes','N/A')}")
        st.markdown(f"**Gaps to Exploit:** {si.get('gaps_to_exploit','N/A')}")
        st.markdown(f"**Unique Angles:** {si.get('unique_angles','N/A')}")
        st.markdown("---")

    # Display and edit keyword table
    if st.session_state.analyzed_keywords_df is not None and not st.session_state.analyzed_keywords_df.empty:
        st.subheader(f"📊 Top Keywords for '{st.session_state.query_topic}':")
        edited = st.data_editor(
            st.session_state.analyzed_keywords_df,
            column_config={
                "Selected": st.column_config.CheckboxColumn("Target?", default=True),
                "Keyword": "Keyword Phrase",
                "Inferred Potential Score": st.column_config.NumberColumn("Score", format="%.1f"),
                "Content Type": st.column_config.TextColumn("Intent"),
                "Requires Own Content": "Own Page?",
                "Rationale for Own Page": "Rationale",
                "Semantically Related Keywords": "Related Keywords for Grouping"
            },
            num_rows="dynamic",
            use_container_width=True,
            key="keyword_selection_data_editor"
        )
        st.session_state.analyzed_keywords_df = edited
        st.session_state.analyzed_keywords_df["Selected"] = st.session_state.analyzed_keywords_df["Selected"].astype(bool)

        # Update primary keyword selection
        selected_rows = st.session_state.analyzed_keywords_df[st.session_state.analyzed_keywords_df["Selected"]]
        if not selected_rows.empty:
            st.session_state.selected_brief_keyword = selected_rows.sort_values(
                by="Inferred Potential Score", ascending=False
            ).iloc[0]["Keyword"]
            st.session_state.related_keywords_for_brief = selected_rows[
                (selected_rows["Keyword"].str.lower() != st.session_state.selected_brief_keyword.lower())
            ]["Keyword"].tolist()
            st.info(f"Primary Keyword for Brief: **{st.session_state.selected_brief_keyword}**")
            st.markdown("Related Keywords (sample): " + ", ".join(st.session_state.related_keywords_for_brief[:10]) + ".")
        else:
            st.warning("Select at least one keyword for the content brief.")
            st.session_state.selected_brief_keyword = None

        # Keyword visualization
        freq = st.session_state.analyzed_keywords_df["Keyword"].str.lower().value_counts().head(20)
        if not freq.empty:
            st.subheader("Keyword Frequency (Top 20)")
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_bar(x=freq.index.str.slice(0, 40), y=freq.values)
                fig.update_layout(title="Keyword Frequency (Top 20)", xaxis_title="Keyword", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(freq)

        # Additional analysis tools
        st.divider()
        st.subheader("Optional: Competitor & Clustering Tools")

        c1, c2, c3 = st.columns(3)
        with c1:
            num_comp = st.number_input("Top competitors to analyze", 1, 10, 3, step=1)
        with c2:
            run_comp_btn = st.button("Run Competitive Analysis")
        with c3:
            run_cluster_btn = st.button("Cluster Top Keywords")

        if run_comp_btn:
            if not st.session_state.selected_brief_keyword:
                st.warning("Select a main keyword first (checkbox) to analyze competitors.")
            else:
                with st.spinner("Analyzing competitors..."):
                    scrapingdog = get_scrapingdog_client()
                    competitive_analyzer = CompetitiveAnalyzer(scrapingdog)
                    st.session_state.competitor_analysis = competitive_analyzer.analyze_competitors(
                        st.session_state.selected_brief_keyword,
                        num_competitors=int(num_comp)
                    )

        # Display competitor analysis
        if st.session_state.competitor_analysis:
            comp = st.session_state.competitor_analysis
            st.markdown(f"**Average word count (successful pages):** {comp.get('avg_word_count', 0)}")
            if comp.get("common_headings"):
                st.markdown("**Common headings across competitors:**")
                for h in comp["common_headings"]:
                    st.write(f"- {h}")
            if comp.get("competitors"):
                comp_df = pd.DataFrame([
                    {
                        "Title": c.get("title",""),
                        "URL": c.get("url",""),
                        "WordCount": c.get("word_count",0),
                        "Snippet": c.get("snippet","")
                    }
                    for c in comp.get("competitors", [])
                ])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                if PLOTLY_AVAILABLE and not comp_df.empty:
                    try:
                        fig = go.Figure()
                        fig.add_bar(x=comp_df["Title"].str.slice(0, 40), y=comp_df["WordCount"])
                        fig.update_layout(
                            title="Word count of analyzed competitor pages",
                            xaxis_title="Title (truncated)",
                            yaxis_title="Word Count"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass

        if run_cluster_btn:
            if st.session_state.analyzed_keywords_df is None or st.session_state.analyzed_keywords_df.empty:
                st.warning("No keywords available to cluster.")
            else:
                with st.spinner("Clustering top keywords with the LLM..."):
                    st.session_state.keyword_clusters = clusterer.create_clusters(
                        st.session_state.analyzed_keywords_df
                    )

        # Display clusters
        if st.session_state.keyword_clusters:
            st.subheader("📁 Keyword Clusters")
            for cluster_name, keywords in st.session_state.keyword_clusters.items():
                with st.expander(f"{cluster_name} ({len(keywords)} keywords)"):
                    st.write(", ".join(keywords))

    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Step 1"):
            st.session_state.current_step = 1
            st.rerun()
    with col2:
        if st.button("Proceed to Audit Analysis (Step 3)", type="primary"):
            if st.session_state.selected_brief_keyword:
                st.session_state.current_step = 3
                # Fetch webpage if URL provided
                if st.session_state.client_url:
                    with st.spinner(f"Fetching {st.session_state.client_url}"):
                        page = fetch_and_parse_url(st.session_state.client_url)
                        if page.startswith(("Error", "No substantial content")):
                            st.warning(f"Could not retrieve webpage content: {page}")
                            st.session_state.fetched_webpage_content = ""
                        else:
                            st.success("Webpage content fetched.")
                            st.session_state.fetched_webpage_content = page
                else:
                    st.session_state.fetched_webpage_content = ""
                st.rerun()
            else:
                st.warning("Select a primary keyword for the brief before proceeding.")


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
                )

            if st.session_state.generated_brief_content and not st.session_state.generated_brief_content.startswith("Error"):
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
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back to Audit Analysis (Step 3)"):
            st.session_state.current_step = 3
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

# Footer
st.markdown("---")
st.info(
    f"💡 **LLM Provider:** {DEFAULT_LLM_PROVIDER} | "
    f"For cloud deployment (HuggingFace Spaces), set environment variable `LLM_PROVIDER=deepseek` "
    f"and configure `DEEPSEEK_API_KEY`."
)
