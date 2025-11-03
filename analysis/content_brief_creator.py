# --------------------------------------------
# Content Brief Creator
# --------------------------------------------

from typing import Optional, List, Dict, Any
from api.llm_client import get_llm_client


class ContentBriefCreator:
    """Create comprehensive SEO content briefs."""

    def __init__(self):
        """Initialize content brief creator."""
        self.llm = get_llm_client()

    def create_brief(
        self,
        keyword: str,
        related_keywords: Optional[List[str]],
        main_topic: str,
        client_url: str,
        webpage_content: str = "",
        audit_findings: str = "",
        llm_analysis: str = "",
        serp_insights: Optional[Dict[str, Any]] = None,
        tone_style: str = "Informative and friendly",
        tone_guidelines: str = ""
    ) -> str:
        """
        Create a comprehensive content brief.

        Args:
            keyword: Target keyword
            related_keywords: Supporting keywords
            main_topic: Main topic
            client_url: Client URL (or empty for net-new content)
            webpage_content: Extracted webpage content
            audit_findings: Uploaded audit report
            llm_analysis: Previous LLM analysis
            serp_insights: SERP analysis data
            tone_style: Base tone style
            tone_guidelines: Strict ToV guidelines from uploaded doc

        Returns:
            Generated content brief in Markdown
        """
        if not self.llm.available:
            return "Error: LLM not available for brief generation."

        # Build supporting keywords list (deduplicated)
        support_list = []
        if related_keywords:
            for kw in related_keywords:
                if isinstance(kw, str) and kw.strip():
                    support_list.append(kw.strip())

        # Deduplicate while preserving order
        seen = set()
        support_list = [
            k for k in support_list
            if not (k.lower() in seen or seen.add(k.lower()))
        ]

        # Build SERP insights block
        serp_block = ""
        if serp_insights:
            serp_block += "\n### Competitive Differentiators (Based on SERP Insights):"
            serp_block += f"\n* **Common Themes in Competitors' Content:** {serp_insights.get('common_themes', 'N/A')}"
            serp_block += f"\n* **Gaps to Exploit in Competitors' Content:** {serp_insights.get('gaps_to_exploit', 'N/A')}"
            serp_block += f"\n* **Unique Angles to Pursue:** {serp_insights.get('unique_angles', 'N/A')}"

        # Build tone header with strict guidelines
        strict_tov = (tone_guidelines or "").strip()
        tone_header = f"**Tone & Style (MANDATORY):** {tone_style}"
        if strict_tov:
            tone_header += f"\n\n**Client Tone of Voice Guidelines (STRICT — MUST COMPLY):**\n{strict_tov}"

        # Build supporting keywords section
        supporting_keywords_md = "**10. Supporting Keywords (use verbatim where natural):**\n"
        if support_list:
            supporting_keywords_md += "\n".join([f"- {k}" for k in support_list[:50]])
        else:
            supporting_keywords_md += "- [None provided]"

        # Build complete prompt
        prompt_lines = [
            "# SEO Content Brief",
            f"**Main Topic:** {main_topic or '[Not specified]'}",
            f"**Target Keyword:** {keyword}",
            f"**Client URL:** {client_url or '[Not provided - net-new content]'}",
            tone_header,
            "",
            "**Client Webpage Content (excerpt):**",
            (webpage_content[:1200] + "…") if webpage_content else "[Not available]",
            "",
            "**SEO Audit Findings (uploaded report):**",
            audit_findings or "[Not provided]",
            "",
            "**LLM's Previous Audit Analysis (if available):**",
            llm_analysis or "[No previous LLM analysis available]",
            "",
            "**Key SERP Insights:**",
            serp_block or "[Not available]",
            "",
            "**Return the entire brief in Markdown.**",
            "",
            "**1. Content Title (SEO Optimized)**",
            "**2. Primary Search Intent & Audience**",
            "**3. Content Outline with Detailed Annotations**",
            "   - Provide a hierarchical structure using H2 and H3 headings",
            "   - For EACH H2 heading, include:",
            "     * 2-3 sentence description of what this section should cover",
            "     * Key talking points (3-5 bullets)",
            "     * Supporting keywords to naturally incorporate",
            "   - For EACH H3 sub-heading, include:",
            "     * 1-2 sentence description",
            "     * Specific points to address (2-3 bullets)",
            "   - Example format:",
            "     ## Main Heading Topic",
            "     *Description:* Explain the core concept...",
            "     *Key Points:*",
            "     - Point 1",
            "     - Point 2",
            "     *Keywords:* keyword1, keyword2",
            "",
            "**4. Key Talking Points & Evidence**",
            "**5. On-page SEO Elements (Title, H1, Meta, URL)**",
            "**6. Internal Links & Anchor Ideas**",
            "**7. FAQs (3–5) and concise answers**",
            "**8. Schema Suggestions**",
            "**9. Competitive Differentiators**",
            supporting_keywords_md,
        ]

        prompt = "\n".join(prompt_lines)

        # Generate brief
        result = self.llm.complete(
            prompt,
            temperature=0.7,
            max_tokens=2000
        )

        if not result or result.startswith("Error"):
            return result or "No valid response from LLM for content brief creation."

        # Clean up formatting
        clean = "\n".join([ln.rstrip() for ln in result.splitlines()])

        if not clean.strip():
            return "No valid response from LLM for content brief creation."

        return clean
