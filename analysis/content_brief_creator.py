# --------------------------------------------
# Content Brief Creator
# --------------------------------------------

from typing import Optional, List, Dict, Any
from api.llm_client import get_llm_client

# Try to import NLP tools for generative search optimization
try:
    from analysis.entity_extractor import EntityExtractor
    from analysis.semantic_analyzer import SemanticAnalyzer
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False


class ContentBriefCreator:
    """Create comprehensive SEO content briefs with generative search optimization."""

    def __init__(self):
        """Initialize content brief creator with NLP capabilities."""
        self.llm = get_llm_client()

        # Initialize NLP tools if available
        if NLP_AVAILABLE:
            self.entity_extractor = EntityExtractor()
            self.semantic_analyzer = SemanticAnalyzer()
            self.nlp_available = (
                self.entity_extractor.available and
                self.semantic_analyzer.available
            )
        else:
            self.entity_extractor = None
            self.semantic_analyzer = None
            self.nlp_available = False

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
        tone_guidelines: str = "",
        competitor_analysis: Optional[Dict[str, Any]] = None
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

        # Build entity guidance for generative search optimization
        entity_guidance = ""
        if competitor_analysis and self.nlp_available:
            entity_guidance = self._build_entity_guidance(competitor_analysis)

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
        ]

        # Add entity guidance if available
        if entity_guidance:
            prompt_lines.extend([
                "**🤖 Generative AI Optimization Guidance:**",
                entity_guidance,
                ""
            ])

        prompt_lines.extend([
            "**Return the entire brief in Markdown.**",
            "",
            "**1. Content Title (SEO Optimized)**",
            "**2. Primary Search Intent & Audience**",
            "**3. Content Outline with Detailed Annotations**",
            "",
            "CRITICAL FORMATTING REQUIREMENT: You MUST use markdown heading syntax:",
            "- ALL H2 headings MUST start with ## (two hash symbols)",
            "- ALL H3 headings MUST start with ### (three hash symbols)",
            "- Do NOT write headings as plain text - they MUST have the ## or ### prefix",
            "",
            "Required format for EACH H2 heading:",
            "## [H2 Heading Text Here]",
            "*Description:* [2-3 sentences explaining what this section covers and why it matters]",
            "*Key Points:*",
            "- [Specific talking point 1]",
            "- [Specific talking point 2]",
            "- [Specific talking point 3-5]",
            "*Keywords to incorporate:* [keyword1, keyword2, keyword3]",
            "",
            "Required format for EACH H3 sub-heading:",
            "### [H3 Sub-heading Text Here]",
            "*Description:* [1-2 sentences on what this subsection covers]",
            "*Key Points:*",
            "- [Specific point 1]",
            "- [Specific point 2]",
            "",
            "Structure the outline hierarchically with 4-6 H2 sections, each with 2-4 H3 subsections.",
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

    def _build_entity_guidance(self, competitor_analysis: Dict[str, Any]) -> str:
        """
        Build entity guidance section for generative search optimization.

        Args:
            competitor_analysis: Competitor analysis data with entity information

        Returns:
            Markdown formatted guidance string
        """
        guidance_lines = []

        # Get entity data
        entity_stats = competitor_analysis.get("entity_analysis", {})
        common_entities = competitor_analysis.get("common_entities", {})

        if entity_stats:
            guidance_lines.append("\n**Entity Coverage Benchmarks (AI Citation Signals):**")
            guidance_lines.append(
                f"- Top competitors mention an average of **{entity_stats.get('avg_orgs_mentioned', 0)} organizations/companies**"
            )
            guidance_lines.append(
                f"- Top competitors cite an average of **{entity_stats.get('avg_people_mentioned', 0)} people/experts**"
            )
            guidance_lines.append(
                "- IMPORTANT: AI models (ChatGPT, Perplexity, Claude) prioritize content that mentions authoritative entities"
            )

        if common_entities:
            guidance_lines.append("\n**Critical Entities to Mention (appear in multiple competitor articles):**")

            # Organizations
            if "ORG" in common_entities:
                org_list = [f"{ent} ({count}x)" for ent, count in common_entities["ORG"][:5]]
                if org_list:
                    guidance_lines.append(f"- **Organizations:** {', '.join(org_list)}")

            # People/Experts
            if "PERSON" in common_entities:
                person_list = [f"{ent} ({count}x)" for ent, count in common_entities["PERSON"][:5]]
                if person_list:
                    guidance_lines.append(f"- **Experts/Authors:** {', '.join(person_list)}")

            # Products/Technologies
            if "PRODUCT" in common_entities:
                product_list = [f"{ent} ({count}x)" for ent, count in common_entities["PRODUCT"][:5]]
                if product_list:
                    guidance_lines.append(f"- **Products/Technologies:** {', '.join(product_list)}")

            guidance_lines.append(
                "\n_Ensure the content outline includes sections that naturally incorporate these entities with proper context._"
            )

        return "\n".join(guidance_lines) if guidance_lines else ""
