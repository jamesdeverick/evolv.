# --------------------------------------------
# Content Brief Creator - Enhanced with Advanced Analysis
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
        competitor_analysis: Optional[Dict[str, Any]] = None,
        advanced_analysis: Optional[Dict[str, Any]] = None,
        page_type: str = "blog_post",
        content_revisions: Optional[Dict[str, Any]] = None
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
            competitor_analysis: Competitor analysis data
            advanced_analysis: Advanced SEO analysis (E-E-A-T, snippets, etc.)
            page_type: Type of page (blog_post, product_page, landing_page, etc.)

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

        # Note: Revision insights are handled in _append_advanced_analysis

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
            "**LLM's Previous Audit Analysis (CRITICAL - USE THIS EXACT STRUCTURE):**",
            llm_analysis or "[No previous LLM analysis available]",
            "",
            "=" * 80,
            "CRITICAL INSTRUCTION - READ CAREFULLY:",
            "=" * 80,
            "The analysis above contains a 'Recommended H2/H3 outline' section.",
            "You MUST copy that EXACT outline structure into section 3 of your content brief.",
            "",
            "DO NOT create new headings. DO NOT reorganize. DO NOT rename.",
            "COPY the H2 and H3 headings EXACTLY as they appear in the 'Recommended H2/H3 outline' above.",
            "",
            "For each heading from that outline, add:",
            "- Description: (2-3 sentences)",
            "- Key Points: (bullet list)",
            "- Keywords to incorporate: (list)",
            "",
            "Example: If the outline says 'H2: Best Practices for EKS Cost Optimization'",
            "You write: 'Best Practices for EKS Cost Optimization (H2)'",
            "Then add the Description, Key Points, and Keywords.",
            "=" * 80,
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

        # Note: Revision insights are appended after brief generation
        # in _append_advanced_analysis to ensure they appear in final output

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
        ])

        prompt = "\n".join(prompt_lines)

        # Generate brief
        result = self.llm.complete(
            prompt,
            temperature=0.7,
            max_tokens=2000
        )

        if not result or result.startswith("Error"):
            return result or "No valid response from LLM for content brief creation."

        # Clean up formatting and enforce markdown headings
        clean = self._enforce_markdown_headings(result)

        if not clean.strip():
            return "No valid response from LLM for content brief creation."

        # Store core brief BEFORE advanced sections are appended
        # Use session state via a module-level flag so it survives Streamlit reruns
        self._core_brief_markdown = clean
        # Also store in a class-level variable as backup
        ContentBriefCreator._last_core_brief = clean

        # ADD ADVANCED ANALYSIS SECTIONS AND REVISION INSIGHTS
        if advanced_analysis or content_revisions:
            clean = self._append_advanced_analysis(clean, advanced_analysis, content_revisions)

        return clean

    def _append_advanced_analysis(
        self,
        brief: str,
        adv: Optional[Dict[str, Any]],
        content_revisions: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Append advanced analysis sections and revision insights to the content brief.

        Args:
            brief: Existing brief content
            adv: Advanced analysis dictionary
            content_revisions: Content revision tracker data

        Returns:
            Enhanced brief with advanced sections and revision insights
        """
        additions = []

        # E-E-A-T Requirements
        if adv and adv.get('eeat_signals') and not adv['eeat_signals'].get('error'):
            eeat = adv['eeat_signals']
            additions.append("\n\n---\n\n## 🏆 E-E-A-T Requirements (CRITICAL)\n")

            if eeat.get('is_ymyl'):
                additions.append(f"\n⚠️ **YMYL TOPIC ({eeat.get('ymyl_category')})** - E-E-A-T is make-or-break for rankings!\n")

            if eeat.get('trust_signals'):
                additions.append("\n### Required Trust Signals:\n")
                additions.append("_Google expects these specific trust indicators for this topic:_\n")
                for signal in eeat['trust_signals'][:6]:
                    additions.append(f"- {signal}\n")

            if eeat.get('expertise_needed'):
                additions.append("\n### Expertise Requirements:\n")
                additions.append("_Content must demonstrate:_\n")
                for req in eeat['expertise_needed'][:4]:
                    additions.append(f"- {req}\n")

            if eeat.get('authority_needed'):
                additions.append("\n### Authority Signals:\n")
                for auth in eeat['authority_needed'][:3]:
                    additions.append(f"- {auth}\n")

            if eeat.get('priority_actions'):
                additions.append("\n### ⚡ PRIORITY ACTIONS:\n")
                for action in eeat['priority_actions'][:3]:
                    additions.append(f"**{action}**\n")

        # Featured Snippet Opportunities
        if adv and adv.get('featured_snippets') and not adv['featured_snippets'].get('error'):
            snippets = adv['featured_snippets']
            high_opps = [o for o in snippets.get('high_opportunity_queries', [])
                        if o.get('win_probability') in ['High', 'Medium']][:4]

            if high_opps:
                additions.append("\n\n## ⭐ Featured Snippet Opportunities\n")
                additions.append("_Format content to win these top-of-page placements:_\n")

                for opp in high_opps:
                    additions.append(f"\n### Target Query: \"{opp.get('query')}\"\n")
                    additions.append(f"**Win Probability:** {opp.get('win_probability')}\n")
                    additions.append(f"**Format Required:** {opp.get('snippet_type').title()}\n")
                    additions.append(f"**Optimization Guide:** {opp.get('optimization_guide')}\n")

            if snippets.get('quick_wins'):
                additions.append("\n### 🎯 Quick Wins:\n")
                for win in snippets['quick_wins'][:3]:
                    additions.append(f"- {win}\n")

        # Critical Content Gaps
        if adv and adv.get('content_gaps') and not adv['content_gaps'].get('error'):
            gaps = adv['content_gaps']
            critical = [g for g in gaps.get('critical_gaps', []) if g.get('priority') in ['High', 'Medium']][:6]

            if critical:
                additions.append("\n\n## ❌ Critical Topics to Cover\n")
                additions.append("_Competitors cover these but you don't - HIGH PRIORITY to add:_\n")

                for gap in critical:
                    additions.append(f"\n### {gap.get('missing_topic')}\n")
                    additions.append(f"**Covered by:** {gap.get('covered_by')}\n")
                    additions.append(f"**Priority:** {gap.get('priority')}\n")
                    additions.append(f"**What to include:** {gap.get('content_suggestion')}\n")

            # Add unique angles section
            if gaps.get('your_unique_angles'):
                additions.append("\n### 🎁 Your Competitive Advantages\n")
                additions.append("_Leverage these unique angles that competitors are missing:_\n")
                for angle in gaps['your_unique_angles'][:4]:
                    additions.append(f"- ✅ {angle}\n")

            # Add blue ocean opportunities
            if gaps.get('blue_ocean_opportunities'):
                additions.append("\n### 🌊 Blue Ocean Opportunities\n")
                additions.append("_Topics NO competitor covers well - easy wins:_\n")
                for opp in gaps['blue_ocean_opportunities'][:3]:
                    additions.append(f"- 💎 {opp}\n")

        # User Intent Deep Dive
        if adv and adv.get('search_intent') and not adv['search_intent'].get('error'):
            intent = adv['search_intent']
            additions.append("\n\n## 🎯 User Intent & Content Expectations\n")

            additions.append(f"\n**What users actually need:** {intent.get('primary_user_need')}\n")
            additions.append(f"\n**User journey stage:** {intent.get('user_journey_stage')}\n")
            additions.append(f"**User sophistication:** {intent.get('sophistication_level')}\n")
            additions.append(f"**Expected content format:** {intent.get('expected_content_format')}\n")
            additions.append(f"**Recommended tone:** {intent.get('content_tone')}\n")
            additions.append(f"**Recommended length:** {intent.get('content_length_recommendation')}\n")

            if intent.get('user_pain_points'):
                additions.append("\n### 😰 User Pain Points to Address:\n")
                for pain in intent['user_pain_points'][:5]:
                    additions.append(f"- {pain}\n")

            if intent.get('must_include_elements'):
                additions.append("\n### ✅ Must Include (Non-Negotiable):\n")
                for element in intent['must_include_elements'][:6]:
                    additions.append(f"- {element}\n")

            if intent.get('success_criteria'):
                additions.append(f"\n**User success criteria:** {intent.get('success_criteria')}\n")

        # Topical Authority
        if adv and adv.get('topical_authority') and not adv['topical_authority'].get('error'):
            ta = adv['topical_authority']

            if ta.get('authority_score'):
                additions.append(f"\n\n## 📊 Topical Authority Score: {ta.get('authority_score')}/100\n")

            if ta.get('missing_subtopics'):
                additions.append("\n### Missing Subtopics (Add These):\n")
                for topic in ta['missing_subtopics'][:5]:
                    additions.append(f"- {topic}\n")

            if ta.get('depth_issues'):
                additions.append("\n### Content Depth Issues:\n")
                for issue in ta['depth_issues'][:4]:
                    additions.append(f"- {issue}\n")

            if ta.get('semantic_gaps'):
                additions.append("\n### Entities/Concepts to Mention:\n")
                for gap in ta['semantic_gaps'][:6]:
                    additions.append(f"- {gap}\n")

        # Add Content Revision Insights
        if content_revisions and not content_revisions.get('error'):
            print("[DEBUG] Adding revision insights to brief...")
            additions.append("\n\n---\n\n## 📝 Content Revision Insights (From Existing Page Analysis)\n")

            # Overall assessment
            if content_revisions.get('overall_assessment'):
                additions.append(f"\n**Overall Assessment:** {content_revisions['overall_assessment']}\n")

            # Quick wins
            if content_revisions.get('quick_wins'):
                additions.append("\n### 🎯 Quick Wins (High Impact, Easy Implementation)\n")
                for win in content_revisions['quick_wins']:
                    additions.append(f"- {win}\n")

            # High priority changes
            revisions = content_revisions.get('revisions', [])
            high_priority = [r for r in revisions if r.get('priority') == 'high']

            if high_priority:
                additions.append("\n### 🔥 High Priority Changes Identified\n")
                additions.append("_These changes will have the biggest impact on SEO/LLM visibility:_\n\n")
                for rev in high_priority[:5]:
                    additions.append(f"**{rev.get('section_name', 'Unnamed')}:**\n")
                    additions.append(f"- **Original:** {rev.get('original_text', '')[:100]}...\n")
                    additions.append(f"- **Suggested:** {rev.get('suggested_text', '')[:100]}...\n")
                    additions.append(f"- **Reason:** {rev.get('reason', 'Not specified')}\n\n")

            # Medium priority
            medium_priority = [r for r in revisions if r.get('priority') == 'medium']
            if medium_priority:
                additions.append("\n### ⚡ Medium Priority Improvements\n")
                for rev in medium_priority[:3]:
                    additions.append(f"- **{rev.get('section_name')}:** {rev.get('reason')}\n")

            # Gap summary
            if revisions:
                keyword_changes = sum(1 for r in revisions if r.get('change_type') == 'keyword_insertion')
                entity_changes = sum(1 for r in revisions if r.get('change_type') == 'entity_addition')
                eeat_changes = sum(1 for r in revisions if r.get('change_type') == 'eeat_strengthening')

                gap_parts = []
                if keyword_changes > 0:
                    gap_parts.append(f"{keyword_changes} keyword gaps")
                if entity_changes > 0:
                    gap_parts.append(f"{entity_changes} missing entities")
                if eeat_changes > 0:
                    gap_parts.append(f"{eeat_changes} E-E-A-T weaknesses")

                if gap_parts:
                    additions.append(f"\n**Gap Summary:** {', '.join(gap_parts)} identified\n")

            additions.append("\n_The content outline above should address these gaps and incorporate the suggested improvements._\n")
            print(f"[DEBUG] Revision insights section added: {len(''.join(additions[-20:]))} characters")

        # Append all additions to the brief
        if additions:
            brief += "".join(additions)

        return brief

    def _enforce_markdown_headings(self, text: str) -> str:
        """
        Aggressively enforce markdown heading syntax using multiple regex patterns.
        """
        import re

        # Pattern 1: Any line followed by "Description:" (with any whitespace/formatting)
        text = re.sub(
            r'\n([A-Z][^\n#]{5,80})\s*\n\s*(?:\*\s*)?Description:',
            r'\n## \1\nDescription:',
            text,
            flags=re.IGNORECASE
        )

        # Pattern 2: Any line followed by "Key Points:"
        text = re.sub(
            r'\n([A-Z][^\n#]{5,80})\s*\n\s*(?:\*\s*)?Key Points:',
            r'\n## \1\nKey Points:',
            text,
            flags=re.IGNORECASE
        )

        # Pattern 3: Indented lines (subsections) followed by Description:
        text = re.sub(
            r'\n\s{2,}([A-Z][^\n#]{5,80})\s*\n\s*(?:\*\s*)?Description:',
            r'\n### \1\nDescription:',
            text,
            flags=re.IGNORECASE
        )

        # Pattern 4: Common section names that should always be H2
        common_sections = [
            'Introduction', 'Overview', 'Conclusion', 'Summary',
            'Benefits', 'Advantages', 'Challenges', 'Solutions',
            'Best Practices', 'Strategies', 'Examples', 'Use Cases',
            'Getting Started', 'Implementation', 'Deployment'
        ]

        for section in common_sections:
            # Only add ## if it's on its own line and doesn't already have #
            text = re.sub(
                rf'\n({section})\s*\n',
                rf'\n## \1\n',
                text,
                flags=re.IGNORECASE
            )

        # Clean up any double ## that might have been added
        text = re.sub(r'\n##\s*##\s*', '\n## ', text)
        text = re.sub(r'\n###\s*###\s*', '\n### ', text)

        return text

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

    def _build_revision_insights(self, content_revisions: Dict[str, Any]) -> str:
        """
        Build revision insights section from tracked changes analysis.

        Args:
            content_revisions: Content revision data from ContentReviser

        Returns:
            Markdown formatted revision insights string
        """
        insights_lines = []

        # Overall assessment
        if content_revisions.get('overall_assessment'):
            insights_lines.append(f"\n**Overall Assessment:** {content_revisions['overall_assessment']}")

        # Quick wins
        if content_revisions.get('quick_wins'):
            insights_lines.append("\n**Quick Wins (High Impact, Easy Implementation):**")
            for win in content_revisions['quick_wins']:
                insights_lines.append(f"- {win}")

        # Priority changes breakdown
        revisions = content_revisions.get('revisions', [])
        if revisions:
            high_priority = [r for r in revisions if r.get('priority') == 'high']
            medium_priority = [r for r in revisions if r.get('priority') == 'medium']

            if high_priority:
                insights_lines.append("\n**High Priority Changes Identified:**")
                for rev in high_priority[:5]:
                    insights_lines.append(
                        f"- **{rev.get('section_name', 'Unnamed')}:** {rev.get('reason', 'No reason provided')}"
                    )

            if medium_priority:
                insights_lines.append("\n**Medium Priority Improvements:**")
                for rev in medium_priority[:3]:
                    insights_lines.append(
                        f"- {rev.get('section_name', 'Unnamed')}: {rev.get('reason', 'No reason provided')}"
                    )

        # Key gaps summary
        if revisions:
            keyword_changes = sum(1 for r in revisions if r.get('change_type') == 'keyword_insertion')
            entity_changes = sum(1 for r in revisions if r.get('change_type') == 'entity_addition')
            eeat_changes = sum(1 for r in revisions if r.get('change_type') == 'eeat_strengthening')

            gap_summary = []
            if keyword_changes > 0:
                gap_summary.append(f"{keyword_changes} keyword gaps")
            if entity_changes > 0:
                gap_summary.append(f"{entity_changes} missing entities")
            if eeat_changes > 0:
                gap_summary.append(f"{eeat_changes} E-E-A-T weaknesses")

            if gap_summary:
                insights_lines.append(f"\n**Gap Summary:** {', '.join(gap_summary)} identified")

        insights_lines.append(
            "\n_The content outline should address these gaps and incorporate the suggested improvements._"
        )

        return "\n".join(insights_lines) if insights_lines else ""

    def to_gap_record_json(
        self,
        brief_markdown: str,
        keyword: str,
        client_url: str,
        page_type: str = "blog_post",
        related_keywords: Optional[List[str]] = None,
        content_revisions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert a generated brief to a structured JSON gap record
        suitable for POSTing to Optimizely Opal webhook.

        Args:
            brief_markdown: The full generated brief in markdown
            keyword: Primary keyword
            client_url: Existing page URL (empty if new page)
            page_type: Type of page
            related_keywords: Secondary keywords
            content_revisions: Content revision data if existing page

        Returns:
            Structured dict matching the agreed Opal gap record schema
        """
        import re

        # Determine action type
        action_type = "optimise_existing_page" if client_url else "create_new_page"

        # Extract title from brief — try multiple patterns in order of reliability
        title = keyword  # default fallback

        # 1. Try H1 markdown heading
        h1_match = re.search(r'^#\s+(.+)$', brief_markdown, re.MULTILINE)
        if h1_match:
            candidate = h1_match.group(1).strip()
            # Strip common LLM prefixes/suffixes
            candidate = re.sub(r'^Title:\s*', '', candidate, flags=re.IGNORECASE)
            candidate = re.sub(r'\s*\(SEO Optimized\)\s*$', '', candidate, flags=re.IGNORECASE)
            candidate = re.sub(r'\s*\(SEO-Optimized\)\s*$', '', candidate, flags=re.IGNORECASE)
            title = candidate.strip()
        else:
            # 2. Try line immediately after "Content Title" or "1. Content Title" label
            title_section_match = re.search(
                r'(?:^1\.\s*)?Content Title[^\n]*\n+[-*]?\s*(.+)',
                brief_markdown,
                re.IGNORECASE | re.MULTILINE
            )
            if title_section_match:
                candidate = title_section_match.group(1).strip().lstrip('#-* ').strip()
                if candidate and not candidate.endswith(':') and len(candidate) > 10:
                    title = candidate

            # 3. Try "Title:" line in On-page SEO elements section
            if title == keyword:
                seo_title_match = re.search(
                    r'[-*]\s*Title:\s*(.+)',
                    brief_markdown,
                    re.IGNORECASE
                )
                if seo_title_match:
                    candidate = seo_title_match.group(1).strip()
                    if candidate and not candidate.endswith(':') and len(candidate) > 10:
                        title = candidate

        # Extract LLM anchor sentence if present
        anchor_match = re.search(
            r'(?:LLM anchor sentence|anchor sentence)[:\s]*["\*]?([^"\n*]{40,200})["\*]?',
            brief_markdown,
            re.IGNORECASE
        )
        llm_anchor_sentence = anchor_match.group(1).strip() if anchor_match else ""

        # Extract meta description if present
        meta_match = re.search(
            r'(?:meta description)[:\s]*["\*]?([^"\n*]{50,160})["\*]?',
            brief_markdown,
            re.IGNORECASE
        )
        meta_description = meta_match.group(1).strip() if meta_match else ""

        # Build recommended revisions array for existing pages
        recommended_revisions = []
        if content_revisions and not content_revisions.get('error'):
            for rev in content_revisions.get('revisions', []):
                recommended_revisions.append({
                    "section": rev.get('section_name', ''),
                    "original": rev.get('original_text', ''),
                    "suggested": rev.get('suggested_text', ''),
                    "reason": rev.get('reason', ''),
                    "priority": rev.get('priority', 'medium'),
                    "change_type": rev.get('change_type', '')
                })

        # page_type is passed in directly from UI selection — no URL inference

        # Use core brief only (without advanced analysis sections) for Opal payload
        # Falls back through: instance var -> class-level var -> full brief_markdown passed in
        opal_brief = (
            getattr(self, '_core_brief_markdown', None)
            or getattr(ContentBriefCreator, '_last_core_brief', None)
            or brief_markdown
        )

        # Build the gap record
        gap_record = {
            "schema_version": "1.0",
            "action_type": action_type,
            "existing_page_url": client_url or None,
            "page_type": page_type,
            "primary_query": keyword,
            "secondary_queries": related_keywords or [],
            "title": title,
            "meta_description": meta_description,
            "llm_anchor_sentence": llm_anchor_sentence,
            "recommended_revisions": recommended_revisions,
            "brief_markdown": opal_brief
        }

        return gap_record

    def export_as_json(
        self,
        brief_markdown: str,
        keyword: str,
        client_url: str,
        page_type: str = "blog_post",
        related_keywords: Optional[List[str]] = None,
        content_revisions: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export brief as a JSON string ready for Opal webhook POST.

        Returns:
            JSON string
        """
        import json
        gap_record = self.to_gap_record_json(
            brief_markdown=brief_markdown,
            keyword=keyword,
            client_url=client_url,
            page_type=page_type,
            related_keywords=related_keywords,
            content_revisions=content_revisions
        )
        return json.dumps(gap_record, indent=2, ensure_ascii=False)

    def post_to_opal(
        self,
        brief_markdown: str,
        keyword: str,
        client_url: str,
        webhook_url: str,
        bearer_token: str,
        auth_header_name: str = "Authorization",
        page_type: str = "blog_post",
        related_keywords: Optional[List[str]] = None,
        content_revisions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        POST the gap record JSON directly to the Opal webhook.

        Args:
            brief_markdown: The full generated brief in markdown
            keyword: Primary keyword
            client_url: Existing page URL (empty if new page)
            webhook_url: Opal-generated webhook URL
            bearer_token: Secret token for auth header
            auth_header_name: Auth header name (default: Authorization)
            page_type: Type of page
            related_keywords: Secondary keywords
            content_revisions: Content revision data if existing page

        Returns:
            Dict with success bool, status_code, and response text
        """
        import requests
        import json

        gap_record = self.to_gap_record_json(
            brief_markdown=brief_markdown,
            keyword=keyword,
            client_url=client_url,
            page_type=page_type,
            related_keywords=related_keywords,
            content_revisions=content_revisions
        )

        headers = {
            auth_header_name: f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                webhook_url,
                headers=headers,
                data=json.dumps(gap_record, ensure_ascii=False),
                timeout=30
            )
            return {
                "success": response.status_code in (200, 201, 202),
                "status_code": response.status_code,
                "response": response.text
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "status_code": None,
                "response": str(e)
            }
