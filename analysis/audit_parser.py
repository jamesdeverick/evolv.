# --------------------------------------------
# Audit Parser - Extract structured requirements from SEO audit briefs
# --------------------------------------------

import json
import re
from typing import Dict, Any, List, Optional
from api.llm_client import get_llm_client


class AuditParser:
    """
    Parse uploaded SEO audit briefs to extract structured, mandatory requirements
    that must be incorporated into generated content briefs.
    
    Turns a wall-of-text audit into a checklist of required elements.
    """

    def __init__(self):
        self.llm = get_llm_client()

    def parse_audit(self, audit_text: str) -> Dict[str, Any]:
        """
        Extract structured requirements from an audit document.
        
        Args:
            audit_text: Raw text of the uploaded audit
            
        Returns:
            Dict with mandatory elements the brief must include
        """
        if not audit_text or len(audit_text.strip()) < 100:
            return {
                "error": "Audit too short or empty",
                "has_audit": False
            }

        if not self.llm.available:
            return {
                "error": "LLM not available",
                "has_audit": False
            }

        # Truncate very long audits to fit context
        audit_excerpt = audit_text[:8000] if len(audit_text) > 8000 else audit_text

        prompt = f"""You are analyzing an SEO audit document. Extract ONLY the specific, actionable requirements that MUST be included in a content brief based on this audit.

AUDIT DOCUMENT:
{audit_excerpt}

Extract the following elements as JSON. Be strict - only include items EXPLICITLY mentioned in the audit. Do not invent or infer.

Return ONLY valid JSON in this exact format (no markdown, no explanation):

{{
  "required_citations": [
    "Specific analyst reports, studies, or third-party sources that MUST be cited (e.g., 'Gartner Magic Quadrant for Container Management 2025', 'Forrester Wave Q3 2025')"
  ],
  "required_product_mentions": [
    "Specific product/service names that MUST appear (e.g., 'SUSE Rancher Prime', 'SEAL tool')"
  ],
  "required_data_points": [
    "Specific statistics or data points to include (e.g., '75% of AI/ML workloads will run in containers')"
  ],
  "required_structural_elements": [
    "Required page components (e.g., 'Key Takeaways box near top', 'FAQ block', 'Freshness stamp')"
  ],
  "prioritized_recommendations": [
    {{"priority": 1, "recommendation": "Specific action from audit", "why": "Reason given in audit"}}
  ],
  "competitive_positioning": [
    "How to position vs specific named competitors (e.g., 'Contrast with Red Hat use of MQ Leader positioning')"
  ],
  "critical_gaps_to_fill": [
    "What's currently missing that MUST be added"
  ],
  "required_internal_links": [
    "Specific URLs or pages to link to"
  ],
  "customer_proof_points": [
    "Specific customer names/quotes to include (e.g., 'AKDB municipal case study', 'Aussie Broadband quote')"
  ],
  "tone_and_positioning_notes": [
    "Specific positioning guidance (e.g., 'Emphasize SUSE as only European-HQ major vendor')"
  ],
  "executive_summary": "One paragraph summary of what the audit says the brief must accomplish"
}}

IMPORTANT RULES:
- If a section has no items from the audit, use empty array []
- Do not paraphrase citations - keep exact names (e.g., "2025 Gartner Magic Quadrant for Container Management" not "Gartner report")
- Include specific dates, versions, quote sources when mentioned
- Prioritized recommendations should be in the ORDER the audit prioritizes them
- Do not add generic SEO best practices - only what the audit specifically calls out"""

        result = self.llm.complete(prompt, temperature=0.2, max_tokens=2500)

        if not result or result.startswith("Error"):
            return {
                "error": f"LLM failed to parse audit: {result}",
                "has_audit": False
            }

        # Parse JSON response with robust error handling
        parsed = self._extract_json(result)
        
        if parsed.get("error"):
            return {
                "error": parsed["error"],
                "raw_response": result[:500],
                "has_audit": False
            }

        parsed["has_audit"] = True
        parsed["audit_length"] = len(audit_text)
        
        # Add derived metadata
        parsed["total_requirements"] = self._count_requirements(parsed)
        
        return parsed

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Robust JSON extraction from LLM response."""
        # Strip markdown code fences
        cleaned = re.sub(r'^```(?:json)?\s*', '', text.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned)
        
        # Find JSON object boundaries
        try:
            start = cleaned.index('{')
            end = cleaned.rindex('}') + 1
            json_str = cleaned[start:end]
        except ValueError:
            return {"error": "No JSON object found in response"}
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try common fixes
            try:
                # Remove trailing commas
                fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                return json.loads(fixed)
            except json.JSONDecodeError as e2:
                return {
                    "error": f"JSON parse failed at pos {e2.pos}: {e2.msg}",
                    "preview": json_str[max(0, e2.pos-50):e2.pos+50] if hasattr(e2, 'pos') else ""
                }

    def _count_requirements(self, parsed: Dict[str, Any]) -> int:
        """Count total number of extracted requirements."""
        total = 0
        for key in [
            'required_citations',
            'required_product_mentions',
            'required_data_points',
            'required_structural_elements',
            'prioritized_recommendations',
            'competitive_positioning',
            'critical_gaps_to_fill',
            'required_internal_links',
            'customer_proof_points',
            'tone_and_positioning_notes'
        ]:
            if isinstance(parsed.get(key), list):
                total += len(parsed[key])
        return total

    def format_for_prompt(self, parsed_audit: Dict[str, Any]) -> str:
        """
        Format parsed audit into a strong, mandatory instruction block
        for the brief generation prompt.
        """
        if not parsed_audit.get("has_audit") or parsed_audit.get("error"):
            return ""

        lines = [
            "=" * 80,
            "🚨 CRITICAL: AUDIT REQUIREMENTS (MANDATORY - NON-NEGOTIABLE)",
            "=" * 80,
            "",
            "The following requirements are extracted from the client's audit brief.",
            "The generated content brief MUST address EVERY item below.",
            "These are not suggestions - they are mandatory inclusions.",
            "",
        ]

        # Executive summary
        if parsed_audit.get("executive_summary"):
            lines.extend([
                "**AUDIT OBJECTIVE:**",
                parsed_audit["executive_summary"],
                ""
            ])

        # Required citations - HIGHEST PRIORITY
        citations = parsed_audit.get("required_citations", [])
        if citations:
            lines.extend([
                "**⚠️ REQUIRED CITATIONS (MUST appear in brief - these are the KEY differentiators):**",
            ])
            for cite in citations:
                lines.append(f"  → {cite}")
            lines.append("")

        # Required product mentions
        products = parsed_audit.get("required_product_mentions", [])
        if products:
            lines.extend([
                "**⚠️ REQUIRED PRODUCT MENTIONS (MUST be named in the brief):**",
            ])
            for product in products:
                lines.append(f"  → {product}")
            lines.append("")

        # Required data points
        data_points = parsed_audit.get("required_data_points", [])
        if data_points:
            lines.extend([
                "**REQUIRED DATA/STATISTICS (must be included):**",
            ])
            for dp in data_points:
                lines.append(f"  → {dp}")
            lines.append("")

        # Customer proof points
        proof = parsed_audit.get("customer_proof_points", [])
        if proof:
            lines.extend([
                "**🚨 MANDATORY CUSTOMER PROOF POINTS (must be named in the brief, ideally in Section 4 or 12):**",
            ])
            for p in proof:
                lines.append(f"  → {p}")
            lines.append("")

        # Prioritized recommendations
        recs = parsed_audit.get("prioritized_recommendations", [])
        if recs:
            lines.extend([
                "**PRIORITIZED RECOMMENDATIONS (address in this ORDER in the outline):**",
            ])
            for rec in recs:
                priority = rec.get("priority", "?") if isinstance(rec, dict) else "?"
                recommendation = rec.get("recommendation", str(rec)) if isinstance(rec, dict) else str(rec)
                why = rec.get("why", "") if isinstance(rec, dict) else ""
                lines.append(f"  {priority}. {recommendation}")
                if why:
                    lines.append(f"     Why: {why}")
            lines.append("")

        # Structural requirements
        structural = parsed_audit.get("required_structural_elements", [])
        if structural:
            lines.extend([
                "**🚨 MANDATORY PAGE STRUCTURE ELEMENTS (must appear in Section 11 of brief AND be integrated into the outline):**",
            ])
            for elem in structural:
                lines.append(f"  → {elem}")
            lines.append("")

        # Critical gaps
        gaps = parsed_audit.get("critical_gaps_to_fill", [])
        if gaps:
            lines.extend([
                "**⚠️ CRITICAL GAPS TO FILL (currently missing, MUST be added):**",
            ])
            for gap in gaps:
                lines.append(f"  → {gap}")
            lines.append("")

        # Competitive positioning
        competitive = parsed_audit.get("competitive_positioning", [])
        if competitive:
            lines.extend([
                "**COMPETITIVE POSITIONING (how to position vs competitors):**",
            ])
            for pos in competitive:
                lines.append(f"  → {pos}")
            lines.append("")

        # Internal links
        links = parsed_audit.get("required_internal_links", [])
        if links:
            lines.extend([
                "**🚨 MANDATORY INTERNAL LINKS (must appear in Section 6 of brief):**",
            ])
            for link in links:
                lines.append(f"  → {link}")
            lines.append("")

        # Tone/positioning notes
        tone = parsed_audit.get("tone_and_positioning_notes", [])
        if tone:
            lines.extend([
                "**POSITIONING NOTES:**",
            ])
            for note in tone:
                lines.append(f"  → {note}")
            lines.append("")

        lines.extend([
            "=" * 80,
            "COMPLIANCE CHECKLIST - Before finalizing brief, verify:",
            "  ✓ Every required citation is incorporated into the outline",
            "  ✓ Every required product name is mentioned in appropriate H2/H3 sections",
            "  ✓ Every data point/statistic is included",
            "  ✓ Structural elements (Key Takeaways, FAQ, etc.) are specified",
            "  ✓ Recommendations are addressed in the priority order given",
            "=" * 80,
            ""
        ])

        return "\n".join(lines)

    def format_for_display(self, parsed_audit: Dict[str, Any]) -> str:
        """
        Format parsed audit for Streamlit display so user can verify what was extracted.
        Returns Markdown for st.markdown().
        """
        if not parsed_audit.get("has_audit"):
            return "⚠️ No audit was parsed."

        if parsed_audit.get("error"):
            return f"❌ Parse error: {parsed_audit['error']}"

        lines = [
            f"### 📋 Extracted {parsed_audit.get('total_requirements', 0)} requirements from audit\n"
        ]

        if parsed_audit.get("executive_summary"):
            lines.append(f"**Objective:** {parsed_audit['executive_summary']}\n")

        sections = [
            ("required_citations", "⚠️ Required Citations", "🔖"),
            ("required_product_mentions", "⚠️ Required Product Mentions", "📦"),
            ("required_data_points", "Required Data Points", "📊"),
            ("customer_proof_points", "Customer Proof Points", "👥"),
            ("prioritized_recommendations", "Prioritized Recommendations", "📋"),
            ("required_structural_elements", "Structural Requirements", "🏗️"),
            ("critical_gaps_to_fill", "Critical Gaps to Fill", "❌"),
            ("competitive_positioning", "Competitive Positioning", "⚔️"),
            ("required_internal_links", "Required Internal Links", "🔗"),
            ("tone_and_positioning_notes", "Positioning Notes", "🎯"),
        ]

        for key, label, emoji in sections:
            items = parsed_audit.get(key, [])
            if not items:
                continue
            lines.append(f"**{emoji} {label}** ({len(items)})")
            for item in items:
                if isinstance(item, dict):
                    priority = item.get("priority", "?")
                    rec = item.get("recommendation", "")
                    lines.append(f"  {priority}. {rec}")
                else:
                    lines.append(f"  - {item}")
            lines.append("")

        return "\n".join(lines)

    def verify_brief_compliance(
        self, 
        brief: str, 
        parsed_audit: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive check of brief against ALL audit requirement types.
        Returns detailed compliance report.
        """
        if not parsed_audit.get("has_audit"):
            return {"compliant": True, "message": "No audit to check against"}

        brief_lower = brief.lower()
        report = {
            "citations_included": [],
            "citations_missing": [],
            "products_included": [],
            "products_missing": [],
            "data_points_included": [],
            "data_points_missing": [],
            "structural_included": [],
            "structural_missing": [],
            "customer_proof_included": [],
            "customer_proof_missing": [],
            "internal_links_included": [],
            "internal_links_missing": [],
            "recommendations_included": [],
            "recommendations_missing": [],
            "gaps_addressed": [],
            "gaps_still_open": [],
            "compliance_score": 0,
            "total_checks": 0,
            "passed_checks": 0,
            "category_scores": {}
        }

        def fuzzy_match(text: str, target: str, min_terms: int = 2) -> bool:
            """Check if key terms from target appear in text."""
            # Extract meaningful terms (skip stopwords, short words)
            stopwords = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 
                        'these', 'those', 'about', 'into', 'over', 'must', 'should'}
            terms = [w.lower().strip('.,;:()[]"\'') 
                     for w in target.split() 
                     if len(w) > 3 and w.lower() not in stopwords]
            if not terms:
                return target.lower() in text
            matches = sum(1 for t in terms if t in text)
            return matches >= min(min_terms, len(terms))

        # 1. Check citations (require 2+ capitalized terms match)
        for citation in parsed_audit.get("required_citations", []):
            report["total_checks"] += 1
            key_terms = [w for w in citation.split() 
                        if len(w) > 3 and (w[0].isupper() or w.isdigit())]
            matches = sum(1 for t in key_terms if t.lower() in brief_lower)
            if matches >= 2 or (len(key_terms) == 1 and matches == 1):
                report["citations_included"].append(citation)
                report["passed_checks"] += 1
            else:
                report["citations_missing"].append(citation)

        # 2. Check product mentions (exact match)
        for product in parsed_audit.get("required_product_mentions", []):
            report["total_checks"] += 1
            if product.lower() in brief_lower:
                report["products_included"].append(product)
                report["passed_checks"] += 1
            else:
                report["products_missing"].append(product)

        # 3. Check data points (look for numbers/percentages)
        for data_point in parsed_audit.get("required_data_points", []):
            report["total_checks"] += 1
            # Extract distinctive elements - numbers and key nouns
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?%?', data_point)
            key_terms = [w for w in data_point.split() 
                        if len(w) > 5 and not w[0].isdigit()][:3]
            
            has_number = any(n in brief for n in numbers)
            has_context = any(t.lower() in brief_lower for t in key_terms)
            
            if has_number and has_context:
                report["data_points_included"].append(data_point)
                report["passed_checks"] += 1
            elif has_number or fuzzy_match(brief_lower, data_point, min_terms=3):
                report["data_points_included"].append(data_point)
                report["passed_checks"] += 1
            else:
                report["data_points_missing"].append(data_point)

        # 4. Check structural elements (keywords like "Key Takeaways", "FAQ", "freshness")
        structural_keywords = {
            'key takeaways': ['key takeaway', 'takeaways box', 'takeaways section'],
            'faq': ['faq', 'frequently asked', 'faqs'],
            'freshness stamp': ['updated', 'last updated', 'freshness', 'refresh', 'date stamp'],
            'schema': ['schema', 'json-ld', 'structured data'],
            'cta': ['call-to-action', 'cta', 'call to action'],
            'meta description': ['meta description', 'meta desc'],
            'internal link': ['internal link', 'internal linking'],
            'author': ['author bio', 'byline', 'author box'],
            'callout': ['callout', 'call-out', 'highlight box'],
        }
        for element in parsed_audit.get("required_structural_elements", []):
            report["total_checks"] += 1
            element_lower = element.lower()
            
            # Direct match first
            found = False
            if fuzzy_match(brief_lower, element, min_terms=2):
                found = True
            else:
                # Check keyword categories
                for category, keywords in structural_keywords.items():
                    if category in element_lower:
                        if any(kw in brief_lower for kw in keywords):
                            found = True
                            break
            
            if found:
                report["structural_included"].append(element)
                report["passed_checks"] += 1
            else:
                report["structural_missing"].append(element)

        # 5. Check customer proof points (name match)
        for proof in parsed_audit.get("customer_proof_points", []):
            report["total_checks"] += 1
            # Extract likely proper nouns (capitalized words)
            names = [w for w in proof.split() 
                    if len(w) > 3 and w[0].isupper() 
                    and w not in {'The', 'Their', 'This', 'That', 'From'}]
            
            if any(name.lower() in brief_lower for name in names):
                report["customer_proof_included"].append(proof)
                report["passed_checks"] += 1
            else:
                report["customer_proof_missing"].append(proof)

        # 6. Check internal links (URL/path match)
        for link in parsed_audit.get("required_internal_links", []):
            report["total_checks"] += 1
            # Look for URL paths or distinctive slug words
            import re
            paths = re.findall(r'/[a-z0-9\-/]+', link.lower())
            
            if paths and any(p in brief.lower() for p in paths if len(p) > 5):
                report["internal_links_included"].append(link)
                report["passed_checks"] += 1
            elif fuzzy_match(brief_lower, link, min_terms=2):
                report["internal_links_included"].append(link)
                report["passed_checks"] += 1
            else:
                report["internal_links_missing"].append(link)

        # 7. Check prioritized recommendations (fuzzy match)
        for rec in parsed_audit.get("prioritized_recommendations", []):
            report["total_checks"] += 1
            rec_text = rec.get("recommendation", "") if isinstance(rec, dict) else str(rec)
            
            if fuzzy_match(brief_lower, rec_text, min_terms=3):
                report["recommendations_included"].append(rec_text)
                report["passed_checks"] += 1
            else:
                report["recommendations_missing"].append(rec_text)

        # 8. Check critical gaps (fuzzy match - gap is "addressed" if brief mentions the topic)
        for gap in parsed_audit.get("critical_gaps_to_fill", []):
            report["total_checks"] += 1
            if fuzzy_match(brief_lower, gap, min_terms=3):
                report["gaps_addressed"].append(gap)
                report["passed_checks"] += 1
            else:
                report["gaps_still_open"].append(gap)

        # Calculate overall score
        if report["total_checks"] > 0:
            report["compliance_score"] = round(
                (report["passed_checks"] / report["total_checks"]) * 100, 1
            )
        else:
            report["compliance_score"] = 100

        # Category-level scores for insight
        def cat_score(included, missing):
            total = len(included) + len(missing)
            return round((len(included) / total) * 100, 1) if total > 0 else None

        report["category_scores"] = {
            "citations": cat_score(report["citations_included"], report["citations_missing"]),
            "products": cat_score(report["products_included"], report["products_missing"]),
            "data_points": cat_score(report["data_points_included"], report["data_points_missing"]),
            "structural": cat_score(report["structural_included"], report["structural_missing"]),
            "customer_proof": cat_score(report["customer_proof_included"], report["customer_proof_missing"]),
            "internal_links": cat_score(report["internal_links_included"], report["internal_links_missing"]),
            "recommendations": cat_score(report["recommendations_included"], report["recommendations_missing"]),
            "gaps": cat_score(report["gaps_addressed"], report["gaps_still_open"]),
        }

        report["compliant"] = report["compliance_score"] >= 80

        return report
