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
                "**REQUIRED CUSTOMER PROOF POINTS:**",
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
                "**REQUIRED PAGE STRUCTURE ELEMENTS:**",
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
                "**REQUIRED INTERNAL LINKS:**",
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
        Check whether the generated brief actually addresses the audit requirements.
        Returns compliance report.
        """
        if not parsed_audit.get("has_audit"):
            return {"compliant": True, "message": "No audit to check against"}

        brief_lower = brief.lower()
        report = {
            "citations_included": [],
            "citations_missing": [],
            "products_included": [],
            "products_missing": [],
            "compliance_score": 0,
            "total_checks": 0,
            "passed_checks": 0
        }

        # Check citations
        for citation in parsed_audit.get("required_citations", []):
            report["total_checks"] += 1
            # Extract key words from citation for fuzzy match
            key_terms = [w for w in citation.split() if len(w) > 4 and w[0].isupper()]
            if any(term.lower() in brief_lower for term in key_terms[:3]):
                report["citations_included"].append(citation)
                report["passed_checks"] += 1
            else:
                report["citations_missing"].append(citation)

        # Check product mentions
        for product in parsed_audit.get("required_product_mentions", []):
            report["total_checks"] += 1
            # Check for exact product name (case insensitive)
            if product.lower() in brief_lower:
                report["products_included"].append(product)
                report["passed_checks"] += 1
            else:
                report["products_missing"].append(product)

        if report["total_checks"] > 0:
            report["compliance_score"] = round(
                (report["passed_checks"] / report["total_checks"]) * 100, 1
            )
        else:
            report["compliance_score"] = 100

        report["compliant"] = report["compliance_score"] >= 80

        return report
