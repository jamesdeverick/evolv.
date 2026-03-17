# --------------------------------------------
# Content Revision Tracker - Track Changes for Existing Content
# analysis/content_reviser.py
# --------------------------------------------

from typing import Dict, List, Any, Optional
import re
import difflib
from api.llm_client import get_llm_client


class ContentReviser:
    """Generate tracked-changes style content revisions for existing pages."""
    
    def __init__(self):
        """Initialize content reviser."""
        self.llm = get_llm_client()
    
    def analyze_and_propose_revisions(
        self,
        page_content: str,
        target_keyword: str,
        keyword_gaps: List[str],
        entity_gaps: List[str],
        competitor_analysis: Optional[Dict[str, Any]] = None,
        eeat_requirements: Optional[Dict[str, Any]] = None,
        url: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze existing content and propose specific inline revisions.
        
        Args:
            page_content: Current page content (HTML or text)
            target_keyword: Primary keyword
            keyword_gaps: Missing keywords that should be added
            entity_gaps: Missing entities/concepts
            competitor_analysis: Competitor data
            eeat_requirements: E-E-A-T requirements
            url: Page URL
            
        Returns:
            Dictionary with proposed revisions in tracked-changes format
        """
        if not self.llm.available:
            return {"error": "LLM not available"}
        
        # Build context for LLM
        context = self._build_analysis_context(
            keyword_gaps,
            entity_gaps,
            competitor_analysis,
            eeat_requirements
        )
        
        prompt = f"""You are a content editor creating SPECIFIC, ACTIONABLE revisions for an existing webpage.

**Page URL:** {url or 'Not provided'}
**Target Keyword:** {target_keyword}

**Current Page Content:**
{page_content[:3000]}

**Analysis Context:**
{context}

**Your Task:**
Identify 5-10 specific sections of the current content that need revision. For each:
1. Extract the EXACT original text (verbatim quote from the content above)
2. Write the suggested replacement text
3. Explain WHY this change improves SEO/LLM visibility
4. Assign priority (high/medium/low)

**CRITICAL INSTRUCTIONS:**
- The "original" text must be a direct quote from the content above
- Keep original text to 1-3 sentences max for focused changes
- Suggested text should be the same length or slightly longer
- Focus on: adding missing keywords naturally, incorporating entities, strengthening E-E-A-T signals, improving LLM-readability
- Do NOT suggest deleting large sections - only rewrites/additions
- Prioritize changes that have highest SEO/LLM impact

**CRITICAL JSON FORMATTING:**
- Return ONLY valid JSON - no markdown, no explanation, no preamble
- Do NOT use triple backticks or code fences
- Ensure all strings are properly escaped (use \\" for quotes inside strings)
- No trailing commas
- change_type must be one of: keyword_insertion, entity_addition, eeat_strengthening, clarity_improvement
- priority must be one of: high, medium, low

Return this exact JSON structure:
{{
  "revisions": [
    {{
      "section_name": "Introduction paragraph",
      "original_text": "Exact quote from current content",
      "suggested_text": "Improved version with keywords woven in naturally",
      "reason": "Specific reason explaining SEO/LLM benefit",
      "priority": "high",
      "change_type": "keyword_insertion"
    }}
  ],
  "overall_assessment": "Brief summary of content gaps",
  "quick_wins": ["Change 1 description", "Change 2 description"]
}}
"""
        
        try:
            result = self.llm.complete(prompt, temperature=0.3, max_tokens=2500)
            
            # Parse JSON with better error handling
            import json
            
            # Clean the response
            clean = result.strip()
            
            # Remove markdown code fences
            clean = re.sub(r'```json\s*', '', clean)
            clean = re.sub(r'\s*```', '', clean)
            
            # Remove any leading/trailing text before/after JSON
            # Find first { and last }
            start = clean.find('{')
            end = clean.rfind('}')
            
            if start == -1 or end == -1:
                print(f"No JSON found in LLM response")
                print(f"LLM Response: {result[:500]}")
                return {
                    "error": "LLM did not return JSON. Response started with: " + result[:100],
                    "revisions": [],
                    "overall_assessment": "Unable to parse LLM response",
                    "quick_wins": []
                }
            
            clean = clean[start:end+1]
            
            # Try to parse
            try:
                revisions = json.loads(clean)
            except json.JSONDecodeError as e:
                print(f"JSON Parse Error: {e}")
                print(f"Attempted to parse: {clean[:500]}")
                return {
                    "error": f"JSON parsing failed at position {e.pos}: {e.msg}. Check LLM response format.",
                    "revisions": [],
                    "overall_assessment": "Unable to parse LLM response - invalid JSON format",
                    "quick_wins": [],
                    "debug_response": clean[:1000]
                }
            
            # Validate structure
            if not isinstance(revisions, dict):
                return {
                    "error": "LLM returned invalid structure (not a dictionary)",
                    "revisions": [],
                    "overall_assessment": "Unable to generate revisions",
                    "quick_wins": []
                }
            
            if 'revisions' not in revisions:
                revisions['revisions'] = []
            
            # Add metadata
            revisions['url'] = url
            revisions['target_keyword'] = target_keyword
            revisions['total_revisions'] = len(revisions.get('revisions', []))
            
            return revisions
            
        except Exception as e:
            print(f"Revision analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Unexpected error: {str(e)}",
                "revisions": [],
                "overall_assessment": "Unable to generate revisions due to error",
                "quick_wins": []
            }
    
    def _build_analysis_context(
        self,
        keyword_gaps: List[str],
        entity_gaps: List[str],
        competitor_analysis: Optional[Dict[str, Any]],
        eeat_requirements: Optional[Dict[str, Any]]
    ) -> str:
        """Build context string for LLM about what's missing."""
        context_parts = []
        
        if keyword_gaps:
            context_parts.append(f"**Missing Keywords (should appear naturally):**\n{', '.join(keyword_gaps[:15])}")
        
        if entity_gaps:
            context_parts.append(f"\n**Missing Entities/Concepts (for LLM visibility):**\n{', '.join(entity_gaps[:10])}")
        
        if competitor_analysis:
            avg_wc = competitor_analysis.get('avg_word_count', 0)
            if avg_wc > 0:
                context_parts.append(f"\n**Competitor Benchmark:** Average word count is {avg_wc} words")
            
            common_headings = competitor_analysis.get('common_headings', [])
            if common_headings:
                context_parts.append(f"\n**Common Competitor Sections:** {', '.join(common_headings[:5])}")
        
        if eeat_requirements:
            trust_signals = eeat_requirements.get('trust_signals', [])
            if trust_signals:
                context_parts.append(f"\n**E-E-A-T Signals Needed:**\n{chr(10).join(['- ' + s for s in trust_signals[:4]])}")
        
        return "\n".join(context_parts) if context_parts else "No specific gaps identified"
    
    def generate_diff_html(self, original: str, suggested: str) -> str:
        """
        Generate HTML diff view for tracked changes.
        
        Args:
            original: Original text
            suggested: Suggested replacement
            
        Returns:
            HTML string showing inline diff
        """
        import difflib
        
        # Generate word-level diff for cleaner display
        original_words = original.split()
        suggested_words = suggested.split()
        
        diff = difflib.unified_diff(
            original_words,
            suggested_words,
            lineterm='',
            n=100  # Large context to show everything
        )
        
        # Build HTML with strikethrough for deletions, underline for additions
        html_parts = []
        html_parts.append('<div style="font-family: monospace; line-height: 1.8;">')
        
        # Original (deletions in red strikethrough)
        html_parts.append('<div style="margin-bottom: 10px;">')
        html_parts.append('<strong>ORIGINAL:</strong><br>')
        html_parts.append(f'<span style="color: #d32f2f; text-decoration: line-through;">{original}</span>')
        html_parts.append('</div>')
        
        # Suggested (additions in green)
        html_parts.append('<div>')
        html_parts.append('<strong>SUGGESTED:</strong><br>')
        html_parts.append(f'<span style="color: #388e3c; font-weight: 500;">{suggested}</span>')
        html_parts.append('</div>')
        
        html_parts.append('</div>')
        
        return "".join(html_parts)
    
    def export_revisions_markdown(self, revisions: Dict[str, Any]) -> str:
        """
        Export revisions as markdown document for editors.
        
        Args:
            revisions: Revisions dictionary from analyze_and_propose_revisions
            
        Returns:
            Markdown formatted document
        """
        if revisions.get('error'):
            return f"# Content Revisions\n\nError: {revisions['error']}"
        
        md_parts = []
        
        # Header
        md_parts.append("# Content Revision Tracker\n")
        md_parts.append(f"**URL:** {revisions.get('url', 'Not provided')}\n")
        md_parts.append(f"**Target Keyword:** {revisions.get('target_keyword', 'Not specified')}\n")
        md_parts.append(f"**Total Proposed Changes:** {revisions.get('total_revisions', 0)}\n")
        md_parts.append("\n---\n")
        
        # Overall assessment
        if revisions.get('overall_assessment'):
            md_parts.append(f"## Overall Assessment\n\n{revisions['overall_assessment']}\n\n")
        
        # Quick wins
        if revisions.get('quick_wins'):
            md_parts.append("## 🎯 Quick Wins (High Impact, Low Effort)\n\n")
            for win in revisions['quick_wins']:
                md_parts.append(f"- {win}\n")
            md_parts.append("\n")
        
        # Detailed revisions
        md_parts.append("## Proposed Revisions\n\n")
        
        for i, rev in enumerate(revisions.get('revisions', []), 1):
            priority_emoji = "🔥" if rev['priority'] == 'high' else "⚡" if rev['priority'] == 'medium' else "💡"
            
            md_parts.append(f"### {priority_emoji} Change #{i}: {rev.get('section_name', 'Unnamed')}\n\n")
            md_parts.append(f"**Priority:** {rev.get('priority', 'medium').upper()}\n")
            md_parts.append(f"**Type:** {rev.get('change_type', 'general').replace('_', ' ').title()}\n\n")
            
            md_parts.append("**ORIGINAL:**\n")
            md_parts.append(f"> {rev.get('original_text', '')}\n\n")
            
            md_parts.append("**SUGGESTED:**\n")
            md_parts.append(f"> {rev.get('suggested_text', '')}\n\n")
            
            md_parts.append(f"**REASON:** {rev.get('reason', 'Not specified')}\n\n")
            md_parts.append("---\n\n")
        
        return "".join(md_parts)
