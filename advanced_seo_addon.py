# Page Type & Wireframe System
# page_optimizer.py

from typing import Dict, List, Optional, Any
from api.llm_client import get_llm_client


class PageTypeManager:
    """Manages different page types and their content requirements."""
    
    PAGE_TYPES = {
        "blog_post": {
            "name": "Blog Post / Article",
            "description": "Informational content, thought leadership, news",
            "typical_length": "1,500-2,500 words",
            "key_elements": [
                "Compelling headline with keyword",
                "Author bio and credentials (E-E-A-T)",
                "Table of contents for long posts",
                "Engaging introduction with hook",
                "Subheadings (H2/H3) every 300-400 words",
                "Images, charts, or infographics",
                "Internal links to related content",
                "CTA (subscribe, download, read more)",
                "Social sharing buttons",
                "Related posts section"
            ],
            "seo_focus": "Long-tail keywords, featured snippets, topical authority"
        },
        "product_page": {
            "name": "Product Page (Commercial)",
            "description": "Selling a specific product or offering",
            "typical_length": "800-1,500 words",
            "key_elements": [
                "Product name with primary keyword in H1",
                "High-quality product images/video",
                "Clear pricing and availability",
                "Product description (benefits > features)",
                "Technical specifications",
                "Customer reviews and ratings",
                "Trust signals (guarantees, certifications)",
                "Strong CTA (Add to Cart, Buy Now)",
                "Related/alternative products",
                "FAQ section",
                "Schema markup (Product, Review, Offer)"
            ],
            "seo_focus": "Commercial keywords, product schema, conversion"
        },
        "service_page": {
            "name": "Service Page",
            "description": "Describing and selling a service offering",
            "typical_length": "1,000-2,000 words",
            "key_elements": [
                "Service name with keyword in H1",
                "Clear value proposition",
                "Service description and process",
                "Benefits and outcomes",
                "Pricing options (if applicable)",
                "Case studies or examples",
                "Client testimonials",
                "Team/expert credentials (E-E-A-T)",
                "Strong CTA (Request Quote, Schedule Consultation)",
                "FAQ section"
            ],
            "seo_focus": "Service keywords, local SEO, trust signals"
        },
        "landing_page": {
            "name": "Landing Page (Campaign)",
            "description": "Focused conversion page for specific campaign",
            "typical_length": "500-1,200 words",
            "key_elements": [
                "Benefit-driven headline",
                "Hero image or video",
                "Clear value proposition",
                "Social proof (testimonials, logos, stats)",
                "Feature/benefit bullets",
                "Lead capture form",
                "Single, prominent CTA",
                "No navigation distractions",
                "Trust badges/security"
            ],
            "seo_focus": "Conversion optimization, paid search quality"
        },
        "guide_tutorial": {
            "name": "Guide / Tutorial",
            "description": "How-to content, step-by-step instructions",
            "typical_length": "2,000-4,000 words",
            "key_elements": [
                "Clear, keyword-rich title",
                "Estimated time to complete",
                "Required materials/prerequisites",
                "Table of contents",
                "Step-by-step instructions with numbers",
                "Screenshots or diagrams for each step",
                "Pro tips or warnings",
                "Video walkthrough (if applicable)",
                "FAQ section",
                "Related guides",
                "Schema markup (HowTo)"
            ],
            "seo_focus": "How-to keywords, featured snippets, video"
        },
        "comparison": {
            "name": "Comparison / vs Page",
            "description": "Comparing products, services, or solutions",
            "typical_length": "1,500-2,500 words",
            "key_elements": [
                "Clear comparison title (X vs Y)",
                "Summary/verdict upfront",
                "Side-by-side comparison table",
                "Detailed pros/cons for each option",
                "Use case recommendations",
                "Pricing comparison",
                "Expert opinion/recommendation",
                "User reviews or ratings",
                "CTA for each option",
                "FAQ section"
            ],
            "seo_focus": "Comparison keywords, decision-stage content"
        }
    }
    
    @classmethod
    def get_page_type_info(cls, page_type_key: str) -> Dict[str, Any]:
        """Get information about a specific page type."""
        return cls.PAGE_TYPES.get(page_type_key, cls.PAGE_TYPES["blog_post"])
    
    @classmethod
    def get_all_page_types(cls) -> Dict[str, str]:
        """Get list of all page types for dropdown."""
        return {key: info["name"] for key, info in cls.PAGE_TYPES.items()}
    
    @classmethod
    def enhance_brief_for_page_type(cls, brief: str, page_type_key: str) -> str:
        """Add page type-specific guidance to content brief."""
        page_info = cls.get_page_type_info(page_type_key)
        
        enhancement = f"""

---

## 📄 Page Type: {page_info['name']}

**Purpose:** {page_info['description']}
**Recommended Length:** {page_info['typical_length']}
**SEO Focus:** {page_info['seo_focus']}

### Required Page Elements:

"""
        for element in page_info['key_elements']:
            enhancement += f"- {element}\n"
        
        return brief + enhancement


class WireframeGenerator:
    """Generates visual wireframes for page optimization."""
    
    def __init__(self):
        self.llm = get_llm_client()
    
    def _parse_html_structure(self, html_content: str) -> Dict[str, Any]:
        """Parse HTML to extract page structure."""
        from bs4 import BeautifulSoup
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract structural elements
            structure = {
                "h1": [h1.get_text(strip=True) for h1 in soup.find_all('h1')],
                "h2": [h2.get_text(strip=True) for h2 in soup.find_all('h2')],
                "h3": [h3.get_text(strip=True) for h3 in soup.find_all('h3')],
                "nav": bool(soup.find('nav')),
                "header": bool(soup.find('header')),
                "footer": bool(soup.find('footer')),
                "main": bool(soup.find('main') or soup.find('article')),
                "sidebar": bool(soup.find('aside')),
                "ctas": len(soup.find_all(['button', 'a'], class_=lambda x: x and ('cta' in x.lower() or 'btn' in x.lower()))),
                "forms": len(soup.find_all('form')),
                "images": len(soup.find_all('img')),
                "videos": len(soup.find_all(['video', 'iframe'])),
                "lists": len(soup.find_all(['ul', 'ol'])),
                "tables": len(soup.find_all('table')),
                "blockquotes": len(soup.find_all('blockquote')),
                "meta_title": soup.find('title').get_text(strip=True) if soup.find('title') else None,
                "meta_desc": soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else None,
                "word_count": len(soup.get_text().split()),
                "has_breadcrumbs": bool(soup.find(['nav', 'ol', 'ul'], class_=lambda x: x and 'breadcrumb' in x.lower())),
                "has_toc": bool(soup.find(['nav', 'div'], class_=lambda x: x and ('toc' in x.lower() or 'table-of-contents' in x.lower()))),
                "internal_links": len([a for a in soup.find_all('a', href=True) if not a['href'].startswith(('http://', 'https://'))]),
                "external_links": len([a for a in soup.find_all('a', href=True) if a['href'].startswith(('http://', 'https://'))]),
            }
            
            # Check for schema markup
            structure["has_schema"] = bool(soup.find('script', type='application/ld+json'))
            
            # Check for author info
            structure["has_author"] = bool(
                soup.find(['div', 'section', 'span'], class_=lambda x: x and 'author' in x.lower()) or
                soup.find('meta', attrs={'name': 'author'})
            )
            
            # Check for social sharing
            structure["has_social_share"] = bool(soup.find(['div', 'a'], class_=lambda x: x and ('share' in x.lower() or 'social' in x.lower())))
            
            # Identify sections (divs/sections with classes or IDs)
            sections = []
            for tag in soup.find_all(['section', 'div'], class_=True):
                classes = ' '.join(tag.get('class', []))
                if any(word in classes.lower() for word in ['section', 'content', 'hero', 'features', 'pricing', 'testimonial', 'faq']):
                    heading = tag.find(['h1', 'h2', 'h3'])
                    sections.append({
                        "type": classes,
                        "heading": heading.get_text(strip=True) if heading else "No heading"
                    })
            structure["sections"] = sections[:10]  # Limit to first 10
            
            return structure
            
        except Exception as e:
            print(f"HTML parsing error: {e}")
            return {
                "error": str(e),
                "h1": [],
                "h2": [],
                "h3": [],
                "word_count": 0
            }
    
    def generate_wireframe(self, url: str, html_content: str, page_type: str, keyword: str = "") -> Dict[str, Any]:
        """Generate wireframe with before/after optimization based on actual HTML structure."""
        
        # Parse HTML structure
        structure = self._parse_html_structure(html_content)
        page_info = PageTypeManager.get_page_type_info(page_type)
        
        # Build structure analysis for LLM
        structure_summary = f"""
**Current HTML Structure Analysis:**

**Headings:**
- H1 tags: {len(structure.get('h1', []))} found → {structure.get('h1', [])}
- H2 tags: {len(structure.get('h2', []))} found
- H3 tags: {len(structure.get('h3', []))} found

**Page Elements:**
- Navigation: {'Yes' if structure.get('nav') else 'No'}
- Header: {'Yes' if structure.get('header') else 'No'}
- Footer: {'Yes' if structure.get('footer') else 'No'}
- Main content area: {'Yes' if structure.get('main') else 'No'}
- Sidebar: {'Yes' if structure.get('sidebar') else 'No'}
- CTAs/Buttons: {structure.get('ctas', 0)}
- Forms: {structure.get('forms', 0)}
- Images: {structure.get('images', 0)}
- Videos: {structure.get('videos', 0)}
- Lists: {structure.get('lists', 0)}
- Tables: {structure.get('tables', 0)}

**SEO Elements:**
- Meta Title: {structure.get('meta_title') or 'Missing'}
- Meta Description: {'Present' if structure.get('meta_desc') else 'Missing'}
- Schema Markup: {'Yes' if structure.get('has_schema') else 'No'}
- Author Info: {'Yes' if structure.get('has_author') else 'No'}
- Breadcrumbs: {'Yes' if structure.get('has_breadcrumbs') else 'No'}
- Table of Contents: {'Yes' if structure.get('has_toc') else 'No'}
- Social Sharing: {'Yes' if structure.get('has_social_share') else 'No'}

**Links:**
- Internal links: {structure.get('internal_links', 0)}
- External links: {structure.get('external_links', 0)}

**Content:**
- Word count: {structure.get('word_count', 0)}

**Identified Sections:**
{chr(10).join([f"- {s.get('heading', 'Unnamed')}" for s in structure.get('sections', [])[:5]])}
"""
        
        prompt = f"""You are an SEO/UX expert analyzing a webpage's HTML structure.

**Page Being Analyzed:**
URL: {url}
Target Page Type: {page_info['name']}
Target Keyword: {keyword or 'Not specified'}

{structure_summary}

**Required Elements for {page_info['name']}:**
{chr(10).join([f"- {elem}" for elem in page_info['key_elements']])}

Based on the ACTUAL HTML structure above, create a wireframe analysis.

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "current_issues": [
    "List specific structural issues found in the HTML (e.g., 'Missing H1 tag', 'No author section', 'Multiple H1 tags')"
  ],
  "recommended_sections": [
    {{
      "name": "Section name (e.g., 'Hero Section', 'Features Section')",
      "purpose": "Why this section is needed",
      "elements": ["Specific HTML elements to include (e.g., 'H1 with keyword', 'CTA button', 'Author bio')"],
      "current_status": "missing" or "present" or "needs_improvement"
    }}
  ],
  "priority_fixes": [
    {{
      "fix": "Specific change to make (e.g., 'Add single H1 tag with target keyword')",
      "impact": "high" or "medium" or "low",
      "effort": "low" or "medium" or "high",
      "current": "What exists now",
      "recommended": "What it should be"
    }}
  ],
  "html_structure_score": 0-100,
  "missing_critical_elements": ["List of critical missing elements"]
}}
"""
        
        try:
            result = self.llm.complete(prompt, temperature=0.3, max_tokens=2000)
            import json, re
            clean = re.sub(r'```json\s*|\s*```', '', result.strip())
            wireframe = json.loads(clean)
            
            # Add the parsed structure data
            wireframe['parsed_structure'] = structure
            wireframe['page_type'] = page_info['name']
            wireframe['url'] = url
            
            return wireframe
            
        except Exception as e:
            print(f"Wireframe generation error: {e}")
            return self._generate_fallback_wireframe(page_type, structure)
    
    def _generate_fallback_wireframe(self, page_type: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic wireframe based on parsed structure when LLM fails."""
        page_info = PageTypeManager.get_page_type_info(page_type)
        
        issues = []
        
        # Check H1
        h1_count = len(structure.get('h1', []))
        if h1_count == 0:
            issues.append("Missing H1 heading tag")
        elif h1_count > 1:
            issues.append(f"Multiple H1 tags found ({h1_count}) - should have exactly one")
        
        # Check basic elements
        if not structure.get('meta_title'):
            issues.append("Missing meta title tag")
        if not structure.get('meta_desc'):
            issues.append("Missing meta description")
        if not structure.get('has_schema'):
            issues.append("No schema markup detected")
        if structure.get('ctas', 0) == 0:
            issues.append("No clear CTAs/buttons found")
        if structure.get('images', 0) == 0:
            issues.append("No images found on page")
        
        return {
            "current_issues": issues or ["HTML structure needs manual review"],
            "recommended_sections": [
                {
                    "name": elem,
                    "purpose": "Required for this page type",
                    "elements": [elem],
                    "current_status": "unknown"
                }
                for elem in page_info['key_elements'][:5]
            ],
            "priority_fixes": [
                {
                    "fix": f"Add {elem}",
                    "impact": "high",
                    "effort": "medium",
                    "current": "Missing",
                    "recommended": elem
                }
                for elem in issues[:5]
            ],
            "html_structure_score": max(0, 100 - (len(issues) * 10)),
            "missing_critical_elements": issues,
            "parsed_structure": structure,
            "page_type": page_info['name']
        }
