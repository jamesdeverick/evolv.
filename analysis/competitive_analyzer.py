# --------------------------------------------
# Competitive Analysis
# --------------------------------------------

import streamlit as st
from typing import Dict, Any, List
from api.scrapingdog_client import ScrapingdogClient
from utils.web_scraping import fetch_and_parse_url, extract_headings
from analysis.keyword_analyzer import extract_common_headings
from config import MAX_COMPETITORS

# Try to import entity extractor for generative search optimization
try:
    from analysis.entity_extractor import EntityExtractor, extract_common_entities
    ENTITY_EXTRACTION_AVAILABLE = True
except ImportError:
    ENTITY_EXTRACTION_AVAILABLE = False


class CompetitiveAnalyzer:
    """Analyze top competitors from SERP with entity and semantic analysis."""

    def __init__(self, scrapingdog_client: ScrapingdogClient):
        """
        Initialize competitive analyzer.

        Args:
            scrapingdog_client: Configured Scrapingdog client
        """
        self.scrapingdog = scrapingdog_client

        # Initialize entity extractor if available
        if ENTITY_EXTRACTION_AVAILABLE:
            self.entity_extractor = EntityExtractor()
        else:
            self.entity_extractor = None

    def analyze_competitors(
        self,
        keyword: str,
        num_competitors: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze top competitors for a keyword.

        Args:
            keyword: Target keyword
            num_competitors: Number of top competitors to analyze

        Returns:
            Dict with competitor analysis data
        """
        if num_competitors > MAX_COMPETITORS:
            num_competitors = MAX_COMPETITORS
            st.warning(f"Limited to maximum {MAX_COMPETITORS} competitors")

        try:
            # Get SERP results
            serp_results, _, _ = self.scrapingdog.analyze_serp(keyword)

            if not serp_results:
                return {}

            data = {
                "competitors": [],
                "avg_word_count": 0,
                "common_headings": [],
                "entity_analysis": {},
                "common_entities": {}
            }

            total_wc = 0
            successful_fetches = 0
            all_headings = []
            all_entities = []

            # Progress tracking
            pb = st.progress(0)
            status = st.empty()

            for i, result in enumerate(serp_results[:num_competitors]):
                url = result.get("url", "")
                status.text(f"Analyzing competitor {i+1}/{num_competitors}: {url}")
                pb.progress((i+1) / num_competitors)

                # Fetch and parse competitor page
                content = fetch_and_parse_url(url)

                if (isinstance(content, str) and
                    not content.startswith("Error") and
                    len(content.strip()) > 100):

                    wc = len(content.split())
                    heads = extract_headings(content)

                    # Extract entities if available
                    entities = {}
                    if self.entity_extractor and self.entity_extractor.available:
                        entities = self.entity_extractor.extract_entities(content)
                        all_entities.append(entities)

                    data["competitors"].append({
                        "url": url,
                        "title": result.get("title", ""),
                        "word_count": wc,
                        "headings": heads,
                        "snippet": result.get("snippet", ""),
                        "entities": entities
                    })

                    total_wc += wc
                    successful_fetches += 1
                    all_headings.append(heads)

            # Clean up progress indicators
            pb.empty()
            status.empty()

            # Calculate statistics
            if successful_fetches > 0:
                data["avg_word_count"] = int(total_wc / successful_fetches)
                data["common_headings"] = extract_common_headings(all_headings)

                # Calculate common entities across competitors
                if all_entities and ENTITY_EXTRACTION_AVAILABLE:
                    data["common_entities"] = extract_common_entities(all_entities, min_frequency=2)

                    # Calculate entity statistics
                    data["entity_analysis"] = {
                        "avg_orgs_mentioned": self._avg_entity_count(all_entities, "ORG"),
                        "avg_people_mentioned": self._avg_entity_count(all_entities, "PERSON"),
                        "avg_products_mentioned": self._avg_entity_count(all_entities, "PRODUCT"),
                        "total_unique_orgs": len(set(
                            ent for entities in all_entities
                            for ent in entities.get("ORG", [])
                        )),
                        "total_unique_people": len(set(
                            ent for entities in all_entities
                            for ent in entities.get("PERSON", [])
                        ))
                    }

            return data

        except Exception as e:
            st.error(f"Error in competitive analysis: {e}")
            return {}

    def _avg_entity_count(self, all_entities: List[Dict], entity_type: str) -> float:
        """
        Calculate average count of a specific entity type across all competitors.

        Args:
            all_entities: List of entity dictionaries from all competitors
            entity_type: Type of entity to count (e.g., "ORG", "PERSON")

        Returns:
            Average count
        """
        if not all_entities:
            return 0.0

        total = sum(len(entities.get(entity_type, [])) for entities in all_entities)
        return round(total / len(all_entities), 1)
