# --------------------------------------------
# Competitive Analysis
# --------------------------------------------

import streamlit as st
from typing import Dict, Any, List
from api.scrapingdog_client import ScrapingdogClient
from utils.web_scraping import fetch_and_parse_url, extract_headings
from analysis.keyword_analyzer import extract_common_headings
from config import MAX_COMPETITORS


class CompetitiveAnalyzer:
    """Analyze top competitors from SERP."""

    def __init__(self, scrapingdog_client: ScrapingdogClient):
        """
        Initialize competitive analyzer.

        Args:
            scrapingdog_client: Configured Scrapingdog client
        """
        self.scrapingdog = scrapingdog_client

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
                "common_headings": []
            }

            total_wc = 0
            successful_fetches = 0
            all_headings = []

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

                    data["competitors"].append({
                        "url": url,
                        "title": result.get("title", ""),
                        "word_count": wc,
                        "headings": heads,
                        "snippet": result.get("snippet", "")
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

            return data

        except Exception as e:
            st.error(f"Error in competitive analysis: {e}")
            return {}
