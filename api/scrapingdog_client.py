# --------------------------------------------
# Scrapingdog API Client
# --------------------------------------------

import json
import streamlit as st
import requests
from requests.adapters import HTTPAdapter, Retry
from typing import Tuple, Optional, Dict, Any
from config import SCRAPINGDOG_TIMEOUT, SCRAPINGDOG_API_KEY


def sd_request(url: str, params: dict, timeout: int = SCRAPINGDOG_TIMEOUT) -> Tuple[int, str, Optional[Dict]]:
    """
    Call Scrapingdog API with retry logic and tolerant JSON parsing.

    Attempts to parse JSON even on non-200 status codes (e.g., 403 with payload).

    Args:
        url: API endpoint URL
        params: Query parameters including API key
        timeout: Request timeout in seconds

    Returns:
        Tuple of (status_code, response_text, parsed_json_or_None)
    """
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    headers = {
        "Accept": "application/json",
        "User-Agent": "seo-assistant/2.0 (+local)"
    }

    try:
        r = session.get(url, params=params, headers=headers, timeout=timeout)
        text = r.text or ""
        data = None

        # Try to parse JSON
        try:
            data = r.json()
        except json.JSONDecodeError:
            # Try to extract JSON from response text
            stripped = text.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    data = json.loads(stripped)
                except Exception:
                    pass

        return r.status_code, text, data
    except requests.exceptions.Timeout:
        return 0, "Request timed out", None
    except requests.exceptions.ConnectionError:
        return 0, "Connection failed", None
    except Exception as e:
        return 0, str(e), None


@st.cache_data(ttl=300, show_spinner=False)
def probe_scrapingdog_status(api_key: str) -> Dict[str, Any]:
    """
    Probe Scrapingdog API connectivity and response format.

    Args:
        api_key: Scrapingdog API key

    Returns:
        Dict with status information
    """
    url = "https://api.scrapingdog.com/google"
    params = {"api_key": api_key, "query": "test"}
    status, body, data = sd_request(url, params, timeout=20)

    result = {
        "ok": False,
        "http_status": status,
        "message": "",
        "related_count": 0,
        "paa_count": 0,
        "organic_count": 0,
        "has_json": bool(data),
        "body_sample": (body or "")[:300],
    }

    if data:
        # Parse response fields (handle multiple naming conventions)
        related = []
        for k in ("relatedSearches", "related_searches"):
            related += [s.get("query") for s in data.get(k, []) if s.get("query")]

        paa = []
        for k in ("peopleAlsoAskedFor", "people_also_asked_for"):
            paa += [q.get("question") for q in data.get(k, []) if q.get("question")]

        organics = data.get("organic_results", []) or []

        result.update({
            "ok": True,  # JSON present = usable
            "related_count": len(related),
            "paa_count": len(paa),
            "organic_count": len(organics),
            "message": "Parsed JSON payload",
        })
    else:
        result["message"] = "No JSON payload"

    return result


class ScrapingdogClient:
    """Client for Scrapingdog SERP API."""

    def __init__(self, api_key: str):
        """
        Initialize Scrapingdog client.

        Args:
            api_key: Scrapingdog API key
        """
        self.api_key = api_key
        self.base_url = "https://api.scrapingdog.com/google"

    @st.cache_data(ttl=1800, show_spinner=False)
    def get_keywords(_self, query: str):
        """
        Get related keywords and SERP data.

        Args:
            query: Search query

        Returns:
            Tuple of (related_searches, people_also_ask, organic_results)
        """
        if not query or not query.strip():
            st.error("Cannot retrieve keywords with an empty query.")
            return [], [], []

        params = {
            "api_key": _self.api_key,
            "query": query,
            "gl": "uk",
            "hl": "en"
        }
        status, body, data = sd_request(_self.base_url, params)

        if data:
            # Parse related searches
            related = []
            for key in ("relatedSearches", "related_searches"):
                related.extend([
                    s.get("query") for s in data.get(key, [])
                    if s.get("query")
                ])

            # Parse people also ask
            paa = []
            for key in ("peopleAlsoAskedFor", "people_also_asked_for"):
                paa.extend([
                    q.get("question") for q in data.get(key, [])
                    if q.get("question")
                ])

            # Parse organic results
            organics = []
            for e in data.get("organic_results", []) or []:
                organics.append({
                    "url": e.get("link") or e.get("url") or "N/A",
                    "title": e.get("title", "N/A"),
                    "snippet": e.get("snippet") or e.get("description") or "No snippet available."
                })

            if status != 200:
                st.warning(
                    f"Scrapingdog returned HTTP {status} but included data. "
                    "Proceeding with parsed JSON."
                )
                with st.expander("Scrapingdog response (first 400 chars)"):
                    st.code(body[:400])

            if not (related or paa or organics):
                st.warning("Scrapingdog JSON parsed but no expected fields found.")
                with st.expander("Raw JSON"):
                    st.code(json.dumps(data, indent=2)[:2000])

            return related, paa, organics

        # Error case
        st.error(
            f"Scrapingdog call failed (HTTP {status}). "
            f"Body (first 400 chars):\n{body[:400]}"
        )
        st.info("Falling back to LLM-only brainstorming.")
        return [], [], []

    @st.cache_data(ttl=1800, show_spinner=False)
    def analyze_serp(_self, keyword: str):
        """
        Analyze SERP for a specific keyword.

        Args:
            keyword: Target keyword

        Returns:
            Tuple of (serp_results, raw_data, debug_messages)
        """
        if not keyword or not keyword.strip():
            st.error("Cannot perform SERP analysis with an empty keyword.")
            return [], {}, ["Empty keyword"]

        params = {
            "api_key": _self.api_key,
            "query": keyword,
            "gl": "uk",
            "hl": "en"
        }
        status, body, data = sd_request(_self.base_url, params)
        debug = []

        if data:
            serp = []
            for e in data.get("organic_results", []) or []:
                serp.append({
                    "url": e.get("link") or e.get("url") or "N/A",
                    "title": e.get("title", "N/A"),
                    "snippet": e.get("snippet") or e.get("description") or "No snippet available."
                })

            if status != 200:
                st.warning(
                    f"SERP analysis: HTTP {status} but data present. Continuing."
                )
                with st.expander("SERP raw body (first 400 chars)"):
                    st.code(body[:400])

            if not serp:
                debug.append("Parsed JSON but no organic_results found.")

            return serp, data, debug

        st.error(
            f"SERP analysis failed (HTTP {status}). "
            f"Body (first 400 chars):\n{body[:400]}"
        )
        return [], {}, ["No JSON"]
