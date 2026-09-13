# --------------------------------------------
# Scrapingdog API Client
# --------------------------------------------

import json
import streamlit as st
import requests
from requests.adapters import HTTPAdapter, Retry
from typing import Tuple, Optional, Dict, Any, List
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
    # Validate API key format
    if not api_key or api_key.strip() == "" or api_key == "your_scrapingdog_api_key_here":
        return {
            "ok": False,
            "http_status": 0,
            "message": "Invalid or missing API key",
            "related_count": 0,
            "paa_count": 0,
            "organic_count": 0,
            "has_json": False,
            "body_sample": "API key is not set or is still the placeholder value. Please get a valid API key from https://www.scrapingdog.com/",
        }

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

    # Check for authentication errors
    if status == 401:
        result.update({
            "message": "Authentication failed - Invalid API key",
            "body_sample": f"HTTP 401 Unauthorized. Please check your API key. Response: {(body or '')[:200]}\n\nGet your API key from: https://app.scrapingdog.com/dashboard"
        })
        return result

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
        self.autocomplete_url = "https://api.scrapingdog.com/google_autocomplete"
        self.trends_url = "https://api.scrapingdog.com/google_trends"

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

        # Check for authentication errors - gracefully fall back to LLM-only mode
        if status in (401, 403):
            st.warning(f"⚠ Scrapingdog auth failed ({status}). Check your API key in sidebar. Continuing in LLM-only mode.")
            return [], [], []

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

        # Check for authentication errors - gracefully fall back to LLM-only mode
        if status in (401, 403):
            st.warning(f"⚠ Scrapingdog auth failed ({status}). Check your API key in sidebar. Continuing in LLM-only mode.")
            return [], {}, [f"HTTP {status}: Authentication failed"]

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

    @st.cache_data(ttl=1800, show_spinner=False)
    def get_autocomplete_suggestions(_self, query: str, country: str = "uk") -> List[str]:
        """
        Get real Google Autocomplete suggestions for a query.

        Used as a cheap real-world corroboration signal: if Google doesn't
        suggest a phrase, it's a reasonable indicator that few people type it.
        Also useful as an additional keyword expansion source (autocomplete
        surfaces real long-tail phrasing an LLM brainstorm can miss).

        Endpoint: https://api.scrapingdog.com/google_autocomplete

        Args:
            query: Seed keyword/phrase
            country: Two-letter country code (default matches rest of client: uk)

        Returns:
            List of suggestion strings (empty list on failure - fails soft,
            never blocks the rest of the pipeline)
        """
        if not query or not query.strip():
            return []

        params = {
            "api_key": _self.api_key,
            "query": query,
            "country": country
        }
        status, body, data = sd_request(_self.autocomplete_url, params, timeout=15)

        if status in (401, 403):
            st.warning(f"⚠ Scrapingdog autocomplete auth failed ({status}).")
            return []

        if not data:
            return []

        # Response shape: a list of suggestion strings, or list of dicts
        # depending on API version - handle both defensively.
        suggestions = []
        raw_list = data if isinstance(data, list) else data.get("suggestions", data.get("data", []))

        for item in raw_list or []:
            if isinstance(item, str):
                suggestions.append(item)
            elif isinstance(item, dict):
                val = item.get("value") or item.get("query") or item.get("suggestion")
                if val:
                    suggestions.append(val)

        return suggestions

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_trends_interest(_self, keywords: Tuple[str, ...], date_range: str = "today 12-m") -> Dict[str, float]:
        """
        Get real relative search interest (0-100 scale) for up to 5 keywords
        in a single call via Google Trends TIMESERIES data.

        This is the main real-demand signal for keyword scoring/filtering -
        a keyword with an interest value of 0 across the whole window is a
        strong signal nobody searches for it, and should be filtered out
        BEFORE spending LLM analysis time on it.

        Endpoint: https://api.scrapingdog.com/google_trends
        Docs confirm: TIMESERIES mode supports up to 5 comma-separated
        queries per call, 5 API credits per request. Batching keywords into
        groups of 5 here (done by the caller) is significantly cheaper than
        one call per keyword.

        Args:
            keywords: Tuple of up to 5 keywords (tuple, not list, so
                      st.cache_data can hash it for caching)
            date_range: Trends date window. "today 12-m" (12 months) is the
                        default - stable enough to avoid single-week noise,
                        recent enough to reflect current demand.

        Returns:
            Dict mapping each keyword -> average interest score (0-100).
            Keywords with no data returned are omitted from the dict (NOT
            assumed to be zero - caller should treat "missing" differently
            from "confirmed zero", same principle as the DataForSEO version
            we discussed but never shipped).
        """
        if not keywords:
            return {}

        kw_list = list(keywords)[:5]  # API hard limit for TIMESERIES
        query_param = ",".join(kw_list)

        params = {
            "api_key": _self.api_key,
            "query": query_param,
            "data_type": "TIMESERIES",
            "hl": "en",
            "date": date_range
        }
        status, body, data = sd_request(_self.trends_url, params, timeout=30)

        if status in (401, 403):
            st.warning(f"⚠ Scrapingdog Trends auth failed ({status}).")
            return {}

        if not data:
            return {}

        timeline = (
            data.get("interest_over_time", {})
                .get("timeline_data", [])
        )

        if not timeline:
            return {}

        # Average the interest value for each keyword across all timeline
        # points in the window - gives a single representative score rather
        # than a whole time series, which is all the scoring logic needs.
        sums: Dict[str, float] = {kw: 0.0 for kw in kw_list}
        counts: Dict[str, int] = {kw: 0 for kw in kw_list}

        for point in timeline:
            for v in point.get("values", []):
                q = v.get("query")
                val = v.get("value")
                if q in sums and val is not None:
                    try:
                        sums[q] += float(val)
                        counts[q] += 1
                    except (TypeError, ValueError):
                        continue

        results = {}
        for kw in kw_list:
            if counts[kw] > 0:
                results[kw] = round(sums[kw] / counts[kw], 1)
            # If counts[kw] == 0, deliberately omitted - "no data" not "zero"

        return results

    def get_trends_interest_batched(self, keywords: List[str], date_range: str = "today 12-m") -> Dict[str, float]:
        """
        Convenience wrapper: chunks an arbitrary-length keyword list into
        groups of 5 (the API's per-call limit for TIMESERIES) and merges
        the results. This is what keyword_analyzer.py should call directly
        rather than handling chunking itself.

        Args:
            keywords: Any number of keywords
            date_range: Passed through to get_trends_interest

        Returns:
            Merged dict of keyword -> interest score across all chunks
        """
        if not keywords:
            return {}

        merged: Dict[str, float] = {}
        for i in range(0, len(keywords), 5):
            chunk = tuple(keywords[i:i + 5])
            chunk_results = self.get_trends_interest(chunk, date_range)
            merged.update(chunk_results)

        return merged
