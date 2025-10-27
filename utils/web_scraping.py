# --------------------------------------------
# Web Scraping Utilities
# --------------------------------------------

import re
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup, Comment
from config import WEB_FETCH_TIMEOUT, MAX_CONTENT_LENGTH


def is_safe_url(url: str) -> bool:
    """
    Validate URL to prevent SSRF attacks.

    Args:
        url: URL to validate

    Returns:
        True if URL is safe to fetch
    """
    try:
        parsed = urlparse(url)

        # Must have a scheme and hostname
        if not parsed.scheme or not parsed.hostname:
            return False

        # Block localhost and internal IPs
        blocked_hosts = [
            'localhost',
            '127.0.0.1',
            '0.0.0.0',
            '::1',
        ]

        hostname_lower = parsed.hostname.lower()

        if hostname_lower in blocked_hosts:
            return False

        # Block private IP ranges
        if (hostname_lower.startswith('192.168.') or
            hostname_lower.startswith('10.') or
            hostname_lower.startswith('172.16.') or
            hostname_lower.startswith('172.17.') or
            hostname_lower.startswith('172.18.') or
            hostname_lower.startswith('172.19.') or
            hostname_lower.startswith('172.20.') or
            hostname_lower.startswith('172.21.') or
            hostname_lower.startswith('172.22.') or
            hostname_lower.startswith('172.23.') or
            hostname_lower.startswith('172.24.') or
            hostname_lower.startswith('172.25.') or
            hostname_lower.startswith('172.26.') or
            hostname_lower.startswith('172.27.') or
            hostname_lower.startswith('172.28.') or
            hostname_lower.startswith('172.29.') or
            hostname_lower.startswith('172.30.') or
            hostname_lower.startswith('172.31.')):
            return False

        return True
    except Exception:
        return False


def fetch_and_parse_url(url: str) -> str:
    """
    Fetch content from a URL and extract readable text with security checks.

    Args:
        url: URL to fetch

    Returns:
        Extracted text content or error message
    """
    # Security validation
    if not is_safe_url(url):
        return "Error: URL is not allowed (security policy)."

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=WEB_FETCH_TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove unwanted elements
        for tag in soup(['script', 'style', 'header', 'footer', 'nav',
                        'aside', 'form', 'noscript', 'iframe', 'svg']):
            tag.decompose()

        # Remove HTML comments
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            try:
                comment.extract()
            except Exception:
                pass

        # Remove ads and popups
        for div in soup.find_all('div', {'class': ['ad', 'banner', 'popup', 'modal']}):
            div.decompose()

        # Try to find main content
        main_elems = soup.find_all(['main', 'article', 'section'])
        if not main_elems:
            main_elems = soup.find_all(
                'div',
                class_=re.compile(
                    r'(content|main|post|body|section|wrapper|container|entry|page|article-content|blog-post)',
                    re.I
                )
            )

        extracted = []
        if main_elems:
            for el in main_elems:
                t = el.get_text(separator="\n", strip=True)
                if t:
                    extracted.append(t)

        # Fallback to all text elements if main content is too short
        if not extracted or len("".join(extracted).strip()) < 200:
            body = soup.body or soup
            for tag in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div']:
                for el in body.find_all(tag):
                    t = el.get_text(separator="\n", strip=True)
                    if t and len(t) > 10:
                        extracted.append(t)

        full_text = "\n\n".join(extracted)

        # Clean up whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', full_text)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)

        # Truncate if too long
        if len(cleaned) > MAX_CONTENT_LENGTH:
            cleaned = cleaned[:MAX_CONTENT_LENGTH] + "\n\n... [Content truncated due to length] ..."

        if not cleaned.strip():
            return "No substantial content could be extracted from the URL."

        return cleaned

    except requests.exceptions.Timeout:
        return "Error fetching URL: Request timed out."
    except requests.exceptions.ConnectionError:
        return "Error fetching URL: Connection failed."
    except requests.exceptions.HTTPError as e:
        return f"Error fetching URL: HTTP {e.response.status_code}"
    except Exception as e:
        return f"Error fetching/parsing URL content: {e}"


def extract_headings(content: str, max_headings: int = 20) -> list:
    """
    Extract potential headings from text content.

    Args:
        content: Text content to analyze
        max_headings: Maximum number of headings to return

    Returns:
        List of extracted headings
    """
    headings = []
    for line in content.split("\n"):
        line = line.strip()
        # Heuristic: 5-80 chars, doesn't end with punctuation, starts with capital
        if 5 <= len(line) <= 80 and not line.endswith(('.', ',')) and line[:1].isupper():
            headings.append(line)

        if len(headings) >= max_headings:
            break

    return headings
