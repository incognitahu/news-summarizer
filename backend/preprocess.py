# backend/preprocess.py
"""
HTML/text preprocessing utilities for News Summarizer.

Features:
- extract_text_from_html(html): tries boilerpy3/readability/newspaper, falls back to BeautifulSoup heuristics.
- clean_html(raw_html): removes scripts, styles, comments, and normalizes.
- remove_boilerplate_candidates(html): heuristics to drop nav/footer/sidebars based on tag names and class/id hints.
- normalize_whitespace(text): tidy up spaces/newlines.
- keep_long_paragraphs(text, min_words=5): heuristic filter to remove tiny noise paragraphs.
- preprocess_for_model(raw_input, return_sentences=False): full pipeline returning cleaned single-string and optional sentence list.
- batch_preprocess_texts(texts): for dataset processing.
"""

from typing import List, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# HTML parsing
from bs4 import BeautifulSoup, Comment

# Optional: boilerplate extractors (used if installed)
_HAS_BOILERPY3 = False
_HAS_READABILITY = False
_HAS_NEWSPAPER = False
try:
    from boilerpy3 import extractors  # type: ignore
    _HAS_BOILERPY3 = True
except Exception:
    _HAS_BOILERPY3 = False

try:
    from readability import Document  # type: ignore
    _HAS_READABILITY = True
except Exception:
    _HAS_READABILITY = False

try:
    from newspaper import Article  # type: ignore
    _HAS_NEWSPAPER = True
except Exception:
    _HAS_NEWSPAPER = False

# Sentence splitting (spaCy preferred, NLTK fallback)
try:
    import spacy
    _HAS_SPACY = True
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
    except Exception:
        _SPACY_NLP = spacy.blank("en")
        _SPACY_NLP.add_pipe("sentencizer")
except Exception:
    _HAS_SPACY = False
    _SPACY_NLP = None

try:
    import nltk
    _HAS_NLTK = True
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
except Exception:
    _HAS_NLTK = False

# -----------------------
# Core functions
# -----------------------
def clean_html(raw_html: str) -> str:
    """
    Remove script/style tags and HTML comments, return cleaned HTML string.
    This does not extract text content; it's an HTML cleanup helper.
    """
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "lxml")
    # Remove scripts and styles
    for s in soup(["script", "style", "noscript", "iframe", "svg", "link", "meta"]):
        s.decompose()
    # Remove comments
    for c in soup.findAll(text=lambda text: isinstance(text, Comment)):
        c.extract()
    # Remove invisible elements and attributes that tend to be boilerplate
    # We won't strip nav/footer here; leave to remove_boilerplate_candidates
    return str(soup)


def remove_boilerplate_candidates(html: str, keep_selectors: Optional[List[str]] = None) -> str:
    """
    Heuristic removal of nav/footers/sidebars based on tag names, IDs, and classes.
    Returns cleaned HTML string.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")

    # Heuristic container selectors to remove
    removal_selectors = [
        "nav", "footer", "aside", "header", "form", "button", "noscript",
        # common class/id hints
        "[class*=nav]", "[id*=nav]",
        "[class*=footer]", "[id*=footer]",
        "[class*=menu]", "[id*=menu]",
        "[class*=sidebar]", "[id*=sidebar]",
        "[class*=comment]", "[id*=comment]",
        "[class*=ads]", "[id*=ads]", "[class*=advert]", "[id*=advert]",
        "[class*=cookie]", "[id*=cookie]",
    ]
    if keep_selectors:
        # allow caller to keep certain selectors (rare)
        for sel in keep_selectors:
            try:
                for node in soup.select(sel):
                    node['data-preserve'] = "1"
            except Exception:
                pass

    # Remove matching nodes (try/except to be defensive)
    for sel in removal_selectors:
        try:
            for node in soup.select(sel):
                # respect preserved items
                if node.attrs.get("data-preserve") == "1":
                    continue
                node.decompose()
        except Exception:
            continue

    # Also remove empty tags and short tags that are unlikely to be content
    for tag in soup.find_all():
        # remove tags with no textual content or with only whitespace
        if tag.name in ["div", "section", "article", "p", "span"] and (not tag.get_text(strip=True)):
            try:
                tag.decompose()
            except Exception:
                pass

    return str(soup)


def _bs_extract_main_text(html: str, min_para_words: int = 3) -> str:
    """
    BeautifulSoup heuristic for extracting main article text:
      - keep <article> if present, otherwise select big <p> blocks
      - filter out very short paragraphs (but keep if total content is small)
      - join paragraphs preserving sentence boundaries
    """
    soup = BeautifulSoup(html, "lxml")

    # Prefer <article> tag if it exists
    article_tag = soup.find("article")
    if article_tag and article_tag.get_text(strip=True):
        paras = [p.get_text(" ", strip=True) for p in article_tag.find_all(["p", "div"]) if p.get_text(strip=True)]
        # If we have short paras but combined they're substantial, keep all
        total_words = sum(len(p.split()) for p in paras)
        if total_words >= min_para_words:
            # Filter individual paras only if we have enough total content
            if total_words > 20:
                paras = [p for p in paras if len(p.split()) >= min_para_words]
            if paras:
                return "\n\n".join(paras)

    # Otherwise, collect <p> tags across the document
    p_tags = soup.find_all("p")
    paras = [p.get_text(" ", strip=True) for p in p_tags if p.get_text(strip=True)]
    
    # Smart filtering: if total is small, keep everything; otherwise filter short paras
    total_words = sum(len(p.split()) for p in paras)
    if total_words >= min_para_words:
        if total_words > 20:  # Enough content to be selective
            paras = [p for p in paras if len(p.split()) >= min_para_words]
        if paras:
            return "\n\n".join(paras)

    # Fallback: take longest text blocks from divs
    div_texts = [d.get_text(" ", strip=True) for d in soup.find_all("div") if d.get_text(strip=True)]
    # sort by word count and keep top few
    div_texts_sorted = sorted(div_texts, key=lambda s: len(s.split()), reverse=True)
    if div_texts_sorted:
        top = div_texts_sorted[:5]
        # split them into paragraphs heuristically
        return "\n\n".join(top)

    return ""


def extract_text_from_html(html: str) -> str:
    """
    Try best-effort text extraction using available libraries:
    - boilerpy3 (article extractor) if available
    - readability-lxml Document if available
    - newspaper3k Article if available
    - fallback to BeautifulSoup heuristics
    """
    if not html:
        return ""

    # First, try boilerpy3
    if _HAS_BOILERPY3:
        try:
            ext = extractors.ArticleExtractor()
            content = ext.get_content(html)
            if content and len(content.split()) > 20:
                return content.strip()
        except Exception as e:
            logger.debug(f"boilerpy3 extraction failed: {e}")

    # readability
    if _HAS_READABILITY:
        try:
            doc = Document(html)
            content = doc.summary()  # HTML summary
            # extract main text from summary HTML
            text = _bs_extract_main_text(content)
            if text and len(text.split()) > 20:
                return text.strip()
        except Exception as e:
            logger.debug(f"readability extraction failed: {e}")

    # newspaper
    if _HAS_NEWSPAPER:
        try:
            a = Article("")  # create empty article
            a.set_html(html)
            a.parse()
            text = a.text
            if text and len(text.split()) > 20:
                return text.strip()
        except Exception as e:
            logger.debug(f"newspaper extraction failed: {e}")

    # Fallback: clean and use simple bs heuristic
    try:
        cleaned_html = clean_html(html)
        cleaned_html = remove_boilerplate_candidates(cleaned_html)
        text = _bs_extract_main_text(cleaned_html)
        if text and len(text.split()) > 10:
            return text.strip()
    except Exception as e:
        logger.warning(f"BeautifulSoup fallback extractor failed: {e}")

    # As last resort, return the visible text from soup (may be noisy)
    try:
        soup = BeautifulSoup(html, "lxml")
        visible_text = soup.get_text(" ", strip=True)
        return visible_text.strip()
    except Exception:
        return ""


# -----------------------
# Text normalization & helpers
# -----------------------
def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    # convert windows newlines, multiple spaces, weird unicode spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # normalize non-breaking spaces
    text = text.replace("\u00A0", " ")
    # collapse multiple newlines to double newline (paragraph sep)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # strip extraneous leading/trailing whitespace
    return text.strip()


def keep_long_paragraphs(text: str, min_words: int = 5) -> str:
    """
    Keep paragraphs that have at least min_words words.
    """
    if not text:
        return ""
    paras = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
    filtered = [p for p in paras if len(p.split()) >= min_words]
    return "\n\n".join(filtered)


def split_into_sentences(text: str) -> List[str]:
    """
    Sentence segmentation using spaCy (preferred) or NLTK fallback or naive regex.
    """
    if not text:
        return []
    if _HAS_SPACY and _SPACY_NLP is not None:
        try:
            doc = _SPACY_NLP(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        except Exception as e:
            logger.debug(f"spaCy sentence split failed: {e}")

    if _HAS_NLTK:
        try:
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(text)
            return [s.strip() for s in sents if s.strip()]
        except Exception as e:
            logger.debug(f"NLTK sentence split failed: {e}")

    # naive regex fallback
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]


# -----------------------
# Helper: HTML detection
# -----------------------
def is_likely_full_html(text: str) -> bool:
    """
    Detect if input is likely a full HTML document vs plain text with occasional tags.
    Returns True if it looks like real HTML structure.
    """
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Strong indicators of full HTML
    html_indicators = ['<html', '<body', '<head', '<!doctype', '<article', '<div']
    strong_match = any(indicator in text_lower for indicator in html_indicators)
    
    # Count total tag occurrences
    tag_count = len(re.findall(r'<[^>]+>', text))
    
    # If has strong HTML structure indicators, definitely HTML
    if strong_match:
        return True
    
    # If has many tags (>5), likely HTML
    if tag_count > 5:
        return True
    
    # If has few tags (<= 3) and doesn't start/end with tags, likely plain text
    if tag_count <= 3:
        return False
    
    # Otherwise, use tag density heuristic
    text_without_tags = re.sub(r'<[^>]+>', '', text).strip()
    if len(text_without_tags) == 0:
        return True  # Only tags, treat as HTML
    
    # If tags take up significant portion, treat as HTML
    tag_ratio = (len(text) - len(text_without_tags)) / len(text)
    return tag_ratio > 0.15  # >15% of content is tags


# -----------------------
# High-level pipeline
# -----------------------
def preprocess_for_model(
    raw_input: str,
    input_is_html: bool = True,
    min_para_words: int = 3,
    return_sentences: bool = False,
    sentence_split: bool = True,
) -> Tuple[str, Optional[List[str]]]:
    """
    Full preprocess pipeline:
      - if input_is_html: extract main content (with auto-detection)
      - clean html, remove boilerplate
      - normalize whitespace
      - optionally return sentence list

    Returns:
      (cleaned_text, sentences or None)
    """
    if not raw_input:
        return "", [] if return_sentences else ("", None)

    text = raw_input
    
    # Smart HTML detection: auto-adjust if input_is_html=True but content isn't really HTML
    if input_is_html and not is_likely_full_html(raw_input):
        logger.info("Input flagged as HTML but appears to be plain text - treating as plain text")
        input_is_html = False
    
    if input_is_html:
        try:
            text = extract_text_from_html(raw_input)
            # Fallback if HTML extraction produced empty/very-short result
            # Lowered threshold to 5 to catch edge cases like "Hello World"
            if not text or len(text.strip()) < 5:
                logger.warning("HTML extraction produced minimal content - using fallback")
                # Simple tag removal fallback
                text = re.sub(r"<[^>]+>", " ", raw_input)
                text = re.sub(r"\s+", " ", text).strip()
                # Skip paragraph filtering for fallback cases - keep whatever we got
                text = normalize_whitespace(text)
                sentences = None
                if return_sentences or sentence_split:
                    sentences = split_into_sentences(text) if sentence_split else None
                if return_sentences:
                    return text, sentences
                else:
                    return text, None
        except Exception as e:
            logger.warning(f"HTML extraction failed: {e}")
            # fallback: remove tags and use visible text
            text = re.sub(r"<[^>]+>", " ", raw_input)
            text = re.sub(r"\s+", " ", text).strip()
            # Skip paragraph filtering for fallback cases
            text = normalize_whitespace(text)
            sentences = None
            if return_sentences or sentence_split:
                sentences = split_into_sentences(text) if sentence_split else None
            if return_sentences:
                return text, sentences
            else:
                return text, None

    # Normalize and filter paragraphs (but preserve short content if total is small)
    original_length = len(text.split())
    text = normalize_whitespace(text)
    text = keep_long_paragraphs(text, min_words=min_para_words)
    
    # If filtering removed everything but we had some content, restore it
    if not text.strip() and original_length > 0:
        logger.warning(f"Paragraph filtering removed all content (had {original_length} words) - restoring")
        text = normalize_whitespace(raw_input if not input_is_html else re.sub(r"<[^>]+>", " ", raw_input))

    # Further cleaning: remove weird header/footer lines (heuristic)
    # Remove lines that look like navigation or "read more" labels
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned_lines = []
    for ln in lines:
        lower = ln.lower()
        if len(ln.split()) < 3 and (lower.startswith("read") or lower.startswith("follow") or "copyright" in lower):
            continue
        cleaned_lines.append(ln)
    text = "\n\n".join(cleaned_lines)
    text = normalize_whitespace(text)

    sentences = None
    if return_sentences or sentence_split:
        sentences = split_into_sentences(text) if sentence_split else None

    if return_sentences:
        return text, sentences
    else:
        return text, None


def batch_preprocess_texts(texts: List[str], input_is_html: bool = True) -> List[str]:
    """
    Preprocess a list of texts or HTML pages; returns list of cleaned strings suitable for model input.
    Useful for dataset preprocessing pipelines.
    """
    cleaned = []
    for t in texts:
        c, _ = preprocess_for_model(t, input_is_html=input_is_html)
        cleaned.append(c)
    return cleaned


# -----------------------
# Minimal self-test (only run when module executed directly)
# -----------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_html = """
    <html><head><title>Test</title></head><body>
      <nav>top menu | login</nav>
      <article>
        <p>First paragraph of the article. It is the lead and should be kept.</p>
        <p>Second paragraph with more details and facts. Contains named entities like Singapore and OpenAI.</p>
      </article>
      <footer>Copyright 2025</footer>
    </body></html>
    """
    clean, sents = preprocess_for_model(sample_html, input_is_html=True, return_sentences=True)
    print("CLEANED TEXT:\n", clean[:400])
    print("\nSENTENCES:", sents)
