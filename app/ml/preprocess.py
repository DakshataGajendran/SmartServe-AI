import re

def clean_ticket_text(text: str) -> str:
    """
    Standardize whitespace and strip outer spaces without destroying
    useful signals (like typos, numbers, or urgency words).
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text