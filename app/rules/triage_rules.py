import re

TEAM_ROUTING = {
    "Hardware": "Desktop / Hardware Support",
    "Access": "IT Support / Identity",
    "Administrative rights": "IT Support / Identity",
    "Storage": "Infrastructure / Storage Team",
    "HR Support": "HR Operations Desk",
    "Purchase": "IT Procurement & Assets",
    "Internal Project": "Internal Systems / PMO",
    "Miscellaneous": "General Service Desk",
}

SECURITY_KEYWORDS = [
    "unauthorized",
    "without authorization",
    "someone accessed",
    "accessing my account",
    "suspicious activity",
    "hacked",
    "breach",
    "phishing",
    "compromised",
    "ransomware",
    "malware",
]


def evaluate_priority(ticket_text: str, category: str) -> tuple[str, str]:
    """
    Transparent, rule-based urgency scoring with explicit audit reasons.
    """
    text = ticket_text.lower()

    if any(k in text for k in ["production down", "prod down", "server down", "system outage", "unavailable for all"]):
        return "Critical", "Production outage / critical availability impact"

    if any(k in text for k in SECURITY_KEYWORDS):
        return "Critical", "Security-sensitive incident detected"

    if any(k in text for k in ["all users", "entire office", "whole team", "everyone affected", "company wide"]):
        return "High", "Multi-user / wide operational impact"

    if any(k in text for k in ["urgent", "asap", "client deadline", "presentation in", "meeting in", "call in", "blocking", "immediately"]):
        return "High", "Time-sensitive / blocking business workflow"

    if any(k in text for k in ["password reset", "forgot password", "new monitor", "software request", "inquiry"]):
        return "Low", "Standard routine request"

    if category in {"Hardware", "Access", "Storage", "Administrative rights"}:
        return "Medium", f"Standard technical triage for {category}"

    return "Low", f"Standard operational queue for {category}"


def evaluate_review_and_routing(ticket_text: str, category: str, confidence: float) -> dict:
    """
    Applies confidence thresholds (0.80 / 0.60) plus responsible-AI edge-case overrides.
    """
    text_lower = ticket_text.lower()
    words = re.findall(r"\b[a-zA-Z]+\b", ticket_text)

    review_reasons = []

    # 1. Edge-case check: Very short, generic, or symbol-only input
    if len(words) < 3:
        review_reasons.append("Insufficient context (< 3 words)")

    # 2. Edge-case check: Security-sensitive ticket override
    if any(k in text_lower for k in SECURITY_KEYWORDS):
        review_reasons.append("Security-sensitive policy override")

    # 3. Edge-case check: Multi-issue / contradictory symptom indicator
    symptom_hits = sum(
        1 for kw in ["wifi", "wi-fi", "vpn", "outlook", "laptop", "screen", "printer", "password", "overheating", "database"]
        if kw in text_lower
    )
    if symptom_hits >= 2 and any(sep in text_lower for sep in [" and ", " also ", " but ", ","]):
        review_reasons.append("Multiple or competing issues detected")

    # 4. Confidence threshold policy
    if confidence >= 0.80:
        confidence_band = "high"
    elif confidence >= 0.60:
        confidence_band = "attention"
        review_reasons.append("Moderate confidence (0.60-0.79)")
    else:
        confidence_band = "low"
        review_reasons.append("Low model confidence (< 0.60)")

    human_review_required = len(review_reasons) > 0
    priority, priority_reason = evaluate_priority(ticket_text, category)

    # 5. Domain Routing Override: VPN / Connectivity issues route to Identity & Network Support
    if any(net_kw in text_lower for net_kw in ["vpn", "wi-fi", "wifi", "sso", "network"]):
        suggested_team = "IT Support / Identity"
    else:
        suggested_team = TEAM_ROUTING.get(category, "General Service Desk")

    return {
        "priority": priority,
        "priority_reason": priority_reason,
        "suggested_team": suggested_team,
        "confidence_band": confidence_band,
        "human_review_required": human_review_required,
        "review_reason": "; ".join(review_reasons) if review_reasons else "Auto-triage approved",
    }