from app.rules.triage_rules import evaluate_priority, evaluate_review_and_routing


def test_vpn_domain_routing_and_meeting_priority():
    res = evaluate_review_and_routing(
        "My office VPN is not connecting and I have a meeting in 10 minutes",
        category="Hardware",
        confidence=0.85,
    )
    assert res["suggested_team"] == "IT Support / Identity"
    assert res["priority"] == "High"
    assert res["human_review_required"] is False
    assert res["review_reason"] == "Auto-triage approved"


def test_security_keyword_forces_critical_and_human_review():
    res = evaluate_review_and_routing(
        "Someone appears to be accessing my account without authorization",
        category="Access",
        confidence=0.96,
    )
    assert res["priority"] == "Critical"
    assert res["human_review_required"] is True
    assert "Security-sensitive policy override" in res["review_reason"]


def test_short_vague_input_triggers_human_review():
    res = evaluate_review_and_routing(
        "Not working",
        category="Hardware",
        confidence=0.91,
    )
    assert res["human_review_required"] is True
    assert "Insufficient context" in res["review_reason"]


def test_multi_symptom_ticket_triggers_human_review():
    res = evaluate_review_and_routing(
        "Laptop overheating, Wi-Fi dropping and Outlook crashing at the same time",
        category="Hardware",
        confidence=0.95,
    )
    assert res["human_review_required"] is True
    assert "Multiple or competing issues detected" in res["review_reason"]


def test_low_confidence_triggers_human_review():
    res = evaluate_review_and_routing(
        "Need assistance with internal portal configuration today",
        category="Miscellaneous",
        confidence=0.58,
    )
    assert res["confidence_band"] == "low"
    assert res["human_review_required"] is True
    assert "Low model confidence" in res["review_reason"]


def test_routine_password_reset_priority():
    priority, _ = evaluate_priority("Need a standard password reset for my account", "Access")
    assert priority == "Low"