import pytest
from fastapi.testclient import TestClient

import app.database.db as db_module
from app.api.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Provides a FastAPI TestClient backed by an isolated temporary SQLite database."""
    test_db = tmp_path / "test_smartserve.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db(test_db)
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_employee_and_admin_login(client):
    emp_res = client.post(
        "/auth/login",
        json={"role": "employee", "username": "EMP-4082", "password": "emp123"},
    )
    assert emp_res.status_code == 200
    assert emp_res.json()["role"] == "employee"
    assert emp_res.json()["user"]["employee_id"] == "EMP-4082"

    bad_emp = client.post(
        "/auth/login",
        json={"role": "employee", "username": "EMP-4082", "password": "wrong"},
    )
    assert bad_emp.status_code == 401

    admin_res = client.post(
        "/auth/login",
        json={"role": "admin", "username": "admin", "password": "admin123"},
    )
    assert admin_res.status_code == 200
    assert admin_res.json()["role"] == "admin"


def test_ticket_lifecycle_and_employee_isolation(client):
    # 1. Clear high-confidence ticket -> Accepted & Routed immediately
    t1 = client.post(
        "/tickets",
        json={
            "ticket_text": "Please order a new ergonomic laptop docking station and monitor for my desk",
            "employee_id": "EMP-4082",
            "user_name": "Arjun Mehta",
            "user_email": "arjun.mehta@smartserve.io",
        },
    )
    assert t1.status_code == 201
    data1 = t1.json()
    assert data1["human_review_required"] is False
    assert data1["status"] == "Accepted & Routed"

    # 2. Vague ticket -> Pending Human Review
    t2 = client.post(
        "/tickets",
        json={
            "ticket_text": "Not working",
            "employee_id": "EMP-5104",
            "user_name": "Priya Nair",
            "user_email": "priya.nair@smartserve.io",
        },
    )
    assert t2.status_code == 201
    data2 = t2.json()
    assert data2["human_review_required"] is True
    assert data2["status"] == "Pending Human Review"

    # 3. Employee filtering returns ONLY that employee's tickets
    emp_4082_tickets = client.get("/tickets?employee_id=EMP-4082").json()
    assert len(emp_4082_tickets) == 1
    assert emp_4082_tickets[0]["employee_id"] == "EMP-4082"

    # 4. Admin approves flagged ticket -> Accepted & Routed
    approved = client.patch(f"/tickets/{data2['ticket_id']}/approve").json()
    assert approved["human_review_required"] is False
    assert approved["status"] == "Accepted & Routed"

    # 5. Admin marks ticket Solved (Closed)
    closed = client.patch(f"/tickets/{data2['ticket_id']}/close").json()
    assert closed["status"] == "Closed"


def test_rejects_invalid_empty_or_symbol_tickets(client):
    res = client.post(
        "/tickets",
        json={"ticket_text": "12345 !!! ???", "employee_id": "EMP-4082"},
    )
    assert res.status_code == 422


def test_120_concurrent_employees_simulation_unique_tickets(client):
    sim1 = client.post("/simulate/employees", json={"employee_count": 120})
    assert sim1.status_code == 200
    body1 = sim1.json()
    assert body1["simulated_employees"] == 120
    assert body1["tickets_processed"] == 120
    assert body1["failed_requests"] == 0

    # Run a second 100-employee simulation and verify zero duplicate sentences across all 220 tickets
    sim2 = client.post("/simulate/employees", json={"employee_count": 100})
    assert sim2.status_code == 200

    all_tickets = client.get("/tickets?limit=300").json()
    assert len(all_tickets) == 220
    unique_texts = {t["ticket_text"] for t in all_tickets}
    assert len(unique_texts) == 220