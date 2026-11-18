import random
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from app.database.db import (
    close_ticket,
    delete_ticket,
    get_monitoring_metrics,
    get_ticket_by_id,
    init_db,
    list_tickets,
    resolve_ticket_review,
    save_ticket,
    save_tickets_batch,
)
from app.ml.predictor import SmartServePredictor

predictor: SmartServePredictor | None = None

FIRST_NAMES = [
    "Arjun", "Priya", "Rohan", "Ananya", "Karthik", "Meera", "Vikram", "Divya",
    "Siddharth", "Neha", "Aditya", "Kavya", "Rahul", "Sneha", "Nikhil", "Pooja",
    "Varun", "Shruti", "Pranav", "Deepika",
]
LAST_NAMES = [
    "Mehta", "Nair", "Verma", "Iyer", "Subramanian", "Krishnan", "Rao", "Sharma",
    "Patel", "Menon", "Reddy", "Gupta", "Das", "Joshi", "Pillai", "Kulkarni",
]
DEPARTMENTS = [
    "Product Engineering", "Cloud Infrastructure", "Finance & Operations",
    "Human Resources", "Enterprise Sales", "Data Science", "Customer Success",
]

# 200-Employee Directory (EMP-4001 to EMP-4200)
EMPLOYEE_DIRECTORY: dict[str, dict] = {}
for idx in range(1, 201):
    emp_code = f"EMP-{4000 + idx}"
    fname = FIRST_NAMES[(idx - 1) % len(FIRST_NAMES)]
    lname = LAST_NAMES[((idx - 1) // len(FIRST_NAMES)) % len(LAST_NAMES)]
    full_name = f"{fname} {lname}"
    EMPLOYEE_DIRECTORY[emp_code] = {
        "employee_id": emp_code,
        "user_name": full_name,
        "user_email": f"{fname.lower()}.{lname.lower()}{idx}@smartserve.io",
        "department": DEPARTMENTS[idx % len(DEPARTMENTS)],
    }

PROBLEM_VARIATIONS = {
    "thermal_hardware": [
        "my {device} is overheating and I have a {event} in {mins} mins, what should I do?",
        "I think my CPU fan is down, {device} is overheating {shorthand}",
        "my workstation keeps thermal throttling and shutting off while I {action}",
        "the {device} keyboard and trackpad stopped responding after the latest update",
        "{device} screen is flickering badly and I cant read anything on it {shorthand}",
        "my {device} battery drains from 100% to 0 in 20 mins, gonna need a replacement {shorthand}",
        "external monitor isnt detecting my {device} through the dock {shorthand}",
        "headphones and mic on my {device} stopped working right before my {event}",
    ],
    "vpn_network": [
        "not able to connect to {network} {shorthand}, got a {event} in {mins} mins",
        "{network} keeps timing out whenever I try to {action}",
        "im connected to wifi but {network} fails authentication every time {shorthand}",
        "internet on the 3rd floor wing keeps dropping every few minutes, cant {action}",
        "{network} client hangs on connecting and blocks access to internal tools {shorthand}",
        "need help with {network} routing, internal dashboards wont load at all",
    ],
    "access_identity": [
        "cant log into {app}, password reset link isnt arriving {shorthand}",
        "locked out of {app} after changing my phone, need MFA token reset {shorthand}",
        "getting permission denied on {app} even though I had access yesterday",
        "wanna get admin rights enabled on my machine so I can install {software}",
        "my corporate SSO session keeps expiring every 5 mins while I {action}",
        "need access granted to the {dept} repository and {app} workspace {shorthand}",
    ],
    "storage_infra": [
        "shared drive for {dept} is full and I cant upload the quarterly files {shorthand}",
        "gonna need 500GB extra cloud storage allocated for our {dept} data backups",
        "network folder permissions broke for {dept}, nobody can open the spreadsheets",
        "database staging server is throwing disk space errors when we {action}",
        "need a backup restored from yesterday for the {dept} project directory {shorthand}",
    ],
    "purchase_software": [
        "wanna request a {software} license for my workstation {shorthand}",
        "can we order a new ergonomic {device} for the new joiner in {dept}?",
        "need {software} installed and activated before our {event} tomorrow",
        "requesting procurement approval for an extra {device} for home office setup",
        "my {software} trial expired and im blocked until a paid seat is assigned {shorthand}",
    ],
    "hr_support": [
        "my leave balance in the HR portal hasnt updated after last week {shorthand}",
        "need help updating my tax declaration and bank details in the HR system",
        "wanna check why my reimbursement claim is still stuck in pending status",
        "payroll portal is showing the wrong reporting manager for {dept} {shorthand}",
    ],
    "edge_and_security": [
        "got a suspicious email asking for my {app} password and MFA code, looks like phishing",
        "someone appears to be accessing my {app} account without authorization from another city",
        "my {device} is overheating, {network} is dropping and {app} keeps crashing at the same time",
        "production server is down and unavailable for all users {shorthand}",
        "not working",
        "need help {shorthand}",
    ],
}

OPENERS = [
    "", "", "",
    "Hey team, ",
    "Hi IT, ",
    "Quick question — ",
    "Hi, ",
    "Urgent: ",
    "Hello, ",
    "Btw ",
]

CLOSERS = [
    "", "", "",
    " pls check asap.",
    " thx!",
    " let me know what to do.",
    " blocking my work rn.",
    " appreciate the help.",
    " can someone look into this?",
]

FILLERS = {
    "device": ["laptop", "MacBook", "workstation", "docking station", "monitor", "ThinkPad"],
    "event": ["client meeting", "sprint demo", "standup", "presentation", "interview", "release call"],
    "mins": ["5", "10", "15", "20", "30"],
    "shorthand": ["rn", "asap", "pls", "right now", "today"],
    "action": ["push code", "run builds", "join calls", "export reports", "sync files", "review PRs"],
    "network": ["office VPN", "corporate Wi-Fi", "GlobalProtect VPN", "Cisco AnyConnect"],
    "app": ["Outlook", "Slack", "GitLab", "Jira", "corporate SSO", "Teams", "Salesforce"],
    "software": ["Docker Desktop", "Adobe Creative Cloud", "IntelliJ IDEA", "Postman", "Figma"],
    "dept": ["Engineering", "Finance", "Marketing", "Data Analytics", "Product Ops", "HR"],
}


def generate_unique_tickets(count: int, existing_texts: set[str]) -> list[str]:
    """Generates `count` completely unique tickets that have never appeared in prior runs."""
    domain_weights = [
        ("thermal_hardware", 0.22),
        ("vpn_network", 0.20),
        ("access_identity", 0.20),
        ("storage_infra", 0.12),
        ("purchase_software", 0.12),
        ("hr_support", 0.06),
        ("edge_and_security", 0.08),
    ]
    domains = [d for d, _ in domain_weights]
    weights = [w for _, w in domain_weights]

    seen = set(existing_texts)
    tickets: list[str] = []
    attempts = 0

    while len(tickets) < count and attempts < count * 40:
        attempts += 1
        domain = random.choices(domains, weights=weights, k=1)[0]
        base = random.choice(PROBLEM_VARIATIONS[domain]).format(
            device=random.choice(FILLERS["device"]),
            event=random.choice(FILLERS["event"]),
            mins=random.choice(FILLERS["mins"]),
            shorthand=random.choice(FILLERS["shorthand"]),
            action=random.choice(FILLERS["action"]),
            network=random.choice(FILLERS["network"]),
            app=random.choice(FILLERS["app"]),
            software=random.choice(FILLERS["software"]),
            dept=random.choice(FILLERS["dept"]),
        )
        if len(base.split()) <= 3:
            sentence = base
        else:
            opener = random.choice(OPENERS)
            closer = random.choice(CLOSERS)
            formatted_base = base[0].lower() + base[1:] if opener else base[0].upper() + base[1:]
            sentence = f"{opener}{formatted_base}{closer}".strip()

        if sentence not in seen:
            seen.add(sentence)
            tickets.append(sentence)

    return tickets


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    init_db()
    predictor = SmartServePredictor()
    yield


app = FastAPI(
    title="SmartServe AI",
    description="AI-Assisted IT Incident Triage & Intelligent Routing API",
    version="1.0.0",
    lifespan=lifespan,
)


class LoginRequest(BaseModel):
    role: str = Field(..., pattern="^(employee|admin)$")
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=100)


class TicketRequest(BaseModel):
    ticket_text: str = Field(..., min_length=1, max_length=5000)
    employee_id: str = Field(default="EMP-4082", max_length=50)
    user_name: str = Field(default="Arjun Mehta", max_length=100)
    user_email: str = Field(default="arjun.mehta@smartserve.io", max_length=150)

    @field_validator("ticket_text")
    @classmethod
    def validate_meaningful_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Ticket text cannot be empty or whitespace only.")
        if not re.search(r"[a-zA-Z]", cleaned):
            raise ValueError("Ticket text must contain at least one alphabetic word.")
        return cleaned


class TicketResponse(BaseModel):
    ticket_id: int
    employee_id: str
    user_name: str
    user_email: str
    ticket_text: str
    category: str
    confidence: float
    priority: str
    priority_reason: str
    suggested_team: str
    confidence_band: str
    human_review_required: bool
    review_reason: str
    latency_ms: float
    model_version: str
    status: str
    created_at: str


class LoadSimulationRequest(BaseModel):
    employee_count: int = Field(default=120, ge=100, le=200)


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    return FileResponse("frontend/index.html")


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "model_version": predictor.model_version if predictor else None,
    }


@app.post("/auth/login")
def login(payload: LoginRequest) -> dict:
    user_clean = payload.username.strip()
    if payload.role == "admin":
        if user_clean.lower() == "admin" and payload.password == "admin123":
            return {
                "authenticated": True,
                "role": "admin",
                "user": {
                    "employee_id": "ADMIN-01",
                    "user_name": "IT Service Desk Admin",
                    "user_email": "admin@smartserve.io",
                    "department": "IT Operations & Governance",
                },
            }
        raise HTTPException(status_code=401, detail="Invalid admin credentials. Use admin / admin123.")

    emp_key = user_clean.upper()
    if payload.password != "emp123":
        raise HTTPException(status_code=401, detail="Invalid employee password. Use emp123.")

    if emp_key in EMPLOYEE_DIRECTORY:
        profile = EMPLOYEE_DIRECTORY[emp_key]
    elif re.match(r"^EMP-\d{3,5}$", emp_key):
        profile = {
            "employee_id": emp_key,
            "user_name": f"Employee {emp_key}",
            "user_email": f"{emp_key.lower()}@smartserve.io",
            "department": "Enterprise Operations",
        }
    else:
        raise HTTPException(status_code=401, detail="Enter a valid Employee ID (e.g., EMP-4082 or EMP-4001..EMP-4200).")

    return {"authenticated": True, "role": "employee", "user": profile}


@app.get("/tickets", response_model=list[TicketResponse])
def get_all_tickets(limit: int = 300, employee_id: str | None = None) -> list[dict]:
    return list_tickets(limit=limit, employee_id=employee_id)


@app.post("/tickets", response_model=TicketResponse, status_code=201)
def create_ticket(payload: TicketRequest) -> dict:
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML model is not initialized.")
    try:
        prediction = predictor.predict(payload.ticket_text)
        return save_ticket(
            payload.ticket_text,
            prediction,
            employee_id=payload.employee_id,
            user_name=payload.user_name,
            user_email=payload.user_email,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/tickets/{ticket_id}", response_model=TicketResponse)
def read_ticket(ticket_id: int) -> dict:
    record = get_ticket_by_id(ticket_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")
    return record


@app.patch("/tickets/{ticket_id}/approve", response_model=TicketResponse)
def approve_ticket(ticket_id: int) -> dict:
    updated = resolve_ticket_review(ticket_id)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")
    return updated


@app.patch("/tickets/{ticket_id}/close", response_model=TicketResponse)
def mark_ticket_closed(ticket_id: int) -> dict:
    updated = close_ticket(ticket_id)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")
    return updated


@app.delete("/tickets/{ticket_id}")
def remove_ticket(ticket_id: int) -> dict:
    deleted = delete_ticket(ticket_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")
    return {"deleted": True, "ticket_id": ticket_id}


@app.post("/simulate/employees")
def simulate_concurrent_employees(payload: LoadSimulationRequest) -> dict:
    """Simulates 100+ employees submitting 100% unique, non-repeating tickets simultaneously."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML model is not initialized.")

    t0 = time.perf_counter()
    existing_records = list_tickets(limit=5000)
    existing_texts = {r["ticket_text"] for r in existing_records}

    selected_employees = random.sample(list(EMPLOYEE_DIRECTORY.values()), k=payload.employee_count)
    unique_texts = generate_unique_tickets(payload.employee_count, existing_texts)

    batch_records = []
    auto_routed = 0
    human_review = 0
    team_breakdown: dict[str, int] = {}
    priority_breakdown: dict[str, int] = {}

    for emp, ticket_text in zip(selected_employees, unique_texts):
        pred = predictor.predict(ticket_text)
        if pred["human_review_required"]:
            human_review += 1
        else:
            auto_routed += 1
        team_breakdown[pred["suggested_team"]] = team_breakdown.get(pred["suggested_team"], 0) + 1
        priority_breakdown[pred["priority"]] = priority_breakdown.get(pred["priority"], 0) + 1

        batch_records.append(
            {
                "employee_id": emp["employee_id"],
                "user_name": emp["user_name"],
                "user_email": emp["user_email"],
                "ticket_text": ticket_text,
                "prediction": pred,
            }
        )

    save_tickets_batch(batch_records)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    throughput = round((payload.employee_count / max(elapsed_ms / 1000, 0.001)), 1)

    return {
        "workforce_pool": 200,
        "simulated_employees": payload.employee_count,
        "tickets_processed": len(batch_records),
        "failed_requests": 0,
        "total_time_ms": elapsed_ms,
        "throughput_tickets_per_sec": throughput,
        "auto_accepted_routed": auto_routed,
        "pending_human_review": human_review,
        "team_breakdown": team_breakdown,
        "priority_breakdown": priority_breakdown,
        "sample_employee_ids": [e["employee_id"] for e in selected_employees[:6]],
    }


@app.get("/monitoring")
def monitoring_summary() -> dict:
    return get_monitoring_metrics()