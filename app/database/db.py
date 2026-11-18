import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("smartserve.db")
REF_DIST_PATH = Path("data/processed/reference_distribution.json")


def _resolve_db(db_path: Path | None = None) -> Path:
    return db_path if db_path is not None else DB_PATH


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(_resolve_db(db_path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: Path | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL DEFAULT 'EMP-4082',
                user_name TEXT NOT NULL DEFAULT 'Arjun Mehta',
                user_email TEXT NOT NULL DEFAULT 'arjun.mehta@smartserve.io',
                ticket_text TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL NOT NULL,
                priority TEXT NOT NULL,
                priority_reason TEXT NOT NULL,
                suggested_team TEXT NOT NULL,
                confidence_band TEXT NOT NULL,
                human_review_required INTEGER NOT NULL,
                review_reason TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                model_version TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Accepted & Routed',
                created_at TEXT NOT NULL
            )
            """
        )
        cols = [row["name"] for row in conn.execute("PRAGMA table_info(tickets)").fetchall()]
        if "status" not in cols:
            conn.execute("ALTER TABLE tickets ADD COLUMN status TEXT NOT NULL DEFAULT 'Accepted & Routed'")
        if "employee_id" not in cols:
            conn.execute("ALTER TABLE tickets ADD COLUMN employee_id TEXT NOT NULL DEFAULT 'EMP-4082'")
        if "user_name" not in cols:
            conn.execute("ALTER TABLE tickets ADD COLUMN user_name TEXT NOT NULL DEFAULT 'Arjun Mehta'")
        if "user_email" not in cols:
            conn.execute("ALTER TABLE tickets ADD COLUMN user_email TEXT NOT NULL DEFAULT 'arjun.mehta@smartserve.io'")
        conn.commit()


def save_ticket(
    ticket_text: str,
    prediction: dict,
    employee_id: str = "EMP-4082",
    user_name: str = "Arjun Mehta",
    user_email: str = "arjun.mehta@smartserve.io",
    db_path: Path | None = None,
) -> dict:
    init_db(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    status = "Pending Human Review" if prediction["human_review_required"] else "Accepted & Routed"
    clean_emp_id = employee_id.strip().upper() or "EMP-4082"
    clean_name = user_name.strip() or "Arjun Mehta"
    clean_email = user_email.strip() or f"{clean_emp_id.lower()}@smartserve.io"

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO tickets (
                employee_id, user_name, user_email, ticket_text, category, confidence,
                priority, priority_reason, suggested_team, confidence_band,
                human_review_required, review_reason, latency_ms,
                model_version, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean_emp_id,
                clean_name,
                clean_email,
                ticket_text,
                prediction["category"],
                prediction["confidence"],
                prediction["priority"],
                prediction["priority_reason"],
                prediction["suggested_team"],
                prediction["confidence_band"],
                int(prediction["human_review_required"]),
                prediction["review_reason"],
                prediction["latency_ms"],
                prediction["model_version"],
                status,
                created_at,
            ),
        )
        conn.commit()
        ticket_id = cursor.lastrowid

    return {
        "ticket_id": ticket_id,
        "employee_id": clean_emp_id,
        "user_name": clean_name,
        "user_email": clean_email,
        "ticket_text": ticket_text,
        **prediction,
        "status": status,
        "created_at": created_at,
    }


def save_tickets_batch(records: list[dict], db_path: Path | None = None) -> None:
    """High-throughput atomic insert for 100+ simulated concurrent employee tickets."""
    init_db(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for item in records:
        pred = item["prediction"]
        status = "Pending Human Review" if pred["human_review_required"] else "Accepted & Routed"
        rows.append(
            (
                item["employee_id"],
                item["user_name"],
                item["user_email"],
                item["ticket_text"],
                pred["category"],
                pred["confidence"],
                pred["priority"],
                pred["priority_reason"],
                pred["suggested_team"],
                pred["confidence_band"],
                int(pred["human_review_required"]),
                pred["review_reason"],
                pred["latency_ms"],
                pred["model_version"],
                status,
                created_at,
            )
        )

    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO tickets (
                employee_id, user_name, user_email, ticket_text, category, confidence,
                priority, priority_reason, suggested_team, confidence_band,
                human_review_required, review_reason, latency_ms,
                model_version, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def list_tickets(limit: int = 300, employee_id: str | None = None, db_path: Path | None = None) -> list[dict]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        if employee_id:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE UPPER(employee_id) = ? ORDER BY ticket_id DESC LIMIT ?",
                (employee_id.strip().upper(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tickets ORDER BY ticket_id DESC LIMIT ?", (limit,)
            ).fetchall()
    results = []
    for r in rows:
        item = dict(r)
        item["human_review_required"] = bool(item["human_review_required"])
        results.append(item)
    return results


def get_ticket_by_id(ticket_id: int, db_path: Path | None = None) -> dict | None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
    if not row:
        return None
    record = dict(row)
    record["human_review_required"] = bool(record["human_review_required"])
    return record


def resolve_ticket_review(ticket_id: int, db_path: Path | None = None) -> dict | None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE tickets SET human_review_required = 0, status = 'Accepted & Routed' WHERE ticket_id = ?",
            (ticket_id,),
        )
        conn.commit()
    return get_ticket_by_id(ticket_id, db_path)


def close_ticket(ticket_id: int, db_path: Path | None = None) -> dict | None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE tickets SET human_review_required = 0, status = 'Closed' WHERE ticket_id = ?",
            (ticket_id,),
        )
        conn.commit()
    return get_ticket_by_id(ticket_id, db_path)


def delete_ticket(ticket_id: int, db_path: Path | None = None) -> bool:
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM tickets WHERE ticket_id = ?", (ticket_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_monitoring_metrics(db_path: Path | None = None) -> dict:
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM tickets").fetchall()

    total = len(rows)
    if total == 0:
        return {
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "low_confidence_pct": 0.0,
            "human_review_pct": 0.0,
            "avg_latency_ms": 0.0,
            "category_distribution": {},
            "team_distribution": {},
            "priority_distribution": {},
            "drift_warning": False,
            "drift_details": "No live tickets recorded yet.",
        }

    avg_conf = round(sum(r["confidence"] for r in rows) / total, 4)
    low_conf_count = sum(1 for r in rows if r["confidence"] < 0.60)
    review_count = sum(1 for r in rows if r["human_review_required"] == 1)
    avg_latency = round(sum(r["latency_ms"] for r in rows) / total, 3)

    counts: dict[str, int] = {}
    team_counts: dict[str, int] = {}
    prio_counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
        team_counts[r["suggested_team"]] = team_counts.get(r["suggested_team"], 0) + 1
        prio_counts[r["priority"]] = prio_counts.get(r["priority"], 0) + 1

    live_dist = {cat: round(cnt / total, 4) for cat, cnt in sorted(counts.items())}

    drift_warning = False
    drift_details = "Insufficient sample size for drift check (need >= 10 tickets)."
    if total >= 10 and REF_DIST_PATH.exists():
        with open(REF_DIST_PATH, "r", encoding="utf-8") as f:
            ref_dist = json.load(f)

        max_shift = 0.0
        shifted_cat = None
        for cat, ref_share in ref_dist.items():
            live_share = live_dist.get(cat, 0.0)
            shift = abs(live_share - ref_share)
            if shift > max_shift:
                max_shift = shift
                shifted_cat = cat

        if max_shift >= 0.20:
            drift_warning = True
            drift_details = (
                f"Distribution shift detected: '{shifted_cat}' shifted by "
                f"{max_shift * 100:.1f}% vs. training baseline."
            )
        else:
            drift_details = f"Distribution stable (max shift {max_shift * 100:.1f}%)."

    return {
        "total_predictions": total,
        "avg_confidence": avg_conf,
        "low_confidence_pct": round((low_conf_count / total) * 100, 2),
        "human_review_pct": round((review_count / total) * 100, 2),
        "avg_latency_ms": avg_latency,
        "category_distribution": live_dist,
        "team_distribution": team_counts,
        "priority_distribution": prio_counts,
        "model_version": rows[-1]["model_version"],
        "drift_warning": drift_warning,
        "drift_details": drift_details,
    }