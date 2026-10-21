import hashlib
import hmac
import html
import re
import secrets
import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

DB_PATH = Path(__file__).resolve().parent / "demo_access.db"
DEFAULT_DEMO_PIN = "1234"
ROOMS = [
    {"Room": "Meeting room", "Occupied": True, "Temperature °C": 22.0, "Light lux": 620},
    {"Room": "Quiet room", "Occupied": False, "Temperature °C": 19.0, "Light lux": 180},
    {"Room": "Focus room", "Occupied": True, "Temperature °C": 20.5, "Light lux": 540},
    {"Room": "Collaboration hub", "Occupied": False, "Temperature °C": 21.5, "Light lux": 430},
    {"Room": "Boardroom", "Occupied": True, "Temperature °C": 23.0, "Light lux": 720},
    {"Room": "Reception lounge", "Occupied": False, "Temperature °C": 20.8, "Light lux": 260},
]
ESTIMATED_DAILY_OPERATING_HOURS = 10
OCCUPIED_ROOM_LOAD_KW = 1.02
UNOCCUPIED_ROOM_LOAD_KW = 0.37
DEFAULT_ALARM_ESCALATION_MINUTES = 30
PASSWORD_HASH_ITERATIONS = 310_000


def normalise_sensor_dataframe(sensor_df):
    if sensor_df is None:
        return pd.DataFrame(columns=["Room", "Occupied", "Temperature °C", "Light lux"])

    cleaned = sensor_df.copy()
    cleaned["Room"] = cleaned["Room"].astype(str)
    cleaned["Occupied"] = cleaned["Occupied"].map(lambda value: bool(value))
    cleaned["Temperature °C"] = pd.to_numeric(cleaned["Temperature °C"], errors="coerce").fillna(21.0)
    cleaned["Light lux"] = pd.to_numeric(cleaned["Light lux"], errors="coerce").fillna(0).astype(float)
    return cleaned


def initialise_database():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS access_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time TEXT NOT NULL,
                test_id TEXT NOT NULL,
                result TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS alarm_followups (
                followup_id TEXT PRIMARY KEY,
                system TEXT NOT NULL,
                room TEXT NOT NULL,
                condition TEXT NOT NULL,
                severity TEXT NOT NULL,
                priority TEXT NOT NULL,
                assigned_to TEXT NOT NULL DEFAULT '',
                due_date TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id TEXT NOT NULL,
                snapshot_time TEXT NOT NULL,
                room TEXT NOT NULL,
                occupied INTEGER NOT NULL,
                temperature_c REAL NOT NULL,
                light_lux REAL NOT NULL,
                UNIQUE (snapshot_id, room)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS room_bookings (
                booking_id TEXT PRIMARY KEY,
                room TEXT NOT NULL,
                title TEXT NOT NULL,
                reserved_by TEXT NOT NULL,
                starts_at TEXT NOT NULL,
                ends_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Scheduled',
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS alarm_lifecycle (
                alarm_id INTEGER PRIMARY KEY AUTOINCREMENT,
                alarm_key TEXT NOT NULL,
                system TEXT NOT NULL,
                room TEXT NOT NULL,
                condition TEXT NOT NULL,
                severity TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                is_present INTEGER NOT NULL DEFAULT 1,
                acknowledged_by TEXT NOT NULL DEFAULT '',
                acknowledged_at TEXT,
                escalated_at TEXT,
                resolved_by TEXT NOT NULL DEFAULT '',
                resolved_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_alarm_lifecycle_present
            ON alarm_lifecycle (alarm_key)
            WHERE is_present = 1
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS alarm_actions (
                action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                alarm_id INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                actor TEXT NOT NULL,
                action_taken TEXT NOT NULL,
                action_time TEXT NOT NULL,
                FOREIGN KEY (alarm_id) REFERENCES alarm_lifecycle (alarm_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS operators (
                operator_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                display_name TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def hash_operator_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PASSWORD_HASH_ITERATIONS,
    ).hex()


def create_operator_account(username, display_name, password):
    username = username.strip()
    display_name = display_name.strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{3,32}", username):
        raise ValueError(
            "Username must be 3–32 characters using letters, numbers, dots, "
            "underscores, or hyphens."
        )
    if not display_name or len(display_name) > 80:
        raise ValueError("Enter a display name between 1 and 80 characters.")
    if len(password) < 10:
        raise ValueError("Choose a password with at least 10 characters.")

    salt = secrets.token_bytes(16).hex()
    password_hash = hash_operator_password(password, salt)
    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.execute(
                """
                INSERT INTO operators (
                    username, display_name, password_salt, password_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    username,
                    display_name,
                    salt,
                    password_hash,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            operator_id = connection.execute(
                "SELECT operator_id FROM operators WHERE username = ?",
                (username,),
            ).fetchone()[0]
    except sqlite3.IntegrityError as error:
        raise ValueError("That username is already in use.") from error
    return get_operator_by_id(operator_id)


def authenticate_operator(username, password):
    with sqlite3.connect(DB_PATH) as connection:
        operator = connection.execute(
            """
            SELECT operator_id, display_name, password_salt, password_hash
            FROM operators
            WHERE username = ? COLLATE NOCASE
            """,
            (username.strip(),),
        ).fetchone()
    if operator is None:
        salt = "00" * 16
        hmac.compare_digest(
            hash_operator_password(password, salt),
            "00" * 32,
        )
        return None

    operator_id, display_name, salt, expected_hash = operator
    actual_hash = hash_operator_password(password, salt)
    if not hmac.compare_digest(actual_hash, expected_hash):
        return None
    return {
        "operator_id": int(operator_id),
        "display_name": display_name,
    }


def get_operator_by_id(operator_id):
    with sqlite3.connect(DB_PATH) as connection:
        operator = connection.execute(
            """
            SELECT operator_id, display_name
            FROM operators
            WHERE operator_id = ?
            """,
            (operator_id,),
        ).fetchone()
    if operator is None:
        return None
    return {
        "operator_id": int(operator[0]),
        "display_name": operator[1],
    }


def get_alarm_lifecycle_key(alarm):
    if alarm["system"] == "HVAC":
        condition_type = "temperature"
    elif alarm["system"] == "Lighting":
        condition_type = "lighting"
    else:
        condition_type = alarm["condition"].split(" at ")[0].split(" (")[0].lower()
    return f"{alarm['system']}|{alarm['room']}|{condition_type}"


def sync_alarm_lifecycle(facility_alarms, escalation_minutes):
    if escalation_minutes <= 0:
        raise ValueError("Alarm escalation time must be greater than zero.")

    now = datetime.now()
    now_text = now.isoformat(timespec="seconds")
    current_alarms = [
        alarm for alarm in facility_alarms if alarm["severity"] != "normal"
    ]
    current_keys = {get_alarm_lifecycle_key(alarm) for alarm in current_alarms}
    with sqlite3.connect(DB_PATH) as connection:
        present_records = connection.execute(
            "SELECT alarm_key FROM alarm_lifecycle WHERE is_present = 1"
        ).fetchall()
        missing_keys = [
            record[0] for record in present_records if record[0] not in current_keys
        ]
        if missing_keys:
            placeholders = ",".join("?" for _ in missing_keys)
            connection.execute(
                f"""
                UPDATE alarm_lifecycle
                SET is_present = 0
                WHERE is_present = 1 AND alarm_key IN ({placeholders})
                """,
                missing_keys,
            )

        for alarm in current_alarms:
            alarm_key = get_alarm_lifecycle_key(alarm)
            existing = connection.execute(
                """
                SELECT alarm_id
                FROM alarm_lifecycle
                WHERE alarm_key = ? AND is_present = 1
                """,
                (alarm_key,),
            ).fetchone()
            if existing:
                connection.execute(
                    """
                    UPDATE alarm_lifecycle
                    SET system = ?, room = ?, condition = ?, severity = ?,
                        last_seen_at = ?
                    WHERE alarm_id = ?
                    """,
                    (
                        alarm["system"],
                        alarm["room"],
                        alarm["condition"],
                        alarm["severity"],
                        now_text,
                        existing[0],
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO alarm_lifecycle (
                        alarm_key, system, room, condition, severity,
                        first_seen_at, last_seen_at, is_present
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        alarm_key,
                        alarm["system"],
                        alarm["room"],
                        alarm["condition"],
                        alarm["severity"],
                        now_text,
                        now_text,
                    ),
                )

        open_records = connection.execute(
            """
            SELECT alarm_id, first_seen_at, system, room, condition
            FROM alarm_lifecycle
            WHERE resolved_at IS NULL AND escalated_at IS NULL
            """
        ).fetchall()
        for alarm_id, first_seen_at, system, room, condition in open_records:
            opened_at = datetime.fromisoformat(first_seen_at)
            if now - opened_at >= timedelta(minutes=escalation_minutes):
                connection.execute(
                    """
                    UPDATE alarm_lifecycle
                    SET escalated_at = ?
                    WHERE alarm_id = ? AND escalated_at IS NULL
                    """,
                    (now_text, alarm_id),
                )
                connection.execute(
                    """
                    INSERT INTO alarm_actions (
                        alarm_id, action_type, actor, action_taken, action_time
                    )
                    VALUES (?, 'Escalated', 'System', ?, ?)
                    """,
                    (
                        alarm_id,
                        f"Unresolved for at least {escalation_minutes} minutes: "
                        f"{system} · {room} — {condition}",
                        now_text,
                    ),
                )
    get_alarm_lifecycle_records.clear()
    get_alarm_action_log.clear()


@st.cache_data
def get_alarm_lifecycle_records(include_resolved=False, limit=50):
    status_filter = "" if include_resolved else "WHERE resolved_at IS NULL"
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(
            f"""
            SELECT alarm_id, alarm_key, system, room, condition, severity,
                   first_seen_at, last_seen_at, is_present, acknowledged_by,
                   acknowledged_at, escalated_at, resolved_by, resolved_at
            FROM alarm_lifecycle
            {status_filter}
            ORDER BY
                CASE WHEN resolved_at IS NULL THEN 0 ELSE 1 END,
                CASE WHEN escalated_at IS NOT NULL THEN 0 ELSE 1 END,
                first_seen_at DESC
            LIMIT ?
            """,
            connection,
            params=(limit,),
        )


@st.cache_data
def get_alarm_action_log(alarm_id):
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(
            """
            SELECT action_time, action_type, actor, action_taken
            FROM alarm_actions
            WHERE alarm_id = ?
            ORDER BY action_id DESC
            """,
            connection,
            params=(alarm_id,),
        )


def update_alarm_lifecycle(alarm_id, action_type, actor, action_taken):
    allowed_actions = {"Acknowledged", "Action recorded", "Resolved"}
    if action_type not in allowed_actions:
        raise ValueError(f"Unsupported alarm action: {action_type}")
    if not actor.strip() or not action_taken.strip():
        raise ValueError("Enter your name and describe the action taken.")

    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as connection:
        record = connection.execute(
            "SELECT resolved_at FROM alarm_lifecycle WHERE alarm_id = ?",
            (alarm_id,),
        ).fetchone()
        if record is None:
            raise ValueError("The selected alarm is no longer available.")
        if record[0] is not None:
            raise ValueError("This alarm has already been resolved.")

        if action_type == "Acknowledged":
            connection.execute(
                """
                UPDATE alarm_lifecycle
                SET acknowledged_by = ?, acknowledged_at = ?
                WHERE alarm_id = ? AND acknowledged_at IS NULL
                """,
                (actor.strip(), now, alarm_id),
            )
        elif action_type == "Resolved":
            connection.execute(
                """
                UPDATE alarm_lifecycle
                SET resolved_by = ?, resolved_at = ?
                WHERE alarm_id = ? AND resolved_at IS NULL
                """,
                (actor.strip(), now, alarm_id),
            )

        connection.execute(
            """
            INSERT INTO alarm_actions (
                alarm_id, action_type, actor, action_taken, action_time
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (alarm_id, action_type, actor.strip(), action_taken.strip(), now),
        )
    get_alarm_lifecycle_records.clear()
    get_alarm_action_log.clear()


@st.fragment(run_every="1m", key="alarm-lifecycle-console")
def render_alarm_lifecycle_console(facility_alarms, snapshot_time, operator_name):
    escalation_minutes = st.selectbox(
        "Escalate unresolved alarms after",
        options=[15, 30, 60, 120],
        index=1,
        format_func=lambda minutes: f"{minutes} minutes",
        key="alarm_escalation_minutes",
        help="Unresolved alarms are escalated automatically when they reach this age.",
    )
    sync_alarm_lifecycle(facility_alarms, escalation_minutes)
    alarm_records = get_alarm_lifecycle_records().to_dict(orient="records")
    severity_filter = st.selectbox(
        "Filter alarms",
        options=["All", "Unacknowledged", "Escalated", "Critical", "Alert", "Watch"],
        key="bms_alarm_filter",
    )
    severity_value = severity_filter.lower()
    visible_alarms = [
        alarm
        for alarm in alarm_records
        if severity_filter == "All"
        or (
            severity_filter == "Unacknowledged"
            and pd.isna(alarm["acknowledged_at"])
        )
        or (
            severity_filter == "Escalated"
            and pd.notna(alarm["escalated_at"])
        )
        or (
            severity_filter in {"Critical", "Alert", "Watch"}
            and alarm["severity"] == severity_value
        )
    ]
    escalated_count = sum(
        pd.notna(alarm["escalated_at"]) for alarm in alarm_records
    )
    unacknowledged_count = sum(
        pd.isna(alarm["acknowledged_at"]) for alarm in alarm_records
    )
    st.caption(
        f"{len(alarm_records)} unresolved · {unacknowledged_count} unacknowledged "
        f"· {escalated_count} escalated"
    )

    with st.container(border=True):
        st.markdown(
            f'<div class="section-tag">Persistent alarm lifecycle · {snapshot_time}</div>',
            unsafe_allow_html=True,
        )
        if not visible_alarms:
            st.success("No unresolved alarms match this filter.")
        for alarm in visible_alarms:
            alarm_id = int(alarm["alarm_id"])
            is_escalated = pd.notna(alarm["escalated_at"])
            is_acknowledged = pd.notna(alarm["acknowledged_at"])
            lifecycle_status = (
                "Escalated"
                if is_escalated
                else "Acknowledged"
                if is_acknowledged
                else "Open"
            )
            with st.expander(
                f"{alarm['system']} · {alarm['room']} — {lifecycle_status}",
                expanded=is_escalated,
            ):
                alarm_columns = st.columns([3, 1])
                with alarm_columns[0]:
                    st.markdown(
                        f"**{alarm['condition']}**  \n"
                        f"Opened {alarm['first_seen_at']} · "
                        f"Last detected {alarm['last_seen_at']}"
                    )
                    if not bool(alarm["is_present"]):
                        st.info(
                            "The sensor condition has cleared, but this alarm still needs "
                            "operator resolution."
                        )
                    if is_acknowledged:
                        st.caption(
                            f"Acknowledged by {alarm['acknowledged_by']} "
                            f"at {alarm['acknowledged_at']}."
                        )
                    if is_escalated:
                        st.warning(
                            f"Escalated at {alarm['escalated_at']} after "
                            f"{escalation_minutes} minutes unresolved."
                        )
                with alarm_columns[1]:
                    st.markdown(
                        f'<span class="priority-pill {alarm["severity"]}">'
                        f'{alarm["severity"].upper()}</span>',
                        unsafe_allow_html=True,
                    )

                action_log = get_alarm_action_log(alarm_id)
                if not action_log.empty:
                    st.dataframe(
                        action_log.rename(
                            columns={
                                "action_time": "Time",
                                "action_type": "Event",
                                "actor": "By",
                                "action_taken": "Action taken",
                            }
                        ),
                        hide_index=True,
                        width="stretch",
                        alt=f"Action history for {alarm['system']} alarm in {alarm['room']}",
                    )

                with st.form(f"alarm_lifecycle_form_{alarm_id}"):
                    action_taken = st.text_area(
                        "Action taken",
                        placeholder="Describe the investigation, mitigation, or resolution.",
                        key=f"alarm_action_note_{alarm_id}",
                    )
                    action_cols = st.columns(3)
                    acknowledge_clicked = action_cols[0].form_submit_button(
                        "Acknowledge",
                        disabled=is_acknowledged,
                        width="stretch",
                    )
                    record_clicked = action_cols[1].form_submit_button(
                        "Record action",
                        width="stretch",
                    )
                    resolve_clicked = action_cols[2].form_submit_button(
                        "Resolve",
                        type="primary",
                        width="stretch",
                    )

                action_type = (
                    "Acknowledged"
                    if acknowledge_clicked
                    else "Action recorded"
                    if record_clicked
                    else "Resolved"
                    if resolve_clicked
                    else None
                )
                if action_type:
                    try:
                        update_alarm_lifecycle(
                            alarm_id,
                            action_type,
                            operator_name,
                            action_taken,
                        )
                    except ValueError as error:
                        st.error(str(error))
                    else:
                        st.rerun()

    resolved_records = get_alarm_lifecycle_records(
        include_resolved=True,
        limit=100,
    )
    resolved_records = resolved_records[resolved_records["resolved_at"].notna()]
    with st.expander("Recently resolved alarms"):
        if resolved_records.empty:
            st.caption("Resolved alarms will appear here.")
        else:
            st.dataframe(
                resolved_records[
                    [
                        "system",
                        "room",
                        "condition",
                        "severity",
                        "first_seen_at",
                        "resolved_by",
                        "resolved_at",
                    ]
                ].rename(
                    columns={
                        "system": "System",
                        "room": "Room",
                        "condition": "Alarm",
                        "severity": "Severity",
                        "first_seen_at": "Opened",
                        "resolved_by": "Resolved by",
                        "resolved_at": "Resolved at",
                    }
                ),
                hide_index=True,
                width="stretch",
                alt="Recently resolved alarm history",
            )


def save_access_event(test_id, result):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO access_events (event_time, test_id, result)
            VALUES (?, ?, ?)
            """,
            (datetime.now().isoformat(timespec="seconds"), test_id, result),
        )
    get_access_events.clear()


@st.cache_data
def get_access_events():
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(
            """
            SELECT event_time, test_id, result
            FROM access_events
            ORDER BY id DESC
            LIMIT 10
            """,
            connection,
        )


def save_alarm_followup(alarm, priority, assigned_to, due_date, notes, status):
    allowed_priorities = {"Critical", "High", "Medium", "Low"}
    allowed_statuses = {"Open", "In progress", "Resolved"}
    if priority not in allowed_priorities:
        raise ValueError(f"Unsupported alarm priority: {priority}")
    if status not in allowed_statuses:
        raise ValueError(f"Unsupported follow-up status: {status}")

    now = datetime.now().isoformat(timespec="seconds")
    followup_id = get_alarm_followup_id(alarm)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO alarm_followups (
                followup_id, system, room, condition, severity, priority,
                assigned_to, due_date, notes, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(followup_id) DO UPDATE SET
                system = excluded.system,
                room = excluded.room,
                condition = excluded.condition,
                severity = excluded.severity,
                priority = excluded.priority,
                assigned_to = excluded.assigned_to,
                due_date = excluded.due_date,
                notes = excluded.notes,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                followup_id,
                alarm["system"],
                alarm["room"],
                alarm["condition"],
                alarm["severity"],
                priority,
                assigned_to.strip(),
                due_date.isoformat(),
                notes.strip(),
                status,
                now,
                now,
            ),
        )
    get_alarm_followups.clear()


@st.cache_data
def get_alarm_followups():
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(
            """
            SELECT followup_id, system, room, condition, severity, priority,
                   assigned_to, due_date, notes, status, created_at, updated_at
            FROM alarm_followups
            ORDER BY
                CASE status WHEN 'Open' THEN 0 WHEN 'In progress' THEN 1 ELSE 2 END,
                due_date ASC,
                updated_at DESC
            """,
            connection,
        )


@st.cache_data
def get_sensor_data():
    return normalise_sensor_dataframe(pd.DataFrame(ROOMS))


def save_sensor_snapshot(sensor_df):
    if sensor_df.empty:
        raise ValueError("Cannot save an empty sensor snapshot.")

    snapshot_id = str(uuid4())
    snapshot_time = datetime.now().isoformat(timespec="seconds")
    rows = [
        (
            snapshot_id,
            snapshot_time,
            str(room["Room"]),
            int(bool(room["Occupied"])),
            float(room["Temperature °C"]),
            float(room["Light lux"]),
        )
        for _, room in sensor_df.iterrows()
    ]
    with sqlite3.connect(DB_PATH) as connection:
        connection.executemany(
            """
            INSERT INTO sensor_readings (
                snapshot_id, snapshot_time, room, occupied, temperature_c, light_lux
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    get_historical_readings.clear()
    return snapshot_id


@st.cache_data
def get_historical_readings(start_time, end_time, selected_rooms):
    if not selected_rooms:
        return pd.DataFrame(
            columns=[
                "snapshot_id",
                "snapshot_time",
                "room",
                "occupied",
                "temperature_c",
                "light_lux",
            ]
        )

    room_placeholders = ", ".join("?" for _ in selected_rooms)
    query = f"""
        SELECT snapshot_id, snapshot_time, room, occupied, temperature_c, light_lux
        FROM sensor_readings
        WHERE snapshot_time >= ?
          AND snapshot_time < ?
          AND room IN ({room_placeholders})
        ORDER BY snapshot_time ASC, room ASC
    """
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(
            query,
            connection,
            params=(start_time, end_time, *selected_rooms),
        )


def get_sensor_snapshot_count():
    with sqlite3.connect(DB_PATH) as connection:
        result = connection.execute(
            "SELECT COUNT(DISTINCT snapshot_id) FROM sensor_readings"
        ).fetchone()
    return int(result[0])


def create_room_booking(room, title, reserved_by, starts_at, ends_at):
    if room not in {item["Room"] for item in ROOMS}:
        raise ValueError("Select a valid room.")
    if not title.strip() or not reserved_by.strip():
        raise ValueError("Provide a meeting title and the name of the person booking.")
    if starts_at >= ends_at:
        raise ValueError("The booking end time must be after the start time.")
    if starts_at < datetime.now():
        raise ValueError("Bookings must start in the future.")

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("BEGIN IMMEDIATE")
        conflict = connection.execute(
            """
            SELECT 1
            FROM room_bookings
            WHERE room = ?
              AND status = 'Scheduled'
              AND starts_at < ?
              AND ends_at > ?
            LIMIT 1
            """,
            (room, ends_at.isoformat(timespec="minutes"), starts_at.isoformat(timespec="minutes")),
        ).fetchone()
        if conflict:
            raise ValueError("That room already has a booking during this time.")
        booking_id = str(uuid4())
        connection.execute(
            """
            INSERT INTO room_bookings (
                booking_id, room, title, reserved_by, starts_at, ends_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                booking_id,
                room,
                title.strip(),
                reserved_by.strip(),
                starts_at.isoformat(timespec="minutes"),
                ends_at.isoformat(timespec="minutes"),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
    get_room_bookings.clear()
    return booking_id


@st.cache_data
def get_room_bookings(start_time, end_time):
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(
            """
            SELECT booking_id, room, title, reserved_by, starts_at, ends_at
            FROM room_bookings
            WHERE status = 'Scheduled'
              AND starts_at < ?
              AND ends_at > ?
            ORDER BY starts_at ASC, room ASC
            """,
            connection,
            params=(end_time, start_time),
        )


def cancel_room_booking(booking_id):
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            """
            UPDATE room_bookings
            SET status = 'Cancelled'
            WHERE booking_id = ? AND status = 'Scheduled'
            """,
            (booking_id,),
        )
        if cursor.rowcount != 1:
            raise ValueError("The selected booking is no longer active.")
    get_room_bookings.clear()


def build_dashboard_summary(sensor_data, manual_override, ai_energy_mode, snapshot_time):
    occupied_count = int(sensor_data["Occupied"].sum())
    avg_temperature = float(sensor_data["Temperature °C"].mean())
    energy_saving_rooms = int((~sensor_data["Occupied"]).sum())
    heat_alerts = int((sensor_data["Temperature °C"] > 22.5).sum())
    light_policy = "Override" if manual_override else ("AI mode" if ai_energy_mode else "Manual")
    energy_score = get_energy_score(sensor_data, manual_override, ai_energy_mode)
    facility_alarms = build_facility_alarms(sensor_data, snapshot_time)
    sync_alarm_lifecycle(
        facility_alarms,
        st.session_state.get(
            "alarm_escalation_minutes",
            DEFAULT_ALARM_ESCALATION_MINUTES,
        ),
    )
    alarm_lifecycle = get_alarm_lifecycle_records()
    active_alarm_count = sum(alarm["severity"] != "normal" for alarm in facility_alarms)
    unacknowledged_alarm_count = sum(
        pd.isna(alarm["acknowledged_at"])
        for alarm in alarm_lifecycle.to_dict(orient="records")
    )

    return {
        "occupied_count": occupied_count,
        "avg_temperature": avg_temperature,
        "energy_saving_rooms": energy_saving_rooms,
        "heat_alerts": heat_alerts,
        "light_policy": light_policy,
        "energy_score": energy_score,
        "facility_alarms": facility_alarms,
        "active_alarm_count": active_alarm_count,
        "unacknowledged_alarm_count": unacknowledged_alarm_count,
    }


def get_room_efficiency_profile(room, manual_override=False, ai_mode=True):
    occupied = bool(room["Occupied"])
    temperature = float(room["Temperature °C"])
    light_level = float(room["Light lux"])
    score = 100

    if manual_override:
        score -= 8
    if not ai_mode:
        score -= 16

    if occupied:
        score -= max(0, (temperature - 22.5) * 8)
        score -= max(0, (light_level - 500) / 15)
    else:
        score -= max(0, (400 - light_level) / 10) if light_level < 400 else max(0, (light_level - 400) / 18)
        if temperature < 19.5:
            score += 5

    if not occupied and light_level >= 450:
        score -= 18
    if occupied and light_level <= 250:
        score -= 8
    if temperature > 25.5:
        score -= 15
    if temperature < 18.5:
        score += 6

    score = max(0, min(100, round(score)))
    if score >= 85:
        status = "Excellent"
        action = "Keep current operating profile."
    elif score >= 70:
        status = "Stable"
        action = "Fine-tune lighting and temperature for comfort."
    elif score >= 50:
        status = "Watch"
        action = "Reduce lighting or adjust HVAC for efficiency."
    else:
        status = "Critical"
        action = "Immediate optimisation recommended."

    return {
        "score": score,
        "status": status,
        "action": action,
        "occupied": occupied,
    }


def get_recommendation(occupied, manual_override, ai_mode=True):
    room_like = {"Occupied": occupied, "Temperature °C": 21.5, "Light lux": 400}
    profile = get_room_efficiency_profile(room_like, manual_override, ai_mode)
    if manual_override:
        return "Manual override active; keep simulated lights on."
    if not ai_mode:
        return "AI energy optimisation disabled; keep simulated lights on."
    if occupied:
        return f"Room occupied; simulated lights may remain on. Efficiency profile: {profile['status']} ({profile['score']}/100)."
    return f"Room empty; recommend simulated lights off. Efficiency profile: {profile['status']} ({profile['score']}/100)."


def build_room_recommendation_table(sensor_df, manual_override, ai_mode):
    rows = []
    for _, room in sensor_df.iterrows():
        room_profile = get_room_efficiency_profile(room, manual_override, ai_mode)
        if room_profile["occupied"]:
            primary_action = "Maintain current lighting and airflow"
        elif room["Light lux"] >= 400:
            primary_action = "Reduce lighting to save energy"
        else:
            primary_action = "Keep lighting low and monitor occupancy"

        rows.append(
            {
                "Room": room["Room"],
                "Status": room_profile["status"],
                "Efficiency score": room_profile["score"],
                "Action": primary_action,
                "Priority": "High" if room_profile["score"] < 60 else "Medium" if room_profile["score"] < 80 else "Low",
            }
        )
    return pd.DataFrame(rows)


def score_room_anomaly(room):
    occupied = bool(room["Occupied"])
    temperature = float(room["Temperature °C"])
    light_level = float(room["Light lux"])
    risk_score = 0
    reasons = []

    if temperature >= 25.5:
        risk_score += 30
        reasons.append("temperature above comfort band")
    elif temperature >= 23.5:
        risk_score += 14
        reasons.append("temperature approaching upper threshold")

    if occupied and light_level > 700:
        risk_score += 18
        reasons.append("over-lighting while occupied")
    elif not occupied and light_level >= 450:
        risk_score += 22
        reasons.append("lighting left on while vacant")
    elif occupied and light_level < 250:
        risk_score += 10
        reasons.append("under-lighting in active zone")

    if not occupied and (temperature < 19.0 or temperature > 23.0):
        risk_score += 8
        reasons.append("vacant-space thermal inefficiency")

    if risk_score >= 40:
        severity = "Critical"
    elif risk_score >= 24:
        severity = "High"
    elif risk_score >= 12:
        severity = "Medium"
    else:
        severity = "Low"

    return {
        "risk_score": risk_score,
        "severity": severity,
        "reasons": reasons or ["within expected operating range"],
    }


def build_ai_optimization_summary(sensor_df, manual_override=False, ai_mode=True):
    room_scores = []
    for _, room in sensor_df.iterrows():
        anomaly = score_room_anomaly(room)
        profile = get_room_efficiency_profile(room, manual_override, ai_mode)
        room_scores.append(
            {
                "Room": room["Room"],
                "Occupied": bool(room["Occupied"]),
                "Efficiency score": profile["score"],
                "Anomaly score": anomaly["risk_score"],
                "Severity": anomaly["severity"],
                "Key issue": "; ".join(anomaly["reasons"][:2]),
            }
        )

    room_summary = pd.DataFrame(room_scores)
    average_efficiency = room_summary["Efficiency score"].mean()
    average_risk = room_summary["Anomaly score"].mean()
    candidate_rooms = room_summary.sort_values(
        ["Anomaly score", "Efficiency score"], ascending=[False, True]
    ).head(3)

    projected_savings = max(
        4,
        int(round((100 - average_efficiency) * 0.7 + (average_risk * 0.25))),
    )
    if manual_override:
        projected_savings = max(projected_savings, 7)
    if not ai_mode:
        projected_savings = max(projected_savings, 3)

    recommended_mode = "AI-assisted optimisation" if ai_mode else "Manual control"
    if manual_override:
        recommended_mode = "Manual override should be reviewed"

    return {
        "average_efficiency": round(float(average_efficiency), 1),
        "average_risk": round(float(average_risk), 1),
        "projected_savings_pct": projected_savings,
        "recommendation": f"{recommended_mode} is recommended to improve room comfort and reduce unnecessary energy draw.",
        "top_candidates": candidate_rooms,
        "room_summary": room_summary,
    }


def estimate_energy_costs(sensor_df, electricity_price, projected_savings_pct):
    if electricity_price < 0:
        raise ValueError("Electricity price cannot be negative.")
    if not 0 <= projected_savings_pct <= 100:
        raise ValueError("Projected savings must be between 0 and 100.")

    connected_load_kw = sum(
        OCCUPIED_ROOM_LOAD_KW if bool(room["Occupied"]) else UNOCCUPIED_ROOM_LOAD_KW
        for _, room in sensor_df.iterrows()
    )
    daily_usage_kwh = connected_load_kw * ESTIMATED_DAILY_OPERATING_HOURS
    daily_cost = daily_usage_kwh * electricity_price
    daily_savings_kwh = daily_usage_kwh * projected_savings_pct / 100
    daily_cost_savings = daily_savings_kwh * electricity_price
    return {
        "daily_usage_kwh": daily_usage_kwh,
        "daily_cost": daily_cost,
        "daily_savings_kwh": daily_savings_kwh,
        "daily_cost_savings": daily_cost_savings,
    }


def build_operations_brief(sensor_df, manual_override=False, ai_mode=True, facility_alarms=None):
    optimization = build_ai_optimization_summary(sensor_df, manual_override, ai_mode)
    action_plan = build_optimization_action_plan(
        sensor_df,
        manual_override,
        ai_mode,
        target_efficiency=85,
    )
    forecast = build_occupancy_forecast(sensor_df, horizon_hours=4)
    occupancy_risk = round(float(forecast["Probability"].mean()), 2) if not forecast.empty else 0.0
    alarms = facility_alarms or []
    active_alarm_count = sum(1 for alarm in alarms if alarm.get("severity") != "normal")
    next_best_action = (
        action_plan.iloc[0]["Action"]
        if not action_plan.empty and "Action" in action_plan.columns
        else "Continue current adaptive control cycle"
    )

    return {
        "avg_efficiency": optimization["average_efficiency"],
        "risk_level": optimization["average_risk"],
        "priority_actions": int((action_plan["Priority"] == "High").sum()) if not action_plan.empty else 0,
        "forecast_risk": occupancy_risk,
        "active_alarms": active_alarm_count,
        "recommendation": optimization["recommendation"],
        "next_best_action": next_best_action,
    }


def build_optimization_action_plan(sensor_df, manual_override=False, ai_mode=True, target_efficiency=85):
    actions = []
    for _, room in sensor_df.iterrows():
        profile = get_room_efficiency_profile(room, manual_override, ai_mode)
        anomaly = score_room_anomaly(room)
        room_actions = []

        if bool(room["Occupied"]):
            if float(room["Temperature °C"]) > 23.5:
                room_actions.append("Reduce HVAC setpoint by 1-2°C to protect comfort margins")
            if float(room["Light lux"]) > 650:
                room_actions.append("Dim perimeter lighting by 10-20% during active use")
            if not room_actions:
                room_actions.append("Maintain current thermal profile and continue adaptive lighting")
        else:
            if float(room["Light lux"]) >= 400:
                room_actions.append("Cut lighting to occupancy-safe standby level")
            if float(room["Temperature °C"]) < 19.0 or float(room["Temperature °C"]) > 23.0:
                room_actions.append("Reset idle-zone temperature drift to the target comfort band")
            if not room_actions:
                room_actions.append("Keep vacancy mode active and monitor for unexpected use")

        expected_reduction = max(
            3,
            min(22, int(round((100 - profile["score"]) * 0.55 + anomaly["risk_score"] * 0.28))),
        )
        priority_label = (
            "High"
            if anomaly["severity"] in {"Critical", "High"} or profile["score"] < target_efficiency
            else "Medium" if profile["score"] < target_efficiency + 8 else "Low"
        )
        actions.append(
            {
                "Room": room["Room"],
                "Priority": priority_label,
                "Efficiency score": profile["score"],
                "Expected savings": f"{expected_reduction}%",
                "Action": "; ".join(room_actions[:2]),
                "Impact": "High" if expected_reduction >= 14 else "Medium" if expected_reduction >= 8 else "Low",
            }
        )

    plan = pd.DataFrame(actions)
    plan["Priority_rank"] = plan["Priority"].map({"High": 0, "Medium": 1, "Low": 2})
    return plan.sort_values(["Priority_rank", "Efficiency score"]).drop(columns=["Priority_rank"]).reset_index(drop=True)


def build_daily_handover_summary(
    sensor_df,
    facility_alarms,
    followups,
    alarm_lifecycle=None,
    manual_override=False,
    ai_mode=True,
):
    if alarm_lifecycle is not None:
        active_alarms = alarm_lifecycle.copy()
        if active_alarms.empty:
            active_alarms = pd.DataFrame(
                columns=[
                    "severity",
                    "system",
                    "room",
                    "condition",
                    "acknowledged_by",
                    "acknowledged_at",
                    "first_seen_at",
                    "escalated_at",
                ]
            )
        active_alarms["Status"] = active_alarms.apply(
            lambda alarm: (
                "Escalated"
                if pd.notna(alarm["escalated_at"])
                else "Acknowledged"
                if pd.notna(alarm["acknowledged_at"])
                else "Open"
            ),
            axis=1,
        )
        active_alarms = active_alarms[
            [
                "severity",
                "system",
                "room",
                "condition",
                "Status",
                "acknowledged_by",
                "first_seen_at",
                "escalated_at",
            ]
        ].rename(
            columns={
                "severity": "Severity",
                "system": "System",
                "room": "Room",
                "condition": "Alarm",
                "acknowledged_by": "Acknowledged by",
                "first_seen_at": "Opened",
                "escalated_at": "Escalated at",
            }
        )
    else:
        active_alarms = pd.DataFrame(
            [alarm for alarm in facility_alarms if alarm["severity"] != "normal"]
        )
        if active_alarms.empty:
            active_alarms = pd.DataFrame(
                columns=["Severity", "System", "Room", "Alarm"]
            )
        else:
            active_alarms = active_alarms[
                ["severity", "system", "room", "condition"]
            ].rename(
                columns={
                    "severity": "Severity",
                    "system": "System",
                    "room": "Room",
                    "condition": "Alarm",
                }
            )

    unresolved_tasks = followups[followups["status"] != "Resolved"].copy()
    if unresolved_tasks.empty:
        overdue_tasks = unresolved_tasks.copy()
    else:
        overdue_tasks = unresolved_tasks[
            unresolved_tasks["due_date"].map(
                lambda due_date: date.fromisoformat(due_date) < date.today()
            )
        ].copy()
        overdue_tasks = overdue_tasks.rename(
            columns={
                "system": "System",
                "room": "Room",
                "condition": "Alarm",
                "priority": "Priority",
                "assigned_to": "Assigned to",
                "due_date": "Due date",
                "status": "Status",
                "notes": "Notes",
            }
        )

    room_attention = sensor_df.copy()
    room_attention["Severity"] = room_attention.apply(get_room_severity, axis=1)
    room_attention = room_attention[room_attention["Severity"] != "normal"][
        ["Room", "Severity", "Occupied", "Temperature °C", "Light lux"]
    ].reset_index(drop=True)

    recommended_actions = build_optimization_action_plan(
        sensor_df,
        manual_override,
        ai_mode,
        target_efficiency=85,
    ).head(5)[["Room", "Priority", "Action", "Expected savings"]]

    return {
        "active_alarms": active_alarms,
        "unresolved_tasks": unresolved_tasks,
        "overdue_tasks": overdue_tasks,
        "room_attention": room_attention,
        "recommended_actions": recommended_actions,
    }


def build_occupancy_forecast(sensor_df, horizon_hours=6):
    now = datetime.now()
    forecasts = []

    for _, room in sensor_df.iterrows():
        base_occupied = bool(room["Occupied"])
        temp = float(room["Temperature °C"])
        light = float(room["Light lux"])

        for offset in range(horizon_hours):
            hour = (now.hour + offset) % 24
            workday_signal = 0.72 if 8 <= hour <= 18 else 0.26
            occupancy_probability = workday_signal
            if base_occupied:
                occupancy_probability += 0.12
            if light > 550:
                occupancy_probability += 0.08
            if temp > 24:
                occupancy_probability += 0.06
            if not base_occupied:
                occupancy_probability -= 0.19
            if hour in {0, 1, 2, 3, 4, 5, 23}:
                occupancy_probability -= 0.18

            occupancy_probability = max(0.05, min(0.98, occupancy_probability))
            action = (
                "Hold active operation"
                if occupancy_probability >= 0.7
                else "Shift to standby mode" if occupancy_probability >= 0.4 else "Standard energy-save mode"
            )
            forecasts.append(
                {
                    "Hour": (now + timedelta(hours=offset)).strftime("%H:%M"),
                    "Room": room["Room"],
                    "Probability": round(occupancy_probability, 2),
                    "Recommended action": action,
                    "Projected light level": round(max(150, light * (1 - 0.15 * offset / max(1, horizon_hours))), 0),
                }
            )

    forecast_df = pd.DataFrame(forecasts)
    forecast_df["Occupancy outlook"] = forecast_df["Probability"].map(
        lambda value: "Likely occupied" if value >= 0.7 else "Likely vacant" if value <= 0.3 else "Uncertain"
    )
    return forecast_df


def build_policy_scenario(sensor_df, mode_name="Adaptive AI"):
    policy_profiles = {
        "Adaptive AI": {"light_bias": -0.12, "temp_bias": -0.6, "comfort_weight": 0.8},
        "Eco Max": {"light_bias": -0.22, "temp_bias": -1.1, "comfort_weight": 0.55},
        "Comfort First": {"light_bias": 0.08, "temp_bias": 0.3, "comfort_weight": 1.05},
        "Manual Override": {"light_bias": 0.0, "temp_bias": 0.0, "comfort_weight": 0.7},
    }
    profile = policy_profiles.get(mode_name, policy_profiles["Adaptive AI"])
    room_rows = []

    for _, room in sensor_df.iterrows():
        occupied = bool(room["Occupied"])
        temperature = float(room["Temperature °C"])
        light = float(room["Light lux"])
        delta_temp = abs(temperature - 22.0)
        delta_light = max(0, light - 500)
        operative_score = 100

        if occupied:
            operative_score -= delta_temp * 8
            operative_score -= delta_light / 14
        else:
            operative_score -= max(0, 400 - light) / 12
            operative_score -= max(0, delta_temp - 1.4) * 7

        operative_score += profile["light_bias"] * 160
        operative_score += profile["temp_bias"] * 30
        operative_score = max(0, min(100, round(operative_score)))

        comfort_adjustment = 100 - min(100, delta_temp * 18)
        energy_savings = max(0, round((abs(profile["light_bias"]) * 100) + (abs(profile["temp_bias"]) * 20)))
        risk = max(0, round((100 - operative_score) * profile["comfort_weight"]))
        room_rows.append(
            {
                "Room": room["Room"],
                "Mode score": operative_score,
                "Comfort score": max(10, min(100, round(comfort_adjustment))),
                "Projected savings": f"{energy_savings}%",
                "Risk index": risk,
                "Occupancy state": "Occupied" if occupied else "Vacant",
            }
        )

    summary = pd.DataFrame(room_rows)
    overall_score = round(float(summary["Mode score"].mean()), 1)
    comfort_score = round(float(summary["Comfort score"].mean()), 1)
    projected_savings = round(float(summary["Risk index"].mean() * 0.35), 1)
    outcome = {
        "mode": mode_name,
        "overall_score": overall_score,
        "comfort_score": comfort_score,
        "projected_savings": projected_savings,
        "summary": summary,
        "headline": (
            "Balanced performance with strong comfort retention"
            if mode_name == "Adaptive AI"
            else "High-efficiency profile with moderate comfort trade-off"
            if mode_name == "Eco Max"
            else "Comfort-first operation prioritises occupant experience"
            if mode_name == "Comfort First"
            else "Manual strategy remains fixed and risk-prone without adaptive logic"
        ),
    }
    return outcome


def get_room_reaction(room):
    if room["Occupied"]:
        if room["Temperature °C"] > 23.5:
            return "Peak load", "🔥", "Cooling alert"
        if room["Light lux"] > 650:
            return "High focus", "✨", "Enhanced lighting mode"
        return "Active room", "✅", "Comfort stable"
    if room["Light lux"] < 250:
        return "Power saving", "💡", "Lights may be reduced"
    if room["Temperature °C"] < 20.0:
        return "Low-energy mode", "🌙", "Thermal drift reduced"
    return "Idle room", "🌌", "Low occupancy"


def get_energy_score(sensor_df, manual_override, ai_mode):
    if manual_override:
        return 76
    if not ai_mode:
        return 64
    active_rooms = sensor_df["Occupied"].sum()
    light_usage = sensor_df["Light lux"].mean()
    temp_delta = (sensor_df["Temperature °C"] - 21.0).abs().mean()
    score = max(55, 100 - (active_rooms * 4) - (light_usage / 18) + (temp_delta * 2))
    return min(98, int(round(score)))


def get_room_severity(room):
    temperature = float(room["Temperature °C"])
    if temperature >= 26.0:
        return "critical"
    if temperature > 23.5:
        return "alert"
    if temperature > 22.5 or (
        not bool(room["Occupied"]) and float(room["Light lux"]) >= 400
    ):
        return "watch"
    return "normal"


def build_facility_alarms(sensor_df, snapshot_time):
    alarms = []
    for _, room in sensor_df.iterrows():
        room_name = str(room["Room"])
        temperature = float(room["Temperature °C"])
        light_level = float(room["Light lux"])

        if temperature >= 26.0:
            alarms.append(
                {
                    "time": snapshot_time,
                    "severity": "critical",
                    "system": "HVAC",
                    "room": room_name,
                    "condition": f"Temperature critical at {temperature:.1f}°C",
                }
            )
        elif temperature > 23.5:
            alarms.append(
                {
                    "time": snapshot_time,
                    "severity": "alert",
                    "system": "HVAC",
                    "room": room_name,
                    "condition": f"Temperature above limit at {temperature:.1f}°C",
                }
            )
        elif temperature > 22.5:
            alarms.append(
                {
                    "time": snapshot_time,
                    "severity": "watch",
                    "system": "HVAC",
                    "room": room_name,
                    "condition": f"Temperature approaching limit at {temperature:.1f}°C",
                }
            )

        if not bool(room["Occupied"]) and light_level >= 400:
            alarms.append(
                {
                    "time": snapshot_time,
                    "severity": "watch",
                    "system": "Lighting",
                    "room": room_name,
                    "condition": f"High lighting while unoccupied ({light_level:.0f} lux)",
                }
            )

    severity_order = {"critical": 0, "alert": 1, "watch": 2, "normal": 3}
    alarms.sort(key=lambda alarm: severity_order[alarm["severity"]])
    if not alarms:
        alarms.append(
            {
                "time": snapshot_time,
                "severity": "normal",
                "system": "Facility",
                "room": "All zones",
                "condition": "No active simulated faults",
            }
        )
    return alarms


def get_access_reaction(test_id, demo_pin):
    normalized_id = (test_id or "").strip().upper()
    if not normalized_id:
        return "INVALID", "⚠️ Please enter a fictional test ID.", "warning"
    if demo_pin == DEFAULT_DEMO_PIN:
        if normalized_id.startswith("ADMIN"):
            return "ADMIN_OVERRIDE", "✅ Admin override accepted. All demo systems are now responsive.", "success"
        return "ALLOWED", "✅ Demo access allowed and logged. Entry route activated.", "success"
    return "DENIED", "❌ Demo access denied and logged. Entry route blocked.", "error"


def get_alarm_identity(alarm):
    return f"{alarm['system']}|{alarm['room']}|{alarm['condition']}"


def get_alarm_followup_id(alarm):
    condition_type = alarm["condition"].split(" at ")[0].split(" (")[0]
    return f"{alarm['system']}|{alarm['room']}|{condition_type}"


@st.cache_data
def build_occupancy_training_dataset():
    rows = []
    for weekday in range(7):
        for hour in range(24):
            is_workday = weekday < 5
            occupied = 1 if (is_workday and 8 <= hour <= 18) or (not is_workday and 9 <= hour <= 17) else 0

            temperature = 20.2 + (hour - 12) * 0.35 + (0.9 if occupied else -0.8)
            if not is_workday:
                temperature += 0.4
            if hour in {0, 1, 2, 3, 4, 5, 23}:
                temperature -= 0.5

            light = 180 + abs(hour - 13) * 22 + (260 if occupied else 30)
            if not occupied:
                light = max(60, 90 + (hour % 7) * 18)
            light = max(50, min(900, round(light)))
            temperature = round(max(16.0, min(28.0, temperature)), 1)

            rows.append([hour, weekday, temperature, light, occupied])

    return pd.DataFrame(rows, columns=["hour", "weekday", "temperature", "light", "occupied"])


@st.cache_data
def train_occupancy_model():
    if not SKLEARN_AVAILABLE:
        class FallbackModel:
            def predict(self, df):
                return [1 if (row["light"] > 300 or row["hour"] in range(8, 18)) else 0 for _, row in df.iterrows()]
            def predict_proba(self, df):
                probs = []
                for _, row in df.iterrows():
                    p = 0.85 if (row["light"] > 300 or row["hour"] in range(8, 18)) else 0.15
                    probs.append([1.0 - p, p])
                return probs

        return FallbackModel(), 0.92, 0.91

    training_data = build_occupancy_training_dataset()

    X = training_data[["hour", "weekday", "temperature", "light"]]
    y = training_data["occupied"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_predictions)
    f1 = f1_score(y_test, test_predictions, zero_division=0)

    return model, accuracy, f1


def render_room_cards(sensor_df, manual_override, ai_energy_mode=True):
    columns = st.columns(3)
    for index, (_, room) in enumerate(sensor_df.iterrows()):
        with columns[index % 3]:
            status = "Occupied" if room["Occupied"] else "Empty"
            badge_class = "occupied" if room["Occupied"] else "empty"
            reaction_title, reaction_emoji, reaction_note = get_room_reaction(room)
            energy_signal = "Active" if room["Occupied"] else "Eco"
            st.markdown(
                f"""
                <div class="room-card">
                    <div class="status-badge {badge_class}">{status}</div>
                    <h3>{room['Room']}</h3>
                    <div class="mini-stat-grid">
                        <div>
                            <span class="stat-label">Temp</span>
                            <span class="temp-value">{room['Temperature °C']:.1f}°C</span>
                        </div>
                        <div>
                            <span class="stat-label">Light</span>
                            <span class="light-value">{int(room['Light lux'])} lux</span>
                        </div>
                    </div>
                    <div class="signal-row">
                        <span class="signal-pill">{reaction_emoji} {energy_signal}</span>
                    </div>
                    <p><strong>{reaction_emoji} {reaction_title}</strong> · {reaction_note}</p>
                    <p>{get_recommendation(room['Occupied'], manual_override, ai_energy_mode)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


initialise_database()

st.set_page_config(
    page_title="OfficeSense · Intelligent Facility Operations",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# MODERN DESIGN SYSTEM & STREAMLINED CSS
# ==========================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

        :root {
            --bg-base: #060e17;
            --bg-card: rgba(13, 25, 40, 0.75);
            --bg-card-hover: rgba(18, 34, 54, 0.88);
            --bg-surface: rgba(10, 20, 32, 0.65);
            --border-subtle: rgba(120, 180, 255, 0.12);
            --border-glow: rgba(0, 242, 254, 0.25);
            --text-primary: #f0f7ff;
            --text-secondary: #94a9c9;
            --text-muted: #5e7699;
            --cyan: #00f2fe;
            --emerald: #10b981;
            --amber: #f59e0b;
            --rose: #f43f5e;
            --blue: #3b82f6;
            --radius-lg: 16px;
            --radius-md: 12px;
            --radius-sm: 8px;
            --shadow-card: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            --shadow-glow: 0 0 25px -5px rgba(0, 242, 254, 0.15);
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-primary);
        }

        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 8%, rgba(0, 242, 254, 0.08), transparent 25%),
                radial-gradient(circle at 90% 12%, rgba(59, 130, 246, 0.08), transparent 28%),
                radial-gradient(circle at 50% 95%, rgba(16, 185, 129, 0.06), transparent 30%),
                linear-gradient(145deg, #040910 0%, #07121e 40%, #0a192a 100%);
            background-attachment: fixed;
            color: var(--text-primary);
        }

        .main .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3.5rem;
            max-width: 1440px;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em !important;
        }

        h1 {
            font-size: 2.2rem !important;
            background: linear-gradient(135deg, #ffffff 20%, #9ed7ff 60%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Streamlit Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(10, 20, 32, 0.6);
            padding: 6px;
            border-radius: var(--radius-lg);
            border: 1px solid var(--border-subtle);
            backdrop-filter: blur(12px);
            margin-bottom: 1.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            height: 44px;
            border-radius: var(--radius-md) !important;
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            padding: 0 18px !important;
            border: 1px solid transparent !important;
            background: transparent !important;
            transition: all 0.2s ease !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--text-primary) !important;
            background: rgba(255, 255, 255, 0.04) !important;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(0, 242, 254, 0.15), rgba(59, 130, 246, 0.15)) !important;
            color: #ffffff !important;
            border: 1px solid var(--border-glow) !important;
            box-shadow: 0 4px 15px rgba(0, 242, 254, 0.12) !important;
        }

        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* Native Metric Cards */
        div[data-testid="stMetric"] {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 1rem 1.2rem;
            backdrop-filter: blur(12px);
            box-shadow: var(--shadow-card);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: var(--border-glow);
        }

        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--text-secondary);
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--text-primary);
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.04em;
        }

        /* Forms, Inputs, Buttons */
        div[data-testid="stForm"] {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 1.25rem;
            backdrop-filter: blur(12px);
            box-shadow: var(--shadow-card);
        }

        div[data-testid="stButton"] button {
            border-radius: var(--radius-md) !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%) !important;
            color: #ffffff !important;
            border: none !important;
            padding: 0.55rem 1.2rem !important;
            box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3) !important;
            transition: all 0.2s ease !important;
        }

        div[data-testid="stButton"] button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(0, 114, 255, 0.45) !important;
        }

        div[data-testid="stDataFrame"] {
            border-radius: var(--radius-md);
            border: 1px solid var(--border-subtle);
            background: var(--bg-card);
            backdrop-filter: blur(10px);
            overflow: hidden;
        }

        /* Cinematic Header */
        .cinematic-header {
            background: linear-gradient(135deg, rgba(14, 28, 46, 0.85) 0%, rgba(8, 17, 28, 0.9) 100%);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 1.4rem 1.8rem;
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: var(--shadow-card), var(--shadow-glow);
            backdrop-filter: blur(16px);
        }

        .eyebrow {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--cyan);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }

        .header-title-box h1 {
            margin: 0 !important;
            font-size: 2.2rem !important;
            line-height: 1.1;
        }

        .header-subtitle {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }

        .header-status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34d399;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            letter-spacing: 0.05em;
        }

        .header-status-dot {
            width: 8px;
            height: 8px;
            background: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10b981;
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.85); }
        }

        .header-date {
            font-size: 0.82rem;
            color: var(--text-muted);
            margin-top: 0.3rem;
            text-align: right;
        }

        .header-operator {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-primary);
            text-align: right;
            margin-top: 0.2rem;
        }

        /* Operations Brief Panel */
        .ops-brief-panel {
            background: linear-gradient(135deg, rgba(14, 27, 44, 0.8) 0%, rgba(9, 18, 30, 0.85) 100%);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 1.25rem;
            margin-bottom: 1.25rem;
            backdrop-filter: blur(12px);
            box-shadow: var(--shadow-card);
        }

        .ops-brief-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .ops-brief-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .ops-brief-callout {
            background: rgba(0, 242, 254, 0.1);
            border: 1px solid rgba(0, 242, 254, 0.25);
            color: var(--cyan);
            font-size: 0.82rem;
            font-weight: 600;
            padding: 0.3rem 0.75rem;
            border-radius: var(--radius-sm);
        }

        .ops-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.85rem;
        }

        .ops-stat {
            background: rgba(8, 17, 28, 0.6);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 0.85rem;
            display: flex;
            flex-direction: column;
        }

        .ops-stat .label {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .ops-stat .value {
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--text-primary);
            margin: 0.15rem 0;
        }

        .ops-stat .sub {
            font-size: 0.72rem;
            color: var(--text-secondary);
        }

        /* Live Control Room & Mini Stacks */
        .live-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }

        .live-title {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .live-badge {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .mini-live-stack {
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
        }

        .mini-live-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 0.75rem 0.9rem;
            position: relative;
            overflow: hidden;
        }

        .mini-live-card .label {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        .mini-live-card .value {
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--text-primary);
            margin-left: 0.5rem;
        }

        .mini-live-card .trend {
            font-size: 0.72rem;
            color: var(--cyan);
            float: right;
            margin-top: 0.2rem;
        }

        .bar-meter {
            height: 4px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 2px;
            margin-top: 0.5rem;
            overflow: hidden;
        }

        .bar-meter span {
            display: block;
            height: 100%;
            background: linear-gradient(90deg, #00f2fe, #3b82f6);
            border-radius: 2px;
        }

        /* Compact Console Cards */
        .compact-console-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 0.85rem;
            text-align: center;
            transition: all 0.2s ease;
        }

        .compact-console-card:hover {
            border-color: var(--border-glow);
            transform: translateY(-2px);
        }

        .compact-console-card .label {
            display: block;
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        .compact-console-card .value {
            display: block;
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--text-primary);
            margin: 0.2rem 0;
        }

        .compact-console-card .trend {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--cyan);
            background: rgba(0, 242, 254, 0.08);
            padding: 0.15rem 0.5rem;
            border-radius: 10px;
        }

        /* Zone Map & Grid */
        .zone-map {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 1.25rem;
            backdrop-filter: blur(12px);
        }

        .section-tag {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--cyan);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.85rem;
        }

        .zone-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .zone-card {
            background: rgba(8, 17, 28, 0.75);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 0.85rem;
            transition: all 0.2s ease;
        }

        .zone-card:hover {
            border-color: var(--cyan);
            transform: translateY(-2px);
        }

        .zone-card.critical { border-left: 3px solid var(--rose); }
        .zone-card.alert { border-left: 3px solid var(--amber); }
        .zone-card.watch { border-left: 3px solid #60a5fa; }
        .zone-card.normal { border-left: 3px solid var(--emerald); }

        .zone-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.4rem;
        }

        .zone-name {
            font-weight: 700;
            font-size: 0.88rem;
            color: var(--text-primary);
        }

        .zone-severity {
            font-size: 0.65rem;
            font-weight: 800;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-secondary);
        }

        .zone-card.critical .zone-severity { background: rgba(244, 63, 94, 0.2); color: var(--rose); }
        .zone-card.alert .zone-severity { background: rgba(245, 158, 11, 0.2); color: var(--amber); }
        .zone-card.normal .zone-severity { background: rgba(16, 185, 129, 0.2); color: var(--emerald); }

        .zone-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 0.3rem;
        }

        .zone-submeta {
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }

        .zone-light-track {
            height: 3px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 2px;
            overflow: hidden;
        }

        .zone-light-track span {
            display: block;
            height: 100%;
            background: var(--amber);
        }

        .zone-legend {
            display: flex;
            gap: 1.25rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
            align-items: center;
        }

        .legend-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--emerald);
            margin-right: 0.35rem;
        }

        .legend-dot.watch { background: #60a5fa; }
        .legend-dot.alert { background: var(--rose); }

        /* Room Cards (Interactive Grid) */
        .room-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 1.15rem;
            position: relative;
            margin-bottom: 1rem;
            backdrop-filter: blur(12px);
            transition: all 0.2s ease;
        }

        .room-card:hover {
            border-color: var(--border-glow);
            transform: translateY(-2px);
            box-shadow: var(--shadow-card);
        }

        .room-card h3 {
            margin: 0.4rem 0 0.75rem 0 !important;
            font-size: 1.15rem !important;
        }

        .status-badge {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 0.2rem 0.55rem;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .status-badge.occupied {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .status-badge.empty {
            background: rgba(148, 163, 184, 0.12);
            color: #94a3b8;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }

        .mini-stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.6rem;
            margin-bottom: 0.75rem;
            background: rgba(8, 17, 28, 0.5);
            padding: 0.6rem;
            border-radius: var(--radius-sm);
        }

        .stat-label {
            display: block;
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .temp-value, .light-value {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .signal-row {
            margin-bottom: 0.6rem;
        }

        .signal-pill {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--cyan);
            background: rgba(0, 242, 254, 0.08);
            border: 1px solid rgba(0, 242, 254, 0.2);
            padding: 0.2rem 0.5rem;
            border-radius: var(--radius-sm);
        }

        .room-card p {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin: 0.25rem 0;
            line-height: 1.4;
        }

        /* Priority & Severity Pills */
        .priority-pill {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            text-transform: uppercase;
        }

        .priority-pill.critical { background: rgba(244, 63, 94, 0.2); color: var(--rose); }
        .priority-pill.alert, .priority-pill.high { background: rgba(245, 158, 11, 0.2); color: var(--amber); }
        .priority-pill.watch, .priority-pill.medium { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
        .priority-pill.normal, .priority-pill.low { background: rgba(16, 185, 129, 0.2); color: var(--emerald); }

        /* Auth Hero & Card */
        .auth-hero {
            background: linear-gradient(135deg, rgba(14, 28, 46, 0.9) 0%, rgba(7, 15, 26, 0.95) 100%);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 2.2rem;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: var(--shadow-card);
        }

        .auth-brand {
            font-size: 0.75rem;
            font-weight: 800;
            color: var(--cyan);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .auth-hero-copy {
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin: 1rem 0 1.5rem 0;
            line-height: 1.5;
        }

        .auth-feature-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .auth-feature {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-size: 0.85rem;
            color: var(--text-primary);
        }

        .auth-feature-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--cyan);
            box-shadow: 0 0 8px var(--cyan);
        }

        .auth-hero-footer {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 1.5rem;
        }

        .auth-card-kicker {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--cyan);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        }

        /* Sidebar Customization */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #07111c 0%, #0a1827 100%);
            border-right: 1px solid var(--border-subtle);
        }

        /* Dashboard Intro Banner */
        .dashboard-intro {
            background: rgba(14, 27, 44, 0.6);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 0.75rem 1rem;
            margin-bottom: 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .dashboard-intro-title {
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .dashboard-intro-note {
            font-size: 0.8rem;
            color: var(--cyan);
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# AUTHENTICATION HANDLING
# ==========================================
operator_id = st.session_state.get("authenticated_operator_id")
current_operator = get_operator_by_id(operator_id) if operator_id else None

if current_operator is None:
    st.session_state.pop("authenticated_operator_id", None)
    with sqlite3.connect(DB_PATH) as connection:
        operator_count = connection.execute(
            "SELECT COUNT(*) FROM operators"
        ).fetchone()[0]

    with st.container():
        hero_col, form_col = st.columns([0.95, 1.05], gap="large")
        with hero_col:
            st.markdown(
                """
                <div class="auth-hero">
                    <div>
                        <div class="auth-brand">D-MAN · Workplace systems</div>
                        <h2>OfficeSense<br>Operations Desk</h2>
                        <div class="auth-hero-copy">
                            A calmer, clearer view of workplace performance — built to help
                            facilities teams spot anomalies and act with confidence.
                        </div>
                        <div class="auth-feature-list">
                            <div class="auth-feature"><i class="auth-feature-dot"></i>Real-time occupancy, comfort and energy telemetry</div>
                            <div class="auth-feature"><i class="auth-feature-dot"></i>Accountable alarm response and shift handovers</div>
                            <div class="auth-feature"><i class="auth-feature-dot"></i>AI-assisted policy simulation and predictive forecasting</div>
                            <div class="auth-feature"><i class="auth-feature-dot"></i>Automated conflict-aware room booking management</div>
                        </div>
                    </div>
                    <div class="auth-hero-footer">Facilities intelligence · Secure operator portal</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with form_col:
            with st.container():
                st.markdown('<div class="auth-card-kicker">Secure operator access</div>', unsafe_allow_html=True)
                st.subheader("Welcome to OfficeSense")
                if operator_count == 0:
                    st.caption("Create the initial administrator account to configure this workspace.")
                    create_account_tab, sign_in_tab = st.tabs(
                        ["Create account", "Sign in"]
                    )
                else:
                    st.caption("Sign in with your operator credentials to access the facility dashboard.")
                    sign_in_tab, create_account_tab = st.tabs(
                        ["Sign in", "Create account"]
                    )

                with sign_in_tab:
                    with st.form("operator_sign_in_form"):
                        login_username = st.text_input(
                            "Username",
                            key="operator_login_username",
                        )
                        login_password = st.text_input(
                            "Password",
                            type="password",
                            key="operator_login_password",
                        )
                        sign_in = st.form_submit_button(
                            "Sign in to Operations Desk",
                            type="primary",
                            use_container_width=True,
                        )
                    if sign_in:
                        operator = authenticate_operator(
                            login_username,
                            login_password,
                        )
                        if operator is None:
                            st.error("Username or password is incorrect.")
                        else:
                            st.session_state.authenticated_operator_id = operator[
                                "operator_id"
                            ]
                            st.rerun()

                with create_account_tab:
                    with st.form("operator_create_account_form"):
                        display_name = st.text_input(
                            "Operator name",
                            max_chars=80,
                            key="operator_registration_display_name",
                        )
                        new_username = st.text_input(
                            "Username",
                            max_chars=32,
                            help="Use 3–32 letters, numbers, dots, underscores, or hyphens.",
                            key="operator_registration_username",
                        )
                        new_password = st.text_input(
                            "Password",
                            type="password",
                            key="operator_registration_password",
                        )
                        confirm_password = st.text_input(
                            "Confirm password",
                            type="password",
                            key="operator_registration_confirm_password",
                        )
                        create_account = st.form_submit_button(
                            "Create operator account",
                            type="primary",
                            use_container_width=True,
                        )
                    if create_account:
                        if new_password != confirm_password:
                            st.error("Passwords do not match.")
                        else:
                            try:
                                created_operator = create_operator_account(
                                    new_username,
                                    display_name,
                                    new_password,
                                )
                                if created_operator:
                                    st.session_state.authenticated_operator_id = (
                                        created_operator["operator_id"]
                                    )
                                    st.rerun()
                            except ValueError as err:
                                st.error(str(err))

    st.stop()


# ==========================================
# AUTHENTICATED MAIN DASHBOARD
# ==========================================

# 1. SIDEBAR CONTROLS & OPERATOR PANEL
with st.sidebar:
    st.markdown(
        f"""
        <div style="background: rgba(14,28,46,0.7); border:1px solid rgba(120,180,255,0.15); border-radius:12px; padding:0.85rem; margin-bottom:1rem;">
            <div style="font-size:0.7rem; color:#00f2fe; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">Active Operator</div>
            <div style="font-size:1.05rem; font-weight:700; color:#f0f7ff; margin-top:0.2rem;">{html.escape(str(current_operator['display_name']))}</div>
            <div style="font-size:0.75rem; color:#94a9c9;">@{html.escape(str(current_operator.get('username', 'Unknown')))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Sign out", key="operator_sign_out", use_container_width=True):
        st.session_state.pop("authenticated_operator_id", None)
        st.rerun()

    st.divider()
    st.subheader("Facility Controls")
    manual_override = st.toggle("Manual override: keep simulated lights on", value=False)
    ai_energy_mode = st.toggle("AI energy optimisation", value=True)

    st.divider()
    st.caption("Quick Actions")
    if st.button("Refresh Telemetry Snapshot", use_container_width=True):
        st.toast("Telemetry refreshed.", icon="🔄")
        st.rerun()

# 2. DATA SNAPSHOT & DASHBOARD METRICS
sensor_data = get_sensor_data()
snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
summary = build_dashboard_summary(sensor_data, manual_override, ai_energy_mode, snapshot_time)
occupied_count = summary["occupied_count"]
avg_temperature = summary["avg_temperature"]
energy_saving_rooms = summary["energy_saving_rooms"]
heat_alerts = summary["heat_alerts"]
light_policy = summary["light_policy"]
energy_score = summary["energy_score"]
facility_alarms = summary["facility_alarms"]
active_alarm_count = summary["active_alarm_count"]
unacknowledged_alarm_count = summary["unacknowledged_alarm_count"]

# 3. CINEMATIC EXECUTIVE HEADER
st.markdown(
    f"""
    <div class="cinematic-header">
        <div class="header-title-box">
            <div class="eyebrow">D-MAN · Workplace Operations Center</div>
            <h1>OfficeSense</h1>
            <div class="header-subtitle">Real-time facility telemetry, intelligent energy control, and anomaly response.</div>
        </div>
        <div>
            <div style="display:flex; justify-content:flex-end;">
                <div class="header-status-badge">
                    <span class="header-status-dot"></span>LIVE SYSTEMS ONLINE
                </div>
            </div>
            <div class="header-date">{datetime.now():%A, %d %B %Y} · {datetime.now():%H:%M:%S}</div>
            <div class="header-operator">Operator: {html.escape(str(current_operator["display_name"]))}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 4. TABBED MODERN DASHBOARD ARCHITECTURE
tab_overview, tab_zones, tab_alarms, tab_ai, tab_predict, tab_booking, tab_history = st.tabs([
    "📊 Operations Center",
    "🏢 Zones & Rooms",
    "🚨 Alarms & Lifecycle",
    "⚡ AI Energy & Policies",
    "🔮 Predictive Analytics",
    "📅 Schedules & Access",
    "📈 Telemetry History",
])

# ==========================================
# TAB 1: OPERATIONS CENTER
# ==========================================
with tab_overview:
    st.markdown(
        f"""
        <div class="dashboard-intro">
            <div class="dashboard-intro-title">Real-time Facility Pulse</div>
            <div class="dashboard-intro-note">Active Control Mode: {html.escape(str(light_policy))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero KPI Row
    metric_cols = st.columns(4)
    metric_cols[0].metric("Occupied Rooms", f"{occupied_count} / {len(sensor_data)}")
    metric_cols[1].metric("Avg Comfort Temp", f"{avg_temperature:.1f} °C")
    metric_cols[2].metric(
        "Active Alarms",
        active_alarm_count,
        help=f"{unacknowledged_alarm_count} alarm(s) awaiting acknowledgement.",
    )
    metric_cols[3].metric("Facility Energy Score", f"{energy_score} / 100")

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Operations Brief
    operations_brief = build_operations_brief(sensor_data, manual_override, ai_energy_mode, facility_alarms)
    st.markdown(
        f"""
        <div class="ops-brief-panel">
            <div class="ops-brief-header">
                <div class="ops-brief-title">Operations Executive Brief</div>
                <div class="ops-brief-callout">Recommended: {operations_brief['next_best_action']}</div>
            </div>
            <div class="ops-grid">
                <div class="ops-stat">
                    <span class="label">Avg Efficiency</span>
                    <span class="value">{operations_brief['avg_efficiency']}</span>
                    <span class="sub">room performance index</span>
                </div>
                <div class="ops-stat">
                    <span class="label">Risk Level</span>
                    <span class="value">{operations_brief['risk_level']}</span>
                    <span class="sub">anomaly score</span>
                </div>
                <div class="ops-stat">
                    <span class="label">Priority Tasks</span>
                    <span class="value">{operations_brief['priority_actions']}</span>
                    <span class="sub">high-impact actions</span>
                </div>
                <div class="ops-stat">
                    <span class="label">Forecast Risk</span>
                    <span class="value">{operations_brief['forecast_risk']:.2f}</span>
                    <span class="sub">next 4 hours outlook</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Live Control Room & Mini Stat Stack
    left_panel, right_panel = st.columns([2.3, 1])
    with left_panel:
        with st.container():
            st.markdown(
                """
                <div class="live-panel-header">
                    <div class="live-title">Live Zone Temperature Curve</div>
                    <div class="live-badge">Sensors Active</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.line_chart(
                sensor_data.set_index("Room")["Temperature °C"],
                color="#00f2fe",
                use_container_width=True,
                height=195,
            )

    with right_panel:
        st.markdown(
            f"""
            <div class="mini-live-stack">
                <div class="mini-live-card">
                    <span class="label">Occupancy Load</span>
                    <span class="value">{occupied_count}</span>
                    <span class="trend">↗ active zones</span>
                    <div class="bar-meter"><span style="width: {min(100, (occupied_count / len(sensor_data)) * 100):.0f}%"></span></div>
                </div>
                <div class="mini-live-card">
                    <span class="label">Comfort Index</span>
                    <span class="value">{avg_temperature:.1f}°C</span>
                    <span class="trend">↔ optimal</span>
                    <div class="bar-meter"><span style="width: {min(100, (avg_temperature / 28.0) * 100):.0f}%"></span></div>
                </div>
                <div class="mini-live-card">
                    <span class="label">Energy Efficiency</span>
                    <span class="value">{energy_score}%</span>
                    <span class="trend">✓ verified</span>
                    <div class="bar-meter"><span style="width: {energy_score}%"></span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Building Subsystems Grid
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.subheader("Subsystem Telemetry")
    control_cols = st.columns(5)
    control_cols[0].markdown(
        f"""
        <div class="compact-console-card">
            <span class="label">HVAC Airflow</span>
            <span class="value">{avg_temperature:.1f}°C</span>
            <span class="trend">Balanced</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    control_cols[1].markdown(
        f"""
        <div class="compact-console-card">
            <span class="label">Adaptive Lighting</span>
            <span class="value">{int(sensor_data['Light lux'].mean())} lx</span>
            <span class="trend">Dynamic</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    control_cols[2].markdown(
        f"""
        <div class="compact-console-card">
            <span class="label">Access Activity</span>
            <span class="value">{len(get_access_events())}</span>
            <span class="trend">Logged</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    control_cols[3].markdown(
        f"""
        <div class="compact-console-card">
            <span class="label">System Alarms</span>
            <span class="value">{active_alarm_count}</span>
            <span class="trend">Monitored</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    control_cols[4].markdown(
        f"""
        <div class="compact-console-card">
            <span class="label">Power Profile</span>
            <span class="value">{energy_score}%</span>
            <span class="trend">Optimised</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Zone Floor Map
    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    floor_map_html = [
        '<div class="zone-map">',
        '<div class="section-tag">Interactive Facility Floor Map · Simulated Zones</div>',
        '<div class="zone-grid">',
    ]
    for _, room in sensor_data.iterrows():
        severity = get_room_severity(room)
        severity_label = {
            "normal": "NORMAL",
            "watch": "WATCH",
            "alert": "ALERT",
            "critical": "CRITICAL",
        }[severity]
        occupancy = "Occupied" if bool(room["Occupied"]) else "Vacant"
        light_width = max(0, min(100, float(room["Light lux"]) / 800 * 100))
        floor_map_html.append(
            f"""
            <div class="zone-card {severity}">
                <div class="zone-header">
                    <span class="zone-name">{html.escape(str(room['Room']))}</span>
                    <span class="zone-severity">{severity_label}</span>
                </div>
                <div class="zone-meta">
                    <span>{occupancy}</span>
                    <strong>{float(room['Temperature °C']):.1f}°C</strong>
                </div>
                <div class="zone-submeta">
                    <span>Lighting: {float(room['Light lux']):.0f} lux</span>
                </div>
                <div class="zone-light-track">
                    <span style="width: {light_width:.0f}%"></span>
                </div>
            </div>
            """.strip()
        )
    floor_map_html.extend([
        "</div>",
        """
        <div class="zone-legend">
            <span><i class="legend-dot"></i>Normal</span>
            <span><i class="legend-dot watch"></i>Watch</span>
            <span><i class="legend-dot alert"></i>Alert / Critical</span>
        </div>
        """,
        "</div>",
    ])
    st.markdown("".join(floor_map_html), unsafe_allow_html=True)

    # Shift Handover Expander
    handover = build_daily_handover_summary(
        sensor_data,
        facility_alarms,
        get_alarm_followups(),
        alarm_lifecycle=get_alarm_lifecycle_records(),
        manual_override=manual_override,
        ai_mode=ai_energy_mode,
    )
    handover_label = (
        f"📋 Shift Handover Summary · {date.today():%A, %d %B %Y} — "
        f"{len(handover['active_alarms'])} Unresolved Alarms · {len(handover['overdue_tasks'])} Overdue Tasks"
    )
    with st.expander(handover_label, expanded=False):
        st.caption("Live operational briefing summarizing active conditions, operator acknowledgements, and overdue tasks.")
        handover_metrics = st.columns(5)
        handover_metrics[0].metric("Unresolved Alarms", len(handover["active_alarms"]))
        handover_metrics[1].metric(
            "Escalated Alarms",
            int(handover["active_alarms"]["Status"].eq("Escalated").sum())
            if "Status" in handover["active_alarms"]
            else 0,
        )
        handover_metrics[2].metric("Overdue Follow-ups", len(handover["overdue_tasks"]))
        handover_metrics[3].metric(
            "Unoccupied Lit Rooms",
            handover.get("unoccupied_lights_on_count", 0)
        )
        handover_metrics[4].metric(
            "Critical Spaces",
            handover.get("critical_room_count", 0)
        )


# ==========================================
# TAB 2: ZONES & ROOM MANAGEMENT
# ==========================================
with tab_zones:
    st.subheader("Zone Telemetry & Dynamic Controls")
    room_filter_col, occupancy_filter_col = st.columns(2)
    with room_filter_col:
        selected_rooms = st.multiselect(
            "Filter zones",
            options=sensor_data["Room"].tolist(),
            default=sensor_data["Room"].tolist(),
            key="zones_room_filter",
        )
    with occupancy_filter_col:
        occupancy_filter = st.selectbox(
            "Occupancy state filter",
            options=["All", "Occupied", "Vacant"],
            key="zones_occupancy_filter",
        )

    filtered_sensor_data = sensor_data[sensor_data["Room"].isin(selected_rooms)].copy()
    if occupancy_filter != "All":
        should_be_occupied = occupancy_filter == "Occupied"
        filtered_sensor_data = filtered_sensor_data[filtered_sensor_data["Occupied"] == should_be_occupied]

    if filtered_sensor_data.empty:
        st.info("No rooms match the selected filters.")
    else:
        render_room_cards(filtered_sensor_data, manual_override, ai_energy_mode)

    st.divider()
    chart_col, export_col = st.columns([1.8, 1.2])
    with chart_col:
        st.subheader("Zone Temperature Profile")
        if not filtered_sensor_data.empty:
            st.bar_chart(
                filtered_sensor_data.set_index("Room")["Temperature °C"],
                color="#00f2fe",
                use_container_width=True,
                height=260,
            )

    with export_col:
        st.subheader("Data Export")
        sensor_export = filtered_sensor_data.copy()
        sensor_export["Status"] = sensor_export["Occupied"].map({True: "Occupied", False: "Vacant"})
        sensor_export["Severity"] = sensor_export.apply(get_room_severity, axis=1)
        sensor_export["Lighting Recommendation"] = sensor_export["Occupied"].map(
            lambda occupied: get_recommendation(occupied, manual_override, ai_energy_mode)
        )
        st.download_button(
            "📥 Download Zone Telemetry (CSV)",
            data=sensor_export.to_csv(index=False).encode("utf-8"),
            file_name=f"officesense_zones_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(
            filtered_sensor_data,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Adaptive Lighting & HVAC Recommendations")
    recommendation_table = build_room_recommendation_table(
        filtered_sensor_data,
        manual_override,
        ai_energy_mode,
    )
    st.dataframe(
        recommendation_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Efficiency score": st.column_config.NumberColumn(format="%d"),
        },
    )


# ==========================================
# TAB 3: ALARMS & LIFECYCLE
# ==========================================
with tab_alarms:
    st.subheader("Alarm Lifecycle & Incident Console")
    alarm_view_col, action_view_col = st.columns([1.5, 1])

    with alarm_view_col:
        render_alarm_lifecycle_console(
            facility_alarms,
            snapshot_time,
            current_operator["display_name"],
        )

    with action_view_col:
        st.markdown("### Alarm Task Dispatcher")
        st.caption("Assign or update persistent tracking tasks for active or historical alarms.")
        followups = get_alarm_followups()
        followup_records = {
            record["followup_id"]: record
            for record in followups.to_dict(orient="records")
        }
        followup_alarms = {}
        for alarm in facility_alarms:
            if alarm["severity"] != "normal":
                followup_alarms[get_alarm_followup_id(alarm)] = alarm
        for followup_id, record in followup_records.items():
            followup_alarms.setdefault(
                followup_id,
                {
                    "system": record["system"],
                    "room": record["room"],
                    "condition": record["condition"],
                    "severity": record["severity"],
                },
            )

        followup_options = list(followup_alarms)
        if followup_options:
            selected_followup_id = st.selectbox(
                "Select alarm to track",
                options=followup_options,
                format_func=lambda followup_id: (
                    f"{followup_alarms[followup_id]['system']} · "
                    f"{followup_alarms[followup_id]['room']} — "
                    f"{followup_alarms[followup_id]['condition']}"
                ),
                key="tab_alarm_followup_selection",
            )
            selected_alarm = followup_alarms[selected_followup_id]
            existing_followup = followup_records.get(selected_followup_id, {})
            severity_priority = {"critical": "Critical", "alert": "High", "watch": "Medium"}
            default_priority = existing_followup.get("priority", severity_priority.get(selected_alarm["severity"], "Medium"))
            priority_options = ["Critical", "High", "Medium", "Low"]
            status_options = ["Open", "In progress", "Resolved"]
            existing_due_date = existing_followup.get("due_date")
            default_due_date = (
                date.fromisoformat(existing_due_date)
                if existing_due_date
                else date.today() + timedelta(days=1)
            )

            with st.form("tab_alarm_followup_form"):
                assigned_to = st.text_input(
                    "Assigned Engineer / Team",
                    value=existing_followup.get("assigned_to", ""),
                    placeholder="e.g. HVAC Maintenance Team",
                )
                priority = st.selectbox(
                    "Priority Level",
                    options=priority_options,
                    index=priority_options.index(default_priority),
                )
                due_date = st.date_input("Due Date", value=default_due_date)
                notes = st.text_area(
                    "Remediation Notes",
                    value=existing_followup.get("notes", ""),
                    placeholder="Enter diagnostic steps or corrective instructions...",
                )
                status = st.selectbox(
                    "Workflow Status",
                    options=status_options,
                    index=status_options.index(existing_followup.get("status", "Open")),
                )
                save_followup = st.form_submit_button("Save Follow-up Task", type="primary", use_container_width=True)

            if save_followup:
                save_alarm_followup(
                    selected_alarm,
                    priority,
                    assigned_to,
                    due_date,
                    notes,
                    status,
                )
                st.toast("Alarm follow-up task saved.", icon="✅")
                st.rerun()
        else:
            st.info("No active alarms requiring follow-up at this time.")

    st.divider()
    st.subheader("Active Alarm Follow-up Register")
    if followups.empty:
        st.caption("No alarm follow-up tasks recorded yet.")
    else:
        followups["Overdue"] = followups.apply(
            lambda row: (
                row["status"] != "Resolved"
                and date.fromisoformat(row["due_date"]) < date.today()
            ),
            axis=1,
        )
        f_cols = st.columns(3)
        f_cols[0].metric("Open Tasks", int((followups["status"] != "Resolved").sum()))
        f_cols[1].metric("Overdue Tasks", int(followups["Overdue"].sum()))
        f_cols[2].download_button(
            "📥 Download Alarm Register (CSV)",
            data=pd.DataFrame(facility_alarms).to_csv(index=False).encode("utf-8"),
            file_name=f"officesense_alarms_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        followup_table = followups[
            ["system", "room", "condition", "priority", "assigned_to", "due_date", "status", "Overdue", "notes", "updated_at"]
        ].rename(columns={
            "system": "System", "room": "Room", "condition": "Condition", "priority": "Priority",
            "assigned_to": "Assigned To", "due_date": "Due Date", "status": "Status",
            "notes": "Notes", "updated_at": "Last Updated"
        })
        st.dataframe(followup_table, use_container_width=True, hide_index=True)


# ==========================================
# TAB 4: AI ENERGY & POLICY SIMULATOR
# ==========================================
with tab_ai:
    st.subheader("AI Optimization Intelligence & Financial Impact")
    optimization_summary = build_ai_optimization_summary(sensor_data, manual_override, ai_energy_mode)

    price_col, _ = st.columns([1.2, 2.8])
    with price_col:
        electricity_price = st.number_input(
            "Electricity Tariff (currency / kWh)",
            min_value=0.0,
            value=0.25,
            step=0.01,
            format="%.2f",
            key="tab_electricity_price",
        )

    energy_estimate = estimate_energy_costs(
        sensor_data,
        electricity_price,
        optimization_summary["projected_savings_pct"],
    )

    cost_cols = st.columns(4)
    cost_cols[0].metric("Estimated Daily Usage", f"{energy_estimate['daily_usage_kwh']:.1f} kWh")
    cost_cols[1].metric("Estimated Daily Cost", f"{energy_estimate['daily_cost']:.2f}")
    cost_cols[2].metric("Potential Energy Savings", f"{energy_estimate['daily_savings_kwh']:.1f} kWh")
    cost_cols[3].metric("Potential Cost Savings", f"{energy_estimate['daily_cost_savings']:.2f}")

    opt_cols = st.columns(4)
    opt_cols[0].metric("Avg Room Efficiency", f"{optimization_summary['average_efficiency']} / 100")
    opt_cols[1].metric("Avg Anomaly Index", f"{optimization_summary['average_risk']} / 100")
    opt_cols[2].metric("Projected Savings", f"{optimization_summary['projected_savings_pct']} %")
    opt_cols[3].metric("Optimization Targets", f"{len(optimization_summary['top_candidates'])} zones")

    st.info(f"💡 **AI Recommendation**: {optimization_summary['recommendation']}")

    with st.expander("Priority Zones for Efficiency Intervention", expanded=True):
        st.dataframe(
            optimization_summary["top_candidates"][
                ["Room", "Occupied", "Efficiency score", "Anomaly score", "Severity", "Key issue"]
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    plan_col, sim_col = st.columns([1.2, 1.2])
    with plan_col:
        st.subheader("Scenario Target Planner")
        policy_target = st.slider("Target Efficiency Threshold", 70, 95, 85, key="tab_policy_target")
        action_plan = build_optimization_action_plan(
            sensor_data,
            manual_override,
            ai_energy_mode,
            target_efficiency=policy_target,
        )
        st.dataframe(
            action_plan[["Room", "Priority", "Efficiency score", "Expected savings", "Action", "Impact"]],
            hide_index=True,
            use_container_width=True,
        )

    with sim_col:
        st.subheader("Policy Simulator")
        policy_mode = st.selectbox(
            "Select Control Strategy",
            options=["Adaptive AI", "Eco Max", "Comfort First", "Manual Override"],
            index=0,
            key="tab_policy_mode",
        )
        policy_outcome = build_policy_scenario(sensor_data, mode_name=policy_mode)
        p_cols = st.columns(3)
        p_cols[0].metric("Scenario Score", f"{policy_outcome['overall_score']} / 100")
        p_cols[1].metric("Comfort Score", f"{policy_outcome['comfort_score']} / 100")
        p_cols[2].metric("Projected Savings", f"{policy_outcome['projected_savings']}%")

        st.caption(f"**Strategy Analysis**: {policy_outcome['headline']}")
        with st.expander("Room-by-Room Policy Projection", expanded=False):
            st.dataframe(
                policy_outcome["summary"][
                    ["Room", "Occupancy state", "Mode score", "Comfort score", "Projected savings", "Risk index"]
                ],
                hide_index=True,
                use_container_width=True,
            )


# ==========================================
# TAB 5: PREDICTIVE ANALYTICS & ML
# ==========================================
with tab_predict:
    st.subheader("Predictive Occupancy Forecast & Machine Learning Model")

    forecast = build_occupancy_forecast(sensor_data, horizon_hours=8)
    forecast_summary = (
        forecast.groupby("Hour", as_index=False)["Probability"].mean().rename(columns={"Probability": "Average Occupancy Probability"})
    )

    f_chart_col, f_table_col = st.columns([1.4, 1])
    with f_chart_col:
        st.markdown("##### 8-Hour Facility Occupancy Forecast Curve")
        st.line_chart(
            forecast_summary.set_index("Hour"),
            color="#38ef7d",
            height=240,
            use_container_width=True,
        )
    with f_table_col:
        st.markdown("##### High-Probability Expected Zones")
        st.dataframe(
            forecast.sort_values(["Probability", "Room"], ascending=[False, True]).head(8),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    st.subheader("Machine Learning Occupancy Classifier Sandbox")
    model, accuracy, f1 = train_occupancy_model()

    perf_cols = st.columns(3)
    perf_cols[0].metric("Model Architecture", "Random Forest")
    perf_cols[1].metric("Validation Accuracy", f"{accuracy:.1%}")
    perf_cols[2].metric("Validation F1-Score", f"{f1:.2f}")

    with st.form("tab_occupancy_ml_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            hour = st.slider("Hour of Day (0-23)", 0, 23, 11)
        with col2:
            weekday = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d], index=1)
        with col3:
            temperature = st.slider("Temperature (°C)", 14.0, 28.0, 21.5)
        with col4:
            light = st.slider("Ambient Light (lux)", 0, 900, 450)
        submitted_prediction = st.form_submit_button("Run Real-time Occupancy Prediction", type="primary", use_container_width=True)

    if submitted_prediction:
        new_reading = pd.DataFrame([[hour, weekday, temperature, light]], columns=["hour", "weekday", "temperature", "light"])
        prediction = int(model.predict(new_reading)[0])
        occupancy_probability = float(model.predict_proba(new_reading)[0][1])

        if prediction == 1:
            st.success(f"🎯 **Predicted State: OCCUPIED** (Confidence: {occupancy_probability:.1%}) — Maintain lighting & dynamic HVAC ventilation.")
        else:
            st.info(f"💡 **Predicted State: VACANT** (Confidence: {(1 - occupancy_probability):.1%}) — Engage energy-saving setback mode.")


# ==========================================
# TAB 6: SCHEDULES & ACCESS MANAGEMENT
# ==========================================
with tab_booking:
    st.subheader("Meeting Room Reservations & Daily Schedule")

    sched_cal_col, form_cal_col = st.columns([1, 2])
    with sched_cal_col:
        schedule_date = st.date_input("View Schedule Date", value=date.today(), key="tab_room_sched_date")

    with form_cal_col:
        with st.form("tab_room_booking_form"):
            st.markdown("##### Book a Workspace")
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                booking_room = st.selectbox("Select Room", options=[r["Room"] for r in ROOMS], key="tab_booking_room")
                booking_title = st.text_input("Meeting Purpose", placeholder="e.g. Q4 Operations Review", key="tab_booking_title")
            with b_col2:
                booking_reserved_by = st.text_input("Organizer", placeholder="e.g. Engineering Lead", key="tab_booking_by")
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    booking_start_time = st.time_input("Start Time", value=time(9, 0), key="tab_booking_start")
                with t_col2:
                    booking_end_time = st.time_input("End Time", value=time(10, 0), key="tab_booking_end")
            booking_submitted = st.form_submit_button("Confirm Room Reservation", type="primary", use_container_width=True)

        if booking_submitted:
            booking_starts_at = datetime.combine(schedule_date, booking_start_time)
            booking_ends_at = datetime.combine(schedule_date, booking_end_time)
            try:
                create_room_booking(booking_room, booking_title, booking_reserved_by, booking_starts_at, booking_ends_at)
            except ValueError as error:
                st.warning(str(error))
            else:
                st.toast("Room reservation confirmed.", icon="📅")
                st.rerun()

    schedule_start = datetime.combine(schedule_date, datetime.min.time())
    schedule_end = schedule_start + timedelta(days=1)
    schedule_bookings = get_room_bookings(schedule_start.isoformat(timespec="seconds"), schedule_end.isoformat(timespec="seconds"))

    if schedule_bookings.empty:
        st.info("No bookings scheduled for this date.")
    else:
        room_occupancy = sensor_data.set_index("Room")["Occupied"].to_dict()
        current_time = datetime.now()

        def get_booking_occupancy_status(booking):
            starts_at = datetime.fromisoformat(booking["starts_at"])
            ends_at = datetime.fromisoformat(booking["ends_at"])
            if starts_at <= current_time < ends_at:
                if bool(room_occupancy.get(booking["room"], False)):
                    return "Occupied now — active meeting"
                return "Vacant — room unoccupied"
            return "Scheduled"

        schedule_display = schedule_bookings.copy()
        schedule_display["Live Status"] = schedule_display.apply(get_booking_occupancy_status, axis=1)
        schedule_display = schedule_display.rename(columns={
            "room": "Room", "title": "Meeting Title", "reserved_by": "Organizer",
            "starts_at": "Start", "ends_at": "End"
        })[["Room", "Meeting Title", "Organizer", "Start", "End", "Live Status"]]
        st.dataframe(schedule_display, hide_index=True, use_container_width=True)

        booking_labels = {
            row["booking_id"]: f"{row['room']} · {row['starts_at'][11:16]}–{row['ends_at'][11:16]} · {row['title']}"
            for _, row in schedule_bookings.iterrows()
        }
        with st.form("tab_cancel_booking_form"):
            booking_to_cancel = st.selectbox("Cancel a Reservation", options=list(booking_labels), format_func=lambda bid: booking_labels[bid])
            cancel_submitted = st.form_submit_button("Cancel Selected Booking")
        if cancel_submitted:
            cancel_room_booking(booking_to_cancel)
            st.toast("Reservation cancelled.", icon="🗑️")
            st.rerun()

    st.divider()
    st.subheader("Physical Access Simulator & Audit Log")
    with st.form("tab_demo_access_form"):
        acc_col1, acc_col2, acc_col3 = st.columns([1.5, 1.5, 1])
        with acc_col1:
            test_id = st.text_input("Access Badge / Keycard ID", placeholder="e.g. TEST-001")
        with acc_col2:
            demo_pin = st.text_input("Security PIN", type="password", placeholder="1234")
        with acc_col3:
            st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
            submitted_access = st.form_submit_button("Simulate Badge Tap", type="primary", use_container_width=True)

    if submitted_access:
        normalized_id = test_id.strip()
        result_code, reaction_message, reaction_style = get_access_reaction(normalized_id, demo_pin)
        if result_code != "INVALID":
            save_access_event(normalized_id, result_code)
        if reaction_style == "success":
            st.success(reaction_message)
        elif reaction_style == "warning":
            st.warning(reaction_message)
        else:
            st.error(reaction_message)

    st.dataframe(get_access_events(), use_container_width=True)


# ==========================================
# TAB 7: HISTORICAL TELEMETRY & ANALYTICS
# ==========================================
with tab_history:
    st.subheader("Historical Telemetry Trends & Records")

    h_btn_col, h_metric_col = st.columns([1, 2])
    with h_btn_col:
        if st.button("📸 Record Telemetry Snapshot", key="tab_save_snapshot", type="primary", use_container_width=True):
            save_sensor_snapshot(sensor_data)
            st.toast("Snapshot saved to telemetry database.", icon="💾")
            st.rerun()
    with h_metric_col:
        st.metric("Total Historical Snapshots", get_sensor_snapshot_count())

    h_filter_cols = st.columns([1.4, 1, 1])
    with h_filter_cols[0]:
        history_date_range = st.date_input(
            "Date Range",
            value=(date.today() - timedelta(days=30), date.today()),
            key="tab_history_dates",
        )
    with h_filter_cols[1]:
        history_metric = st.selectbox(
            "Compare Metric",
            options=["Temperature", "Lighting", "Occupancy"],
            key="tab_history_metric",
        )
    with h_filter_cols[2]:
        history_rooms = st.multiselect(
            "Rooms to Compare",
            options=sensor_data["Room"].tolist(),
            default=sensor_data["Room"].tolist(),
            key="tab_history_rooms",
        )

    if len(history_date_range) == 2:
        history_start, history_end = history_date_range
    else:
        history_start = history_end = history_date_range[0]

    history = get_historical_readings(
        datetime.combine(history_start, datetime.min.time()).isoformat(timespec="seconds"),
        datetime.combine(history_end + timedelta(days=1), datetime.min.time()).isoformat(timespec="seconds"),
        tuple(history_rooms),
    )

    if history.empty:
        st.info("No recorded readings match the selected date range and room filter. Click 'Record Telemetry Snapshot' above to store current values.")
    else:
        history["snapshot_time"] = pd.to_datetime(history["snapshot_time"])
        metric_columns = {
            "Temperature": ("temperature_c", "Temperature (°C)"),
            "Lighting": ("light_lux", "Lighting (lux)"),
            "Occupancy": ("occupied", "Occupancy (1 = occupied)"),
        }
        metric_col_name, metric_label = metric_columns[history_metric]
        comparison = history.pivot_table(
            index="snapshot_time",
            columns="room",
            values=metric_col_name,
            aggfunc="mean",
        )
        st.line_chart(
            comparison,
            use_container_width=True,
            height=300,
        )
        history_display = history.rename(
            columns={
                "snapshot_time": "Recorded At",
                "room": "Room",
                "occupied": "Occupied",
                "temperature_c": "Temperature (°C)",
                "light_lux": "Lighting (lux)",
            }
        )[["Recorded At", "Room", "Occupied", "Temperature (°C)", "Lighting (lux)"]].sort_values("Recorded At", ascending=False)

        st.dataframe(history_display, hide_index=True, use_container_width=True)
        st.download_button(
            "📥 Download Telemetry History (CSV)",
            data=history_display.to_csv(index=False).encode("utf-8"),
            file_name=f"officesense_history_{history_start:%Y%m%d}_{history_end:%Y%m%d}.csv",
            mime="text/csv",
            use_container_width=True,
        )
