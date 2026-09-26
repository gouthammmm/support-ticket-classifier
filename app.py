"""
app.py — Support Ticket Classifier & Auto-Responder

A Flask web app that:
  1. Accepts a support ticket (subject + description) via a web form.
  2. Classifies it into a category using a trained TF-IDF + Logistic
     Regression model, and assigns a priority via keyword rules.
  3. Generates a draft auto-response.
  4. Stores every ticket in a local SQLite database.
  5. Shows a dashboard with ticket stats and history.

Run:
    pip install -r requirements.txt
    python model/train_model.py      # first time only, builds the model
    python app.py
    -> open http://127.0.0.1:5000
"""

import csv
import hashlib
import hmac
import io
import os
import pickle
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from functools import wraps
from urllib.parse import urlparse

from flask import (
    Flask, abort, flash, jsonify, make_response, redirect, render_template,
    request, session, url_for,
)

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from responder import generate_reply
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "tickets.db")))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ticket_classifier.pkl")

app = Flask(__name__)
is_production = os.environ.get("ENVIRONMENT", "development").lower() == "production"
configured_secret = os.environ.get("ERP_TICKET_SECRET_KEY", "")
if is_production and (len(configured_secret) < 32 or configured_secret.startswith("replace-")):
    raise RuntimeError("Set ERP_TICKET_SECRET_KEY to a unique random value of at least 32 characters.")
app.secret_key = configured_secret or secrets.token_hex(32)
app.config.update(
    MAX_CONTENT_LENGTH=1 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("ERP_COOKIE_SECURE", "0").lower() in {"1", "true", "yes"},
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    SESSION_REFRESH_EACH_REQUEST=False,
)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri=os.environ.get("RATELIMIT_STORAGE_URL", "memory://"),
    headers_enabled=True,
)
trusted_proxy_count = int(os.environ.get("ERP_TRUSTED_PROXY_COUNT", "0"))
if trusted_proxy_count > 0:
    app.wsgi_app = ProxyFix(
        app.wsgi_app,
        x_for=trusted_proxy_count,
        x_proto=trusted_proxy_count,
        x_host=0,
        x_port=0,
        x_prefix=0,
    )

CATEGORIES = [
    "Technical Issue", "Billing", "Account Access", "Feature Request",
    "General Inquiry",
]
STATUSES = ["Open", "In Progress", "Resolved"]
REVIEW_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model file not found. Run `python model/train_model.py` first."
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


URGENT_WORDS = ["urgent", "asap", "immediately", "down", "crash", "crashed",
                "crashing", "crashes", "frozen", "froze", "not working", "locked",
                "unable", "blank screen", "everyone", "entire company", "production",
                "all users"]
MEDIUM_WORDS = ["slow", "delay", "error", "issue", "problem", "soon"]


def contains_term(text: str, terms) -> bool:
    return any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text)
        for term in terms
    )


def score_priority(text: str) -> str:
    t = text.lower()
    if contains_term(t, URGENT_WORDS):
        return "High"
    if contains_term(t, MEDIUM_WORDS):
        return "Medium"
    return "Low"


def classify(text: str):
    model = get_model()
    vec = model["vectorizer"].transform([text])
    category = model["clf"].predict(vec)[0]
    proba = model["clf"].predict_proba(vec)[0]
    confidence = float(max(proba))
    priority = score_priority(text)
    return category, priority, confidence


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH, timeout=15) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=15000")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                subject TEXT,
                description TEXT,
                category TEXT,
                priority TEXT,
                confidence REAL,
                response TEXT,
                created_at TEXT,
                status TEXT NOT NULL DEFAULT 'Open',
                assigned_to TEXT NOT NULL DEFAULT '',
                reviewed_category TEXT,
                reviewed_at TEXT,
                updated_at TEXT
            )
        """)
        existing_columns = {
            row[1] for row in con.execute("PRAGMA table_info(tickets)").fetchall()
        }
        migrations = {
            "status": "TEXT NOT NULL DEFAULT 'Open'",
            "assigned_to": "TEXT NOT NULL DEFAULT ''",
            "reviewed_category": "TEXT",
            "reviewed_at": "TEXT",
            "updated_at": "TEXT",
            "internal_notes": "TEXT NOT NULL DEFAULT ''",
            "resolution": "TEXT NOT NULL DEFAULT ''",
            "resolved_at": "TEXT",
        }
        for column, definition in migrations.items():
            if column not in existing_columns:
                con.execute(f"ALTER TABLE tickets ADD COLUMN {column} {definition}")
        con.execute("UPDATE tickets SET updated_at = created_at WHERE updated_at IS NULL")
        con.execute(
            "UPDATE tickets SET resolved_at = updated_at "
            "WHERE status = 'Resolved' AND resolved_at IS NULL"
        )
        con.execute("""
            CREATE TABLE IF NOT EXISTS staff_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('owner', 'agent')),
                created_at TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS staff_invites (
                token_hash TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('agent')),
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT '',
                keywords TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_by INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_tickets_status_id ON tickets(status, id DESC)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_active_category ON knowledge_articles(is_active, category)")
        con.commit()


def save_ticket(name, email, subject, description, category, priority, confidence, response):
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            """INSERT INTO tickets
               (name, email, subject, description, category, priority, confidence,
                response, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, email, subject, description, category, priority, confidence,
             response, created_at, created_at),
        )
        con.commit()
        return cur.lastrowid


def ticket_filter_values(args):
    search = args.get("q", "").strip()[:100]
    status = args.get("status", "")
    category = args.get("category", "")
    if status not in STATUSES:
        status = ""
    if category not in CATEGORIES:
        category = ""
    return {"q": search, "status": status, "category": category}


def ticket_where(filters):
    clauses = []
    params = []
    if filters["q"]:
        pattern = f"%{filters['q']}%"
        clauses.append(
            "(name LIKE ? OR email LIKE ? OR subject LIKE ? OR description LIKE ?)"
        )
        params.extend([pattern] * 4)
    if filters["status"]:
        clauses.append("status = ?")
        params.append(filters["status"])
    if filters["category"]:
        clauses.append("COALESCE(reviewed_category, category) = ?")
        params.append(filters["category"])
    return (" WHERE " + " AND ".join(clauses) if clauses else "", params)


def fetch_tickets(filters=None, limit=200, offset=0):
    filters = filters or {"q": "", "status": "", "category": ""}
    where, params = ticket_where(filters)
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        query = f"SELECT * FROM tickets{where} ORDER BY id DESC"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        rows = con.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def fetch_ticket(ticket_id):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        return dict(row) if row else None


def article_tokens(value):
    return set(re.findall(r"[a-z0-9]{3,}", (value or "").lower()))


def suggest_knowledge_articles(ticket, limit=3):
    category = ticket.get("reviewed_category") or ticket.get("category") or ""
    query_tokens = article_tokens(
        f"{ticket.get('subject', '')} {ticket.get('description', '')}"
    )
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        articles = con.execute(
            "SELECT id, title, category, keywords, content, updated_at "
            "FROM knowledge_articles WHERE is_active = 1"
        ).fetchall()

    ranked = []
    for row in articles:
        article = dict(row)
        title_tokens = article_tokens(article["title"])
        keyword_tokens = article_tokens(article["keywords"])
        category_match = bool(article["category"] and article["category"] == category)
        keyword_hits = query_tokens & keyword_tokens
        title_hits = query_tokens & title_tokens
        score = (4 if category_match else 0) + (2 * len(keyword_hits)) + len(title_hits)
        if score:
            article["match_score"] = score
            article["match_reason"] = (
                "Category match" if category_match else "Related keywords"
            )
            ranked.append(article)
    return sorted(ranked, key=lambda item: (-item["match_score"], item["title"].lower()))[:limit]


def count_tickets(filters=None):
    filters = filters or {"q": "", "status": "", "category": ""}
    where, params = ticket_where(filters)
    with sqlite3.connect(DB_PATH) as con:
        return con.execute(f"SELECT COUNT(*) FROM tickets{where}", params).fetchone()[0]


def spreadsheet_safe(value):
    if isinstance(value, str) and re.match(r"^\s*[=+\-@]", value):
        return "'" + value
    return value


def fetch_stats():
    with sqlite3.connect(DB_PATH) as con:
        cat_counts = dict(con.execute(
            "SELECT COALESCE(reviewed_category, category), COUNT(*) "
            "FROM tickets GROUP BY COALESCE(reviewed_category, category)"
        ).fetchall())
        pri_counts = dict(con.execute(
            "SELECT priority, COUNT(*) FROM tickets GROUP BY priority"
        ).fetchall())
        status_counts = dict(con.execute(
            "SELECT status, COUNT(*) FROM tickets GROUP BY status"
        ).fetchall())
        total = con.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        needs_review = con.execute(
            "SELECT COUNT(*) FROM tickets "
            "WHERE reviewed_at IS NULL AND confidence < ?",
            (REVIEW_SCORE_THRESHOLD,),
        ).fetchone()[0]
    return {
        "by_category": cat_counts,
        "by_priority": pri_counts,
        "by_status": status_counts,
        "total": total,
        "needs_review": needs_review,
    }


def csrf_token():
    token = session.get("_csrf_token")
    if token is None:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


@app.context_processor
def inject_csrf_token():
    return {"csrf_token": csrf_token, "staff": current_staff()}


@app.before_request
def protect_post_requests():
    if request.method == "POST":
        expected = session.get("_csrf_token", "")
        supplied = request.form.get("csrf_token", "")
        if not expected or not hmac.compare_digest(expected, supplied):
            abort(400, description="The form expired. Reload the page and try again.")


def current_staff():
    staff_id = session.get("staff_user_id")
    if not staff_id:
        return None
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT id, email, role FROM staff_users WHERE id = ?", (staff_id,)
        ).fetchone()
        return dict(row) if row else None


def staff_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_staff():
            return redirect(url_for("login", next=request.full_path.rstrip("?")))
        return view(*args, **kwargs)
    return wrapped


def owner_required(view):
    @wraps(view)
    @staff_required
    def wrapped(*args, **kwargs):
        staff = current_staff()
        if not staff or staff["role"] != "owner":
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def staff_count():
    with sqlite3.connect(DB_PATH) as con:
        return con.execute("SELECT COUNT(*) FROM staff_users").fetchone()[0]


def valid_email(email):
    return bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email))


def safe_next_path(candidate):
    candidate = candidate or ""
    parsed = urlparse(candidate or "")
    if candidate.startswith("/") and not candidate.startswith("//") and not parsed.netloc:
        return candidate
    return url_for("dashboard")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/setup", methods=["GET", "POST"])
@limiter.limit("10 per hour")
def initial_setup():
    if staff_count():
        return redirect(url_for("login"))
    setup_secret = os.environ.get("ERP_SETUP_TOKEN", "")
    local_first_setup = (
        not is_production and request.remote_addr in {"127.0.0.1", "::1"}
    )
    if not setup_secret and not local_first_setup:
        return render_template("setup.html", setup_disabled=True), 503
    if request.method == "POST":
        supplied_secret = request.form.get("setup_token", "")
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirmation = request.form.get("confirm_password", "")
        if setup_secret and not hmac.compare_digest(supplied_secret, setup_secret):
            flash("The setup token is not valid.", "error")
        elif not valid_email(email):
            flash("Enter a valid email address.", "error")
        elif len(password) < 12:
            flash("Use a password with at least 12 characters.", "error")
        elif password != confirmation:
            flash("The passwords do not match.", "error")
        else:
            try:
                with sqlite3.connect(DB_PATH, timeout=15) as con:
                    con.execute("BEGIN IMMEDIATE")
                    if con.execute("SELECT COUNT(*) FROM staff_users").fetchone()[0]:
                        return redirect(url_for("login"))
                    cur = con.execute(
                        "INSERT INTO staff_users (email, password_hash, role, created_at) "
                        "VALUES (?, ?, 'owner', ?)",
                        (email, generate_password_hash(password),
                         datetime.now(timezone.utc).replace(microsecond=0).isoformat()),
                    )
                    con.commit()
                session.clear()
                session["staff_user_id"] = cur.lastrowid
                session.permanent = True
                flash("Owner account created. Invite teammates from the Team page.", "success")
                return redirect(url_for("dashboard"))
            except sqlite3.IntegrityError:
                flash("An account already exists for that email.", "error")
    return render_template(
        "setup.html", setup_disabled=False, require_setup_token=bool(setup_secret)
    )


@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per 15 minutes")
def login():
    if not staff_count():
        return redirect(url_for("initial_setup"))
    if current_staff():
        return redirect(url_for("dashboard"))
    next_path = safe_next_path(request.args.get("next") or request.form.get("next"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            user = con.execute(
                "SELECT id, email, password_hash FROM staff_users WHERE email = ?",
                (email,),
            ).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["staff_user_id"] = user["id"]
            session.permanent = True
            return redirect(next_path)
        flash("Email or password is incorrect.", "error")
    return render_template("login.html", next_path=next_path)


@app.route("/logout", methods=["POST"])
@staff_required
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("index"))


@app.route("/join/<token>", methods=["GET", "POST"])
@limiter.limit("10 per hour")
def join_team(token):
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        invite = con.execute(
            "SELECT email, role, expires_at, used_at FROM staff_invites WHERE token_hash = ?",
            (token_hash,),
        ).fetchone()
    if not invite or invite["used_at"] or invite["expires_at"] <= now:
        return render_template("join.html", invite=None, token=token), 410
    if request.method == "POST":
        password = request.form.get("password", "")
        confirmation = request.form.get("confirm_password", "")
        if len(password) < 12:
            flash("Use a password with at least 12 characters.", "error")
        elif password != confirmation:
            flash("The passwords do not match.", "error")
        else:
            try:
                with sqlite3.connect(DB_PATH, timeout=15) as con:
                    con.execute("BEGIN IMMEDIATE")
                    con.row_factory = sqlite3.Row
                    current_invite = con.execute(
                        "SELECT email, role, expires_at, used_at FROM staff_invites WHERE token_hash = ?",
                        (token_hash,),
                    ).fetchone()
                    if (not current_invite or current_invite["used_at"]
                            or current_invite["expires_at"] <= now):
                        return render_template("join.html", invite=None, token=token), 410
                    cur = con.execute(
                        "INSERT INTO staff_users (email, password_hash, role, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (current_invite["email"], generate_password_hash(password),
                         current_invite["role"], now),
                    )
                    con.execute(
                        "UPDATE staff_invites SET used_at = ? WHERE token_hash = ?",
                        (now, token_hash),
                    )
                    con.commit()
                session.clear()
                session["staff_user_id"] = cur.lastrowid
                session.permanent = True
                flash("Your staff account is ready.", "success")
                return redirect(url_for("dashboard"))
            except sqlite3.IntegrityError:
                flash("An account already exists for this email.", "error")
    return render_template("join.html", invite=dict(invite), token=token)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/healthz", methods=["GET"])
def healthz():
    with sqlite3.connect(DB_PATH, timeout=5) as con:
        con.execute("SELECT 1").fetchone()
    return jsonify(status="ok")


@app.route("/classify", methods=["POST"])
@limiter.limit("20 per hour")
def classify_ticket():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    subject = request.form.get("subject", "").strip()
    description = request.form.get("description", "").strip()
    website = request.form.get("website", "").strip()

    if website:
        return render_template("result.html", ticket_id=None, subject=subject)

    if len(name) > 100 or len(email) > 254 or len(subject) > 200 or len(description) > 5000:
        flash("Please keep name, email, subject, and description within the displayed limits.")
        return redirect(url_for("index"))

    if not valid_email(email):
        flash("Enter a valid email so the support team can follow up.", "error")
        return redirect(url_for("index"))

    if not description:
        flash("Please describe the issue before submitting.")
        return redirect(url_for("index"))

    full_text = f"{subject}. {description}"
    category, priority, confidence = classify(full_text)

    ticket_id = save_ticket(name, email, subject, description, category, priority, confidence, "")
    response = generate_reply(name, category, priority, ticket_id)

    with sqlite3.connect(DB_PATH) as con:
        con.execute("UPDATE tickets SET response = ? WHERE id = ?", (response, ticket_id))
        con.commit()

    return render_template(
        "result.html",
        ticket_id=ticket_id, subject=subject,
    )


@app.route("/dashboard")
@staff_required
def dashboard():
    filters = ticket_filter_values(request.args)
    page = request.args.get("page", 1, type=int) or 1
    page_size = 25
    total_filtered = count_tickets(filters)
    page_count = max(1, (total_filtered + page_size - 1) // page_size)
    page = min(max(1, page), page_count)
    tickets = fetch_tickets(filters, limit=page_size, offset=(page - 1) * page_size)
    stats = fetch_stats()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        staff_members = [dict(row) for row in con.execute(
            "SELECT id, email, role FROM staff_users ORDER BY email"
        ).fetchall()]
    return render_template(
        "dashboard.html", tickets=tickets, stats=stats, filters=filters,
        categories=CATEGORIES, statuses=STATUSES, page=page,
        page_count=page_count, total_filtered=total_filtered,
        staff_members=staff_members,
    )


@app.route("/tickets/<int:ticket_id>")
@staff_required
def ticket_workspace(ticket_id):
    ticket = fetch_ticket(ticket_id)
    if not ticket:
        abort(404)
    articles = suggest_knowledge_articles(ticket)
    return render_template(
        "ticket_workspace.html", ticket=ticket, articles=articles,
        categories=CATEGORIES, statuses=STATUSES,
    )


@app.route("/tickets/<int:ticket_id>/work", methods=["POST"])
@staff_required
def save_ticket_work(ticket_id):
    ticket = fetch_ticket(ticket_id)
    if not ticket:
        abort(404)
    status = request.form.get("status", "")
    internal_notes = request.form.get("internal_notes", "").strip()
    resolution = request.form.get("resolution", "").strip()
    if status not in STATUSES:
        flash("Choose a valid ticket status.", "error")
    elif len(internal_notes) > 10000 or len(resolution) > 10000:
        flash("Keep work notes and resolution summaries under 10,000 characters.", "error")
    elif status == "Resolved" and not resolution:
        flash("Add a short resolution summary before resolving this ticket.", "error")
    else:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        resolved_at = now if status == "Resolved" else None
        with sqlite3.connect(DB_PATH) as con:
            con.execute(
                "UPDATE tickets SET internal_notes = ?, resolution = ?, status = ?, "
                "resolved_at = ?, updated_at = ? WHERE id = ?",
                (internal_notes, resolution, status, resolved_at, now, ticket_id),
            )
            con.commit()
        flash(f"Work notes for ticket #{ticket_id} saved.", "success")
    return redirect(url_for("ticket_workspace", ticket_id=ticket_id))


@app.route("/knowledge")
@staff_required
def knowledge_library():
    staff = current_staff()
    include_archived = staff["role"] == "owner" and request.args.get("archived") == "1"
    active = 0 if include_archived else 1
    search = request.args.get("q", "").strip()[:100]
    category = request.args.get("category", "")
    if category not in CATEGORIES:
        category = ""
    clauses = ["a.is_active = ?"]
    params = [active]
    if search:
        pattern = f"%{search}%"
        clauses.append("(a.title LIKE ? OR a.keywords LIKE ? OR a.content LIKE ?)")
        params.extend([pattern, pattern, pattern])
    if category:
        clauses.append("(a.category = ? OR a.category = '')")
        params.append(category)
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        articles = [dict(row) for row in con.execute(
            "SELECT a.*, u.email AS author_email FROM knowledge_articles a "
            "LEFT JOIN staff_users u ON u.id = a.created_by WHERE "
            + " AND ".join(clauses) + " ORDER BY a.updated_at DESC, a.title COLLATE NOCASE",
            params,
        ).fetchall()]
    return render_template(
        "knowledge.html", articles=articles, categories=CATEGORIES,
        search=search, category=category, include_archived=include_archived,
    )


@app.route("/knowledge/new", methods=["POST"])
@owner_required
def create_knowledge_article():
    title = request.form.get("title", "").strip()
    category = request.form.get("category", "").strip()
    keywords = request.form.get("keywords", "").strip()
    content = request.form.get("content", "").strip()
    if len(title) > 160 or not title:
        flash("Enter a guide title with no more than 160 characters.", "error")
    elif category and category not in CATEGORIES:
        flash("Choose a valid guide category.", "error")
    elif len(keywords) > 500 or len(content) > 10000 or not content:
        flash("Add guide steps and keep keywords under 500 and content under 10,000 characters.", "error")
    else:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        staff = current_staff()
        with sqlite3.connect(DB_PATH) as con:
            con.execute(
                "INSERT INTO knowledge_articles "
                "(title, category, keywords, content, created_by, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (title, category, keywords, content, staff["id"], now, now),
            )
            con.commit()
        flash("Troubleshooting guide added to the team library.", "success")
    return redirect(url_for("knowledge_library"))


@app.route("/knowledge/<int:article_id>")
@staff_required
def view_knowledge_article(article_id):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT a.*, u.email AS author_email FROM knowledge_articles a "
            "LEFT JOIN staff_users u ON u.id = a.created_by WHERE a.id = ? AND a.is_active = 1",
            (article_id,),
        ).fetchone()
    if not row:
        abort(404)
    return render_template("knowledge_article.html", article=dict(row))


@app.route("/knowledge/<int:article_id>/edit", methods=["GET", "POST"])
@owner_required
def edit_knowledge_article(article_id):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM knowledge_articles WHERE id = ?", (article_id,)
        ).fetchone()
    if not row:
        abort(404)
    article = dict(row)
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        category = request.form.get("category", "").strip()
        keywords = request.form.get("keywords", "").strip()
        content = request.form.get("content", "").strip()
        if len(title) > 160 or not title:
            flash("Enter a guide title with no more than 160 characters.", "error")
        elif category and category not in CATEGORIES:
            flash("Choose a valid guide category.", "error")
        elif len(keywords) > 500 or len(content) > 10000 or not content:
            flash("Add guide steps and keep keywords under 500 and content under 10,000 characters.", "error")
        else:
            now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            with sqlite3.connect(DB_PATH) as con:
                con.execute(
                    "UPDATE knowledge_articles SET title = ?, category = ?, keywords = ?, "
                    "content = ?, updated_at = ? WHERE id = ?",
                    (title, category, keywords, content, now, article_id),
                )
                con.commit()
            flash("Troubleshooting guide updated.", "success")
            return redirect(url_for("view_knowledge_article", article_id=article_id))
        article.update(title=title, category=category, keywords=keywords, content=content)
    return render_template(
        "knowledge_edit.html", article=article, categories=CATEGORIES,
    )


@app.route("/knowledge/<int:article_id>/archive", methods=["POST"])
@owner_required
def toggle_knowledge_article(article_id):
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT is_active FROM knowledge_articles WHERE id = ?", (article_id,)
        ).fetchone()
        if not row:
            abort(404)
        new_active = 0 if row[0] else 1
        con.execute(
            "UPDATE knowledge_articles SET is_active = ?, updated_at = ? WHERE id = ?",
            (new_active, datetime.now(timezone.utc).replace(microsecond=0).isoformat(), article_id),
        )
        con.commit()
    flash("Guide restored." if new_active else "Guide archived.", "success")
    return redirect(url_for("knowledge_library", archived="" if new_active else "1"))


@app.route("/team")
@owner_required
def team():
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        members = [dict(row) for row in con.execute(
            "SELECT id, email, role, created_at FROM staff_users ORDER BY created_at"
        ).fetchall()]
        invites = [dict(row) for row in con.execute(
            "SELECT email, role, created_at, expires_at FROM staff_invites "
            "WHERE used_at IS NULL AND expires_at > ? ORDER BY created_at DESC",
            (now,),
        ).fetchall()]
    return render_template("team.html", members=members, invites=invites)


@app.route("/team/invite", methods=["POST"])
@owner_required
@limiter.limit("20 per hour")
def create_invite():
    email = request.form.get("email", "").strip().lower()
    if not valid_email(email):
        flash("Enter a valid email address.", "error")
        return redirect(url_for("team"))
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    with sqlite3.connect(DB_PATH) as con:
        if con.execute("SELECT 1 FROM staff_users WHERE email = ?", (email,)).fetchone():
            flash("That email already belongs to a team member.", "error")
            return redirect(url_for("team"))
        con.execute(
            "INSERT INTO staff_invites (token_hash, email, role, created_at, expires_at) "
            "VALUES (?, ?, 'agent', ?, ?)",
            (token_hash, email, now.isoformat(), (now + timedelta(days=7)).isoformat()),
        )
        con.commit()
    invite_path = url_for("join_team", token=token)
    public_base_url = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    parsed_base_url = urlparse(public_base_url)
    local_http_origin = (
        parsed_base_url.scheme == "http"
        and parsed_base_url.hostname in {"localhost", "127.0.0.1", "::1"}
    )
    if is_production and (
        (parsed_base_url.scheme != "https" and not local_http_origin)
        or not parsed_base_url.netloc
        or parsed_base_url.username or parsed_base_url.password
        or parsed_base_url.path not in {"", "/"}
        or parsed_base_url.query or parsed_base_url.fragment
    ):
        flash("Set PUBLIC_BASE_URL to your public HTTPS origin before creating invite links.", "error")
        return redirect(url_for("team"))
    invite_link = f"{public_base_url}{invite_path}" if public_base_url else invite_path
    flash(f"{email}|{invite_link}", "invite")
    return redirect(url_for("team"))


@app.route("/tickets/<int:ticket_id>/update", methods=["POST"])
@staff_required
def update_ticket(ticket_id):
    status = request.form.get("status", "")
    reviewed_category = request.form.get("reviewed_category", "")
    assigned_to = request.form.get("assigned_to", "").strip()[:100]
    if status not in STATUSES:
        flash("Choose a valid ticket status.")
        return redirect(url_for("dashboard"))
    if reviewed_category and reviewed_category not in CATEGORIES:
        flash("Choose a valid category.")
        return redirect(url_for("dashboard"))
    current_ticket = fetch_ticket(ticket_id)
    if not current_ticket:
        flash("Ticket not found.")
        return redirect(url_for("dashboard"))
    if (status == "Resolved" and current_ticket["status"] != "Resolved"
            and not current_ticket.get("resolution")):
        flash("Open the ticket workspace and record the fix before resolving it.", "error")
        return redirect(url_for("ticket_workspace", ticket_id=ticket_id))
    with sqlite3.connect(DB_PATH) as con:
        staff_emails = {row[0] for row in con.execute("SELECT email FROM staff_users").fetchall()}
    if assigned_to and assigned_to not in staff_emails:
        flash("Choose a current staff member as the assignee.", "error")
        return redirect(url_for("dashboard"))

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    reviewed_at = now if reviewed_category else None
    resolved_at = (
        (current_ticket.get("resolved_at") or now) if status == "Resolved" else None
    )
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            """UPDATE tickets
               SET status = ?, assigned_to = ?, reviewed_category = ?,
                   reviewed_at = ?, resolved_at = ?, updated_at = ?
               WHERE id = ?""",
            (status, assigned_to, reviewed_category or None, reviewed_at,
             resolved_at, now, ticket_id),
        )
        con.commit()
    if cur.rowcount == 0:
        flash("Ticket not found.")
    else:
        flash(f"Ticket #{ticket_id} updated.")
    filters = ticket_filter_values({
        "q": request.form.get("q", ""),
        "category": request.form.get("filter_category", ""),
        "status": request.form.get("filter_status", ""),
    })
    page = request.form.get("page", "1")
    return redirect(url_for("dashboard", **filters, page=page))


@app.route("/tickets/export.csv")
@staff_required
def export_tickets():
    filters = ticket_filter_values(request.args)
    tickets = fetch_tickets(filters, limit=None)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "id", "name", "email", "subject", "description", "predicted_category",
        "reviewed_category", "priority", "model_score", "status", "resolution",
        "resolved_at", "assigned_to", "created_at", "updated_at",
    ])
    for ticket in tickets:
        writer.writerow([spreadsheet_safe(value) for value in [
            ticket["id"], ticket["name"], ticket["email"], ticket["subject"],
            ticket["description"], ticket["category"], ticket["reviewed_category"],
            ticket["priority"], ticket["confidence"], ticket["status"],
            ticket["resolution"], ticket["resolved_at"], ticket["assigned_to"],
            ticket["created_at"], ticket["updated_at"],
        ]])
    response = make_response("\ufeff" + output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=tickets.csv"
    return response


@app.route("/api/tickets")
@limiter.limit("60 per hour")
def api_tickets():
    api_token = os.environ.get("ERP_TICKET_API_TOKEN")
    if not api_token:
        return jsonify(error="Ticket API is disabled; configure ERP_TICKET_API_TOKEN."), 503

    scheme, separator, supplied_token = request.headers.get("Authorization", "").partition(" ")
    if (scheme.lower() != "bearer" or not separator
            or not hmac.compare_digest(supplied_token, api_token)):
        return jsonify(error="A valid bearer token is required."), 401
    return jsonify(fetch_tickets(limit=1000))


init_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
