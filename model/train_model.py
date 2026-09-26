"""
train_model.py
Builds a labeled sample dataset of IT/ERP support tickets and trains a
TF-IDF + Logistic Regression classifier to predict ticket CATEGORY.
Priority is scored separately with a lightweight keyword/rule model
(fast, explainable, no training data needed).

Run:
    python model/train_model.py

Produces:
    model/ticket_classifier.pkl   -> {"vectorizer":..., "clf":..., "labels":...}
"""

import argparse
import hashlib
import os
import pickle
import random
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

random.seed(42)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 1. Synthetic but realistic training data (5 categories x many phrasings)
# ---------------------------------------------------------------------------
TEMPLATES = {
    "Technical Issue": [
        "The ERP system crashes every time I try to open the {module} module.",
        "I'm getting a 500 error when generating the {module} report.",
        "The application is extremely slow when loading {module} data.",
        "Unable to sync {module} data between the warehouse and head office.",
        "The system froze while I was posting a {module} transaction.",
        "Login page throws a blank screen after entering credentials.",
        "The {module} dashboard is not loading any widgets today.",
        "Print function for {module} invoices is not working.",
        "Getting 'connection timed out' when accessing {module} from the branch office.",
        "The mobile app keeps logging me out while using {module}.",
    ],
    "Billing": [
        "I was charged twice for this month's {module} subscription.",
        "Can you explain the extra line item on our latest invoice?",
        "Our invoice for {module} licenses shows the wrong quantity.",
        "We need a refund for the duplicate {module} module purchase.",
        "The renewal price for {module} seems higher than last year's quote.",
        "Please send a corrected invoice; the VAT amount looks wrong.",
        "When is the next billing cycle for our {module} add-on?",
        "We'd like to switch from monthly to annual billing.",
        "Payment failed but the amount was still deducted from our account.",
        "Requesting an itemized breakdown of charges for {module} usage.",
    ],
    "Account Access": [
        "I forgot my password and the reset link isn't arriving.",
        "New employee needs access to the {module} module, please provision an account.",
        "My account got locked after too many login attempts.",
        "Please revoke {module} access for an employee who left the company.",
        "Two-factor authentication code is not being sent to my phone.",
        "I need admin rights on the {module} module for my new role.",
        "Can you merge my two user accounts into one login?",
        "Our SSO integration is not letting anyone log in this morning.",
        "Please reset the manager's permissions on {module}.",
        "I can log in but I don't see the {module} menu anymore.",
    ],
    "Feature Request": [
        "It would help if the {module} module supported bulk CSV import.",
        "Can we get a dark mode option for the dashboard?",
        "Please add an approval workflow before {module} entries are finalized.",
        "We'd like an export-to-Excel button on the {module} report screen.",
        "Requesting a custom field to track project codes in {module}.",
        "Could you add email alerts when {module} stock falls below threshold?",
        "It would be great to have a mobile widget for {module} approvals.",
        "Can the {module} module support multi-currency entries?",
        "Please consider adding role-based dashboards for {module}.",
        "We'd like scheduled automatic backups for {module} data.",
    ],
    "General Inquiry": [
        "What are your support hours for {module} related issues?",
        "Can you share the user guide for the {module} module?",
        "Is there a training session available for new {module} users?",
        "Just checking, is the system down for maintenance tonight?",
        "How do I contact my account manager about {module}?",
        "Where can I find release notes for the latest {module} update?",
        "Do you offer onboarding sessions for the {module} module?",
        "What integrations are available for {module}?",
        "Is there a sandbox environment to test {module} changes?",
        "Could you point me to documentation on {module} best practices?",
    ],
}

MODULES = ["inventory", "payroll", "procurement", "finance", "HR",
           "sales", "warehouse", "CRM", "asset management", "reporting"]

URGENT_WORDS = ["urgent", "asap", "immediately", "down", "crash", "crashed",
                "crashing", "crashes", "frozen", "froze", "not working", "locked",
                "unable", "blank screen", "everyone", "entire company", "production",
                "all users"]
MEDIUM_WORDS = ["slow", "delay", "error", "issue", "problem", "soon"]


def make_priority(text: str) -> str:
    t = text.lower()
    if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", t) for w in URGENT_WORDS):
        return "High"
    if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", t) for w in MEDIUM_WORDS):
        return "Medium"
    return "Low"


synthetic_rows = []
for category, templates in TEMPLATES.items():
    for template_index, template in enumerate(templates):
        for _ in range(4):  # a few module variations per template
            module = random.choice(MODULES)
            text = template.format(module=module)
            synthetic_rows.append({
                "text": text,
                "category": category,
                "priority": make_priority(text),
                # Keep variants of the same sentence together during evaluation.
                "template_id": f"{category}:{template_index}",
            })


def load_reviewed_tickets(csv_path):
    if not csv_path:
        return []
    feedback = pd.read_csv(csv_path).fillna("")
    required_columns = {"subject", "description", "reviewed_category"}
    missing = required_columns - set(feedback.columns)
    if missing:
        raise ValueError(
            "Feedback CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    rows = []
    seen_labels = {}
    for _, record in feedback.iterrows():
        subject = str(record["subject"]).strip()
        description = str(record["description"]).strip()
        category = str(record["reviewed_category"]).strip()
        text = f"{subject}. {description}".strip(" .")
        if not text or not category:
            continue
        if category not in TEMPLATES:
            continue
        normalized_text = re.sub(r"\s+", " ", text.lower()).strip()
        group_id = "reviewed:" + hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        previous_label = seen_labels.get(group_id)
        if previous_label and previous_label != category:
            raise ValueError(
                "The feedback CSV has identical ticket text with conflicting reviewed categories. "
                "Resolve those labels before training."
            )
        seen_labels[group_id] = category
        rows.append({
            "text": text,
            "category": category,
            "priority": make_priority(text),
            "template_id": group_id,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Train the support-ticket classifier on synthetic examples and optional reviewed tickets."
    )
    parser.add_argument(
        "--feedback-csv",
        help="Optional dashboard CSV export with human-reviewed categories.",
    )
    args = parser.parse_args()

    feedback_rows = load_reviewed_tickets(args.feedback_csv)
    if args.feedback_csv:
        print(f"Loaded {len(feedback_rows)} human-reviewed tickets from {args.feedback_csv}")

    synthetic_df = pd.DataFrame(synthetic_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    feedback_texts = {
        re.sub(r"\s+", " ", row["text"].lower()).strip() for row in feedback_rows
    }
    if feedback_texts:
        normalized_synthetic = synthetic_df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        excluded = normalized_synthetic.isin(feedback_texts)
        if excluded.any():
            print(f"Replacing {int(excluded.sum())} synthetic duplicate(s) with human-reviewed labels")
        synthetic_training_df = synthetic_df.loc[~excluded]
    else:
        synthetic_training_df = synthetic_df
    training_df = pd.concat(
        [synthetic_training_df, pd.DataFrame(feedback_rows)], ignore_index=True
    ).sample(frac=1, random_state=42).reset_index(drop=True)
    data_dir = os.path.join(BASE_DIR, "data")
    model_dir = os.path.join(BASE_DIR, "model")
    os.makedirs(data_dir, exist_ok=True)
    synthetic_df.drop(columns=["template_id"]).to_csv(
        os.path.join(data_dir, "sample_tickets.csv"), index=False
    )
    print(f"Generated {len(synthetic_df)} synthetic sample tickets -> data/sample_tickets.csv")

# ---------------------------------------------------------------------------
# 2. Train TF-IDF + Logistic Regression for CATEGORY
# ---------------------------------------------------------------------------
# Split by original sentence template, so module variations of one sentence
# cannot appear in both train and test sets.
    template_groups = training_df[["template_id", "category"]].drop_duplicates(
        subset="template_id"
    )
    train_template_ids, test_template_ids = train_test_split(
        template_groups["template_id"],
        test_size=0.2,
        random_state=42,
        stratify=template_groups["category"],
    )
    train_df = training_df[training_df["template_id"].isin(train_template_ids)]
    test_df = training_df[training_df["template_id"].isin(test_template_ids)]
    X_train, y_train = train_df["text"], train_df["category"]
    X_test, y_test = test_df["text"], test_df["category"]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    clf.fit(X_train_vec, y_train)

    print("\nClassification report on held-out test set:")
    print(classification_report(y_test, clf.predict(X_test_vec), zero_division=0))

    labels = sorted(training_df["category"].unique().tolist())

    with open(os.path.join(model_dir, "ticket_classifier.pkl"), "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "clf": clf, "labels": labels}, f)

    print("Saved trained model -> model/ticket_classifier.pkl")


if __name__ == "__main__":
    main()
