import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="GitHub Repo Success Predictor", layout="centered")

DATA_PATH = "data/top_github_repos_2026.csv"
ASOF = pd.Timestamp("2026-05-07", tz="UTC")

def normalize_topics(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = s.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    s = s.replace(";", ",").replace("|", ",")
    parts = [p.strip().lower().replace(" ", "-") for p in s.split(",")]
    parts = [p for p in parts if p]
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return ",".join(out)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"]).copy()

    # Define target inside the app (top 20%)
    star_threshold = float(df["stars"].quantile(0.80))
    df["high_star_repo"] = (df["stars"] >= star_threshold).astype(int)

    # Dates -> features
    for c in ["created_at", "updated_at"]:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    df["repo_age_days"] = (ASOF - df["created_at"]).dt.days
    df["days_since_update"] = (ASOF - df["updated_at"]).dt.days

    if "topics" in df.columns:
        df["topics"] = df["topics"].apply(normalize_topics)

    return df, star_threshold

@st.cache_resource
def train_model(df):
    feature_cols = [
        "forks", "open_issues", "watchers",
        "language", "license", "topics",
        "is_fork", "has_wiki", "archived",
        "size_kb", "default_branch",
        "repo_age_days", "days_since_update",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df["high_star_repo"].astype(int).copy()

    # boolean-ish cleanup
    for c in ["is_fork", "has_wiki", "archived"]:
        if c in X.columns:
            X[c] = X[c].astype(str).str.lower().replace({"true": 1, "false": 0})
            X[c] = pd.to_numeric(X[c], errors="coerce")

    numeric_cols = [c for c in feature_cols if c in [
        "forks","open_issues","watchers","size_kb","repo_age_days","days_since_update",
        "is_fork","has_wiki","archived"
    ]]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols),
        ],
        remainder="drop"
    )

    model = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    # quick eval (optional)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

    return pipeline, feature_cols, auc

st.title("GitHub Repo Success Predictor")
st.caption("Trains a model on the included dataset at startup (cached). No joblib pickle loading.")

# Load + train
df, star_threshold = load_data()
pipeline, feature_cols, auc = train_model(df)

st.write(f"Dataset rows: **{len(df)}**")
st.write(f"High-star threshold (top 20%): **{star_threshold:,.0f} stars**")
st.write(f"Validation ROC AUC (quick split): **{auc:.3f}**")

st.subheader("Enter repo metadata")

# Build nice dropdowns from dataset
languages = sorted([x for x in df["language"].dropna().unique().tolist()]) if "language" in df.columns else []
licenses = sorted([x for x in df["license"].dropna().unique().tolist()]) if "license" in df.columns else []
branches = sorted([x for x in df["default_branch"].dropna().unique().tolist()]) if "default_branch" in df.columns else ["main"]

inputs = {}

if "forks" in feature_cols:
    inputs["forks"] = st.number_input("Forks", min_value=0, value=1000, step=100)
if "open_issues" in feature_cols:
    inputs["open_issues"] = st.number_input("Open issues", min_value=0, value=50, step=5)
if "watchers" in feature_cols:
    inputs["watchers"] = st.number_input("Watchers", min_value=0, value=1000, step=100)
if "size_kb" in feature_cols:
    inputs["size_kb"] = st.number_input("Repo size (KB)", min_value=0, value=200000, step=10000)

if "repo_age_days" in feature_cols:
    inputs["repo_age_days"] = st.number_input("Repo age (days)", min_value=0, value=2000, step=30)
if "days_since_update" in feature_cols:
    inputs["days_since_update"] = st.number_input("Days since last update", min_value=0, value=30, step=1)

def yes_no(label, default=False):
    return 1 if st.selectbox(label, ["No", "Yes"], index=(1 if default else 0)) == "Yes" else 0

if "is_fork" in feature_cols:
    inputs["is_fork"] = yes_no("Is fork?", default=False)
if "has_wiki" in feature_cols:
    inputs["has_wiki"] = yes_no("Has wiki?", default=True)
if "archived" in feature_cols:
    inputs["archived"] = yes_no("Archived?", default=False)

if "language" in feature_cols:
    if languages:
        inputs["language"] = st.selectbox("Primary language", languages, index=(languages.index("Python") if "Python" in languages else 0))
    else:
        inputs["language"] = st.text_input("Primary language", value="Python")

if "license" in feature_cols:
    if licenses:
        inputs["license"] = st.selectbox("License", licenses, index=0)
    else:
        inputs["license"] = st.text_input("License", value="mit")

if "default_branch" in feature_cols:
    inputs["default_branch"] = st.selectbox("Default branch", branches, index=(branches.index("main") if "main" in branches else 0))

if "topics" in feature_cols:
    st.caption("Comma-separated topics, e.g. `machine-learning, deep-learning, pytorch`")
    inputs["topics"] = st.text_input("Topics", value="machine-learning, data-science")

X_pred = pd.DataFrame([{c: inputs.get(c, None) for c in feature_cols}], columns=feature_cols)

st.divider()
if st.button("Predict"):
    proba = float(pipeline.predict_proba(X_pred)[0, 1])
    pred = int(proba >= 0.5)
    st.metric("High-star probability", f"{proba:.2%}")
    st.write("Prediction:", "HIGH-STAR repo" if pred else "Not high-star")
