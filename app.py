import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GitHub Repo Success Predictor", layout="centered")

@st.cache_resource
def load_artifact():
    return joblib.load("github_repo_success_model.joblib")

artifact = load_artifact()
pipeline = artifact["pipeline"]
feature_cols = artifact["feature_cols"]

st.title("GitHub Repo Success Predictor")
st.write("Predict whether a repo is likely to be a **high-star** repository (as defined in the dataset).")

st.subheader("Repo inputs")

# Build inputs dynamically based on model features
inputs = {}

# Numeric-ish inputs
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

# Boolean-ish inputs
def bool_select(label, default=False):
    return 1 if st.selectbox(label, ["No", "Yes"], index=(1 if default else 0)) == "Yes" else 0

if "is_fork" in feature_cols:
    inputs["is_fork"] = bool_select("Is fork?", default=False)
if "has_wiki" in feature_cols:
    inputs["has_wiki"] = bool_select("Has wiki?", default=True)
if "archived" in feature_cols:
    inputs["archived"] = bool_select("Archived?", default=False)

# Categorical inputs
if "language" in feature_cols:
    inputs["language"] = st.text_input("Primary language", value="Python")
if "license" in feature_cols:
    inputs["license"] = st.text_input("License", value="mit")
if "default_branch" in feature_cols:
    inputs["default_branch"] = st.text_input("Default branch", value="main")

if "topics" in feature_cols:
    st.caption("Comma-separated topics, e.g. `machine-learning, deep-learning, pytorch`")
    inputs["topics"] = st.text_input("Topics", value="machine-learning, data-science")

# Build one-row dataframe with exact feature order
row = {c: inputs.get(c, None) for c in feature_cols}
X = pd.DataFrame([row], columns=feature_cols)

st.divider()
if st.button("Predict"):
    proba = float(pipeline.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)

    st.metric("High-star probability", f"{proba:.2%}")
    st.write("Prediction:", "HIGH-STAR repo" if pred == 1 else "Not high-star")
