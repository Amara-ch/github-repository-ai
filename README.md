# github-repository-ai
# GitHub Repo Success Predictor (2026)

Predict whether a GitHub repository is likely to be a **high-star** repo using metadata features (language, forks, watchers, topics, license, repo age, recency of updates, etc.).  
Includes an interactive **Streamlit app** for live predictions.

## Dataset
**Top GitHub Repositories 2026** (627 most-starred repos) with:
- stars, forks, watchers, open issues
- language, topics, license
- created/updated timestamps
- repo flags (archived, has wiki, fork)
- size and default branch

## Approach
- Cleaned and normalized metadata (especially `topics`)
- Engineered time-based features:
  - `repo_age_days`
  - `days_since_update`
- Trained a classification model to predict **high_star_repo**
- Packaged the full preprocessing + model pipeline and deployed via Streamlit

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` – Streamlit UI for predictions
- `github_repo_success_model.joblib` – trained model + preprocessing pipeline
- `requirements.txt` – dependencies
