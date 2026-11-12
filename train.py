"""Train the LocalBiz open/closed predictor on the Yelp Open Dataset.

Usage (run from the localbiz/ folder, with the venv active):
    python -I train.py data/yelp_academic_dataset_business.json

Reads Yelp's line-delimited business JSON, builds a small tabular model that
predicts whether a business is currently open (is_open), and saves:
    models/model.joblib   the trained sklearn pipeline
    models/meta.json      dropdown options + metrics + top factors for the app

Honest framing: is_open is a current snapshot, not verified long-term survival.
The model finds correlations (category, location, rating, review volume), it is
not a guarantee. The app says this plainly.
"""
import sys, json, time, pathlib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

TOP_CATEGORIES = 40      # keep the most common business types, rest -> "Other"
TOP_CITIES = 40
HERE = pathlib.Path(__file__).resolve().parent
MODELS = HERE / "models"


def primary_category(cats):
    if not cats:
        return "Other"
    return str(cats).split(",")[0].strip() or "Other"


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue
            if b.get("is_open") is None or not b.get("categories"):
                continue
            rows.append({
                "category": primary_category(b.get("categories")),
                "city": (b.get("city") or "Unknown").strip() or "Unknown",
                "state": (b.get("state") or "NA").strip() or "NA",
                "stars": float(b.get("stars") or 0),
                "review_count": int(b.get("review_count") or 0),
                "is_open": int(b.get("is_open")),
            })
    return pd.DataFrame(rows)


def cap(df, col, n):
    keep = set(df[col].value_counts().head(n).index)
    df[col] = df[col].where(df[col].isin(keep), other="Other")
    return sorted(keep)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -I train.py <path-to-yelp_academic_dataset_business.json>")
    src = sys.argv[1]
    print(f"Loading {src} ...")
    df = load(src)
    print(f"  {len(df):,} businesses with a label + category")
    if len(df) < 1000:
        sys.exit("Too few rows — is this the Yelp business JSON?")

    cats = cap(df, "category", TOP_CATEGORIES)
    cities = cap(df, "city", TOP_CITIES)
    states = sorted(df["state"].unique().tolist())
    open_rate = df["is_open"].mean()

    X = df[["category", "city", "state", "stars", "review_count"]]
    y = df["is_open"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["category", "city", "state"]),
    ], remainder="passthrough")
    model = Pipeline([
        ("pre", pre),
        # natural prior (no class_weight) so predict_proba is calibrated to the
        # true ~80% open base rate — the app frames risk relative to that rate.
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=16, min_samples_leaf=5,
            n_jobs=-1, random_state=42)),
    ])
    print("Training RandomForest ...")
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, proba)
    baseline = max(open_rate, 1 - open_rate)  # always-guess-majority accuracy

    # top factors: map RF importances back to readable feature names
    names = model.named_steps["pre"].get_feature_names_out()
    imps = model.named_steps["rf"].feature_importances_
    top = sorted(zip(names, imps), key=lambda t: t[1], reverse=True)[:12]
    top_factors = [[n.split("__", 1)[-1], round(float(i), 4)] for n, i in top]

    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "model.joblib")
    meta = {
        "categories": cats, "cities": cities, "states": states,
        "open_rate": round(float(open_rate), 4),
        "n_rows": int(len(df)),
        "metrics": {
            "test_accuracy": round(float(acc), 4),
            "test_auc": round(float(auc), 4),
            "baseline_accuracy": round(float(baseline), 4),
        },
        "top_factors": top_factors,
        "trained_at": time.strftime("%Y-%m-%d"),
    }
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone. test_acc={acc:.3f}  AUC={auc:.3f}  baseline={baseline:.3f}  open_rate={open_rate:.3f}")
    print(f"Saved models/model.joblib + models/meta.json")


if __name__ == "__main__":
    main()
