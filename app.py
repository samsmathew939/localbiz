"""LocalBiz — predict whether a local business is likely open or closed.

A quick, honest closure-risk read for a storefront or small-business idea.
The model is trained by train.py on the Yelp Open Dataset.
"""
import json
import pathlib
import urllib.parse
import pandas as pd
import streamlit as st

HERE = pathlib.Path(__file__).resolve().parent
MODELS = HERE / "models"

st.set_page_config(page_title="LocalBiz", page_icon="📈", layout="centered")


def _safe(text):
    """Escape markdown-special chars in untrusted text before display."""
    for ch in r"\`*_{}[]()#+-.!|":
        text = text.replace(ch, "\\" + ch)
    return text


@st.cache_resource
def load_model():
    # Trusted artifact: models/model.joblib is produced by our own train.py and
    # committed to this repo — not untrusted user input. Safe to joblib.load.
    import joblib
    model = joblib.load(MODELS / "model.joblib")
    meta = json.loads((MODELS / "meta.json").read_text(encoding="utf-8"))
    return model, meta


st.title("📈 LocalBiz")
st.markdown(
    "**Is a local business likely to still be open — or at risk of closing?** "
    "Pick a business type and place, and LocalBiz gives it a survival-risk read "
    "from patterns in 150,000+ real businesses."
)

if not (MODELS / "model.joblib").exists():
    st.warning(
        "⚙️ The model isn't trained yet. Download the Yelp Open Dataset "
        "(https://www.yelp.com/dataset), put `yelp_academic_dataset_business.json` "
        "in the `data/` folder, and run `python -I train.py "
        "data/yelp_academic_dataset_business.json`."
    )
    st.stop()

model, meta = load_model()

c1, c2 = st.columns(2)
with c1:
    category = st.selectbox("Business type", meta["categories"])
    city = st.selectbox("City", meta["cities"])
with c2:
    stars = st.slider("Star rating", 1.0, 5.0, 3.5, 0.5)
    review_count = st.number_input("Number of reviews", min_value=0, value=30, step=10)

# state most associated with the chosen city, else first available
state = meta["states"][0] if meta["states"] else "NA"

base = meta.get("open_rate", 0.8)  # dataset average: share of businesses open

if st.button("Check survival risk", type="primary"):
    row = pd.DataFrame([{
        "category": category, "city": city, "state": state,
        "stars": float(stars), "review_count": int(review_count),
    }])
    p_open = float(model.predict_proba(row)[0][1])
    avg = base * 100

    # Risk is judged RELATIVE to the dataset average (most businesses are open),
    # so a number near the average is "typical", not alarming.
    if p_open >= base + 0.05:
        st.success(f"## 🟢 Lower risk than average\n**{p_open*100:.0f}%** likely to be open "
                   f"— better than the typical business ({avg:.0f}%).")
    elif p_open >= base - 0.10:
        st.warning(f"## 🟡 About average\n**{p_open*100:.0f}%** likely to be open "
                   f"— close to the typical business ({avg:.0f}%).")
    else:
        st.error(f"## 🔴 Higher risk than average\n**{p_open*100:.0f}%** likely to be open "
                 f"— below the typical business ({avg:.0f}%).")

    st.caption(f"{_safe(category)} in {_safe(city)} · {stars:.1f}★ · {review_count} reviews")

    # map
    q = urllib.parse.quote_plus(f"{category} in {city}")
    st.link_button("🗺️ See businesses like this on Google Maps",
                   f"https://www.google.com/maps/search/{q}")

    st.divider()
    st.markdown("**What drives the prediction most** (across all businesses):")
    for name, imp in meta["top_factors"][:6]:
        if name in ("stars", "review_count"):
            label = {"stars": "Star rating", "review_count": "Number of reviews"}[name]
        elif "_" in name:
            kind, val = name.split("_", 1)
            prefix = {"category": "Business type", "city": "City", "state": "State"}.get(kind, kind)
            label = f"{prefix}: {val}"
        else:
            label = name
        st.write(f"- {_safe(str(label))}")

with st.expander("How accurate is this? (honest limits)"):
    m = meta["metrics"]
    st.markdown(
        f"- Trained on **{meta['n_rows']:,}** real businesses from the Yelp Open Dataset.\n"
        f"- **AUC {m['test_auc']:.2f}** — given one open and one closed business, the model ranks "
        f"them correctly {m['test_auc']*100:.0f}% of the time (a coin flip is 50%). That's the honest "
        f"measure here.\n"
        f"- Raw accuracy ({m['test_accuracy']*100:.0f}%) isn't the useful number: **{base*100:.0f}%** of "
        f"businesses are open, so always guessing \"open\" already scores ~{m['baseline_accuracy']*100:.0f}%. "
        f"The model's value is **ranking risk**, not labeling.\n"
        f"- `is_open` is a **current snapshot**, not verified long-term survival. This finds "
        f"correlations (type, place, rating, reviews) — it is **not a guarantee**.\n"
        f"- Source: Yelp Open Dataset. Trained {meta['trained_at']}."
    )
