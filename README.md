# 📈 LocalBiz — local-business survival-risk predictor

**Live demo:** _(deploying — link goes here)_

Pick a **business type** and **place**, and LocalBiz gives it a survival-risk read
(🟢 lower / 🟡 medium / 🔴 higher) from patterns in **150,000+ real businesses** in
the [Yelp Open Dataset](https://www.yelp.com/dataset). Project 3 of a 3-app AI
portfolio — the **Prediction** pillar (machine learning on messy real data).

## What it does

1. You choose a business type, city, star rating, and review count.
2. A RandomForest model (trained on real Yelp data) predicts the probability the
   business is **open vs. closed**.
3. You get a risk band, a Google Maps link to businesses like it, and the top
   factors driving the prediction.

## Honest limits

- `is_open` is a **current snapshot** in Yelp's data, not verified long-term survival.
- The model finds **correlations** (type, location, rating, review volume) — it is
  **not a guarantee** about any specific business.
- Accuracy is reported in-app against a dumb always-guess-majority baseline, so you
  can see how much signal the model actually adds.

## Run it yourself

```bash
# 1. get the data (one-time, terms-gated): https://www.yelp.com/dataset
#    put yelp_academic_dataset_business.json into data/
# 2. train the model
python -I train.py data/yelp_academic_dataset_business.json
# 3. launch
./scripts/dev.ps1        # or: streamlit run app.py
```

## Stack

Streamlit · pandas · scikit-learn (RandomForest) · Yelp Open Dataset

## What's next

- Add neighborhood/foot-traffic features for a sharper signal
- Calibrate probabilities so the % reads as a true likelihood
- Time-aware labels (track closures over time) instead of a single snapshot
