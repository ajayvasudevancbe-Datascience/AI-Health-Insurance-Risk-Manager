# 🏥 AI Health Insurance Assistant

A beginner-friendly Streamlit app that combines a simple chatbot with a
Machine Learning model to give an **educational estimate** of a health
insurance premium.

> ⚠️ **Disclaimer:** This app produces an ML-based estimate for learning
> purposes only. It is **not** an official insurance quote, and not
> financial or medical advice.

---

## 1. Project Structure

```
health-insurance-chatbot/
│
├── app.py                          # Main Streamlit app (chatbot + prediction UI)
├── train_model.py                  # Generates a demo model (see section 3)
├── health_insurance_model.pkl      # Trained ML model (place here — see section 3)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Files Git should ignore
└── .streamlit/
    └── secrets.toml.example        # Template for optional Gemini API key
```

---

## 2. Local Setup

1. **Clone / download** this folder.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate       # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Make sure the model file exists** (see section 3 below).

5. **Run the app**:
   ```bash
   streamlit run app.py
   ```

6. Open the URL Streamlit prints in your terminal (usually
   `http://localhost:8501`).

---

## 3. Where `health_insurance_model.pkl` Goes

The app expects a trained scikit-learn pipeline saved at:

```
health-insurance-chatbot/health_insurance_model.pkl
```

**Option A — Use your own trained model**
If you already trained a model (for example on the classic Kaggle
`insurance.csv` dataset with columns `age, sex, bmi, children, smoker,
region, charges`), just export it with `joblib.dump(...)` and place the
resulting `.pkl` file directly in the project's **root folder**, next to
`app.py`. Make sure:
- It accepts a pandas DataFrame with columns:
  `age, sex, bmi, children, smoker, region`
- It has a `.predict()` method (a scikit-learn `Pipeline` that includes
  its own preprocessing is easiest, since `app.py` passes raw values).

**Option B — Generate a demo model (no dataset needed)**
No real dataset was provided with this project, so a helper script,
`train_model.py`, is included. It builds a small synthetic dataset with
realistic-ish relationships (older age, higher BMI, and smoking increase
the estimated premium) and trains a `RandomForestRegressor` on it, then
saves it as `health_insurance_model.pkl`.

Run it once from the project folder:
```bash
python train_model.py
```

This creates `health_insurance_model.pkl` automatically. Swap it out
later with your own real model whenever you have one — `app.py` doesn't
need to change.

**If the file is missing:** the app will **not crash**. It shows a clear
warning banner and disables only the prediction section; the chatbot
still works normally.

---

## 4. How the Chatbot Works

The chatbot works in two modes:

1. **Rule-based mode (default, no API key needed)** — `app.py` includes a
   small keyword-matching function (`rule_based_response`) that
   recognizes common questions (e.g. about BMI, smoking, age, regions)
   and returns a simple, pre-written explanation. This mode works
   completely offline and requires no setup.

2. **Optional LLM mode (Google Gemini)** — If you add a `GEMINI_API_KEY`
   to `.streamlit/secrets.toml` (see section 5), the app automatically
   switches to using Gemini for more natural, flexible answers. If the
   Gemini call ever fails for any reason (bad key, no internet, quota
   limits), the app automatically falls back to the rule-based responder
   instead of crashing.

The chatbot also has light integration with the prediction: after you
generate an estimate, the assistant is given a short note about your
most recent predicted premium so it can answer follow-up questions like
*"Why is my premium high?"* in context.

---

## 5. (Optional) Adding a Gemini API Key

If you'd like more natural chatbot responses:

1. Get an API key from [Google AI Studio](https://aistudio.google.com/).
2. Create the folder `.streamlit/` in the project root if it doesn't
   already exist.
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
4. Replace the placeholder with your real key:
   ```toml
   GEMINI_API_KEY = "your-real-key-here"
   ```
5. Install the optional package:
   ```bash
   pip install google-generativeai
   ```
   (also uncomment it in `requirements.txt` if you plan to deploy).

**Never** commit `secrets.toml` to GitHub — it's already listed in
`.gitignore`. Never hard-code the key inside `app.py`.

If you skip this whole section, the app works fine using the built-in
rule-based chatbot.

---

## 6. How the ML Prediction Works

1. You fill out the form: **Age, Sex, BMI, Children, Smoker, Region**.
2. `app.py` validates the inputs (e.g. age between 18–100, BMI between
   10–60).
3. The values are placed into a single-row pandas DataFrame with the
   exact column names the model expects.
4. The trained pipeline (loaded from `health_insurance_model.pkl`)
   preprocesses the categorical fields (`sex`, `smoker`, `region`) and
   passes numeric fields through, then predicts a premium.
5. The result is displayed as **Estimated Insurance Premium: ₹XX,XXX**,
   along with a short, transparent note about which factors likely
   pushed the estimate up (e.g. smoker status, higher age, higher BMI).

The prediction is always labeled as an **estimate**, never as an actual
quote.

---

## 7. Deploying to Streamlit Community Cloud

1. Push this project to a **public or private GitHub repository**,
   including `health_insurance_model.pkl` (unless it's very large — see
   note below) but **excluding** `.streamlit/secrets.toml`.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in
   with GitHub.
3. Click **"New app"**, then select:
   - Repository: your repo
   - Branch: `main` (or whichever you used)
   - Main file path: `app.py`
4. If you're using the optional Gemini key, add it under **App settings
   → Secrets** in the same `TOML` format as `secrets.toml.example`:
   ```toml
   GEMINI_API_KEY = "your-real-key-here"
   ```
5. Click **Deploy**. Streamlit Cloud will install everything listed in
   `requirements.txt` automatically.

**Note on the model file size:** GitHub works best with files under
~100MB. The demo model created by `train_model.py` is small and safe to
commit. If your own trained model is very large, consider using
[Git LFS](https://git-lfs.com/) or loading it from external storage at
startup instead.

---

## 8. Notes & Limitations

- This is an **educational demo**, not a production insurance tool.
- The demo model (from `train_model.py`) is trained on **synthetic**
  data, not real insurance records — replace it with your own trained
  model for anything beyond learning/demo purposes.
- The chatbot's rule-based answers are intentionally simple and general;
  they are not personalized financial or medical guidance.
