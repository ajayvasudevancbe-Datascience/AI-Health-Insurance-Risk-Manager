"""
app.py
------
AI Health Insurance Assistant

A beginner-friendly Streamlit app that combines:
1. A simple rule-based / optional-LLM chatbot that explains health
   insurance concepts.
2. A Machine Learning model that estimates a health insurance premium
   from basic user details (age, sex, bmi, children, smoker, region).

Run locally with:
    streamlit run app.py

IMPORTANT: This app produces an educational ESTIMATE only. It is not an
official insurance quote, and not medical or financial advice.
"""

import os
import re

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Optional LLM integration (Google Gemini)
# --------------------------------------------------------------------------
# If a Gemini API key is available in st.secrets, the chatbot will use it
# for more natural conversational answers. Otherwise, it automatically
# falls back to the built-in rule-based responder below — the app never
# crashes just because no API key was configured.

USE_GEMINI = False
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
except Exception:
    GEMINI_API_KEY = None

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
        USE_GEMINI = True
    except Exception:
        # If the package isn't installed or configuration fails, silently
        # fall back to rule-based answers instead of crashing the app.
        USE_GEMINI = False


MODEL_PATH = "health_insurance_model.pkl"

VALID_REGIONS = ["northeast", "northwest", "southeast", "southwest"]

SYSTEM_CONTEXT = (
    "You are a friendly, beginner-friendly assistant that explains general "
    "health insurance concepts in simple terms. You are not a licensed "
    "insurance agent, financial advisor, or medical professional. Never "
    "claim any number is an official insurance quote. Keep answers short "
    "(2-4 sentences) and easy to understand."
)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained model. Returns None if the file is missing/broken,
    so the rest of the app can degrade gracefully instead of crashing."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.session_state["model_load_error"] = str(e)
        return None


# --------------------------------------------------------------------------
# Rule-based chatbot fallback (no API key required)
# --------------------------------------------------------------------------
def rule_based_response(question: str) -> str:
    q = question.lower()

    def has(*keywords):
        return any(k in q for k in keywords)

    if has("what affects", "factors", "affect my premium", "affect premium"):
        return (
            "Several factors can influence an estimated premium: age, BMI, "
            "smoking status, number of children, and region. In many "
            "datasets, smoking status and age tend to have the strongest "
            "effect."
        )

    if has("why is my premium high", "why high", "why is it high", "why so high"):
        return (
            "A higher estimated premium is usually driven by a combination "
            "of factors — most commonly older age, a higher BMI, or a "
            "'smoker' status. If several of these apply to you, the model "
            "will typically estimate a higher premium."
        )

    if has("bmi", "body mass index"):
        return (
            "BMI (Body Mass Index) is a number calculated from your height "
            "and weight. It's used here as one of the inputs the prediction "
            "model considers, since BMI is often associated with certain "
            "health risk factors."
        )

    if has("smok"):
        return (
            "Smoking is generally associated with higher healthcare risk, "
            "so many insurance prediction models — including this one — "
            "estimate higher premiums for smokers compared to non-smokers."
        )

    if has("does age", "age affect", "age impact"):
        return (
            "Yes — age is typically one of the strongest predictors of "
            "premium cost. Older age groups are usually associated with "
            "higher estimated premiums in most models."
        )

    if has("how does the prediction work", "how does prediction work", "how does this work", "how it works"):
        return (
            "The prediction comes from a Machine Learning model trained on "
            "example data. It looks at your age, sex, BMI, number of "
            "children, smoker status, and region, then estimates a premium "
            "based on patterns it learned during training. It's an "
            "educational estimate, not an official quote."
        )

    if has("what is health insurance", "health insurance?"):
        return (
            "Health insurance is a way to help cover medical costs. You (or "
            "an employer) typically pay a regular amount called a premium, "
            "and in exchange the insurer helps pay for covered medical "
            "expenses when you need care."
        )

    if has("compare", "compare plan", "compare insurance"):
        return (
            "When comparing insurance plans, it helps to look at the "
            "premium, the deductible (what you pay before coverage kicks "
            "in), the coverage limits, and which doctors/hospitals are "
            "included in the network. This app doesn't compare real plans — "
            "it only gives a simple educational premium estimate."
        )

    if has("region"):
        return (
            "Region can affect the estimated premium because healthcare "
            "costs and risk patterns can vary from one area to another in "
            "the training data used by the model."
        )

    if has("children", "kids", "dependents"):
        return (
            "The number of children/dependents is included as an input "
            "because having more dependents can be associated with "
            "different overall healthcare usage patterns in the data."
        )

    if has("hello", "hi", "hey"):
        return "Hi there! Ask me anything about health insurance, or fill out the form to get an estimated premium."

    # Check if the user is quoting a predicted premium amount
    money_match = re.search(r"₹?\s?([\d,]{3,})", q)
    if money_match and has("premium", "why"):
        return (
            "Your estimated premium is influenced by the details entered "
            "into the prediction model — age, BMI, smoking status, number "
            "of children, and region. Age and smoking status tend to have "
            "a particularly strong effect in many insurance datasets."
        )

    return (
        "I can help explain general health insurance concepts like premiums, "
        "BMI, or how age and smoking status affect estimates. Try one of the "
        "quick-question buttons, or ask me something like 'What affects my "
        "premium?'"
    )


def gemini_response(question: str, prediction_context: str | None) -> str:
    """Get a response from Gemini, with the rule-based responder as a
    safety-net fallback if the API call fails for any reason."""
    try:
        prompt = SYSTEM_CONTEXT
        if prediction_context:
            prompt += f"\n\nContext about the user's current estimate: {prediction_context}"
        prompt += f"\n\nUser question: {question}"
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return rule_based_response(question)


def get_bot_response(question: str) -> str:
    prediction_context = None
    if st.session_state.get("last_prediction") is not None:
        prediction_context = (
            f"The user's most recent estimated premium was "
            f"₹{st.session_state['last_prediction']:,.0f}, based on their "
            f"submitted age, sex, BMI, children, smoker status, and region."
        )

    if USE_GEMINI:
        return gemini_response(question, prediction_context)
    return rule_based_response(question)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
def validate_inputs(age, bmi, children):
    errors = []
    if not (18 <= age <= 100):
        errors.append("Age must be between 18 and 100.")
    if not (10.0 <= bmi <= 60.0):
        errors.append("BMI must be between 10 and 60.")
    if not (0 <= children <= 10):
        errors.append("Number of children must be between 0 and 10.")
    return errors


# --------------------------------------------------------------------------
# Streamlit page config & session state
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Health Insurance Assistant",
    page_icon="🏥",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm your AI Health Insurance Assistant. Ask me a question, or fill out the form on the right to get an estimated premium.",
        }
    ]

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

model = load_model()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🏥 AI Health Insurance Assistant")
st.caption(
    "Educational demo — provides an ML-based premium **estimate**, not an official insurance quote."
)

if model is None:
    st.warning(
        "⚠️ The prediction model file `health_insurance_model.pkl` was not found "
        "(or could not be loaded). The premium prediction section will be "
        "disabled, but you can still chat with the assistant. See README.md "
        "for how to add the model file, or run `python train_model.py` to "
        "generate a demo model."
    )

st.divider()

col_chat, col_predict = st.columns([1.2, 1], gap="large")

# --------------------------------------------------------------------------
# Chat column
# --------------------------------------------------------------------------
with col_chat:
    st.subheader("💬 Chat with Insurance Assistant")

    quick_questions = [
        "What affects my premium?",
        "Why is my premium high?",
        "What is BMI?",
        "How does smoking affect premium?",
        "Does age affect premium?",
        "How does the prediction work?",
        "What is health insurance?",
        "How can I compare insurance plans?",
    ]

    st.write("Quick questions:")
    qcols = st.columns(2)
    for i, qq in enumerate(quick_questions):
        if qcols[i % 2].button(qq, key=f"quick_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": qq})
            answer = get_bot_response(qq)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    chat_container = st.container(height=350)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    user_input = st.chat_input("Ask your question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        answer = get_bot_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# --------------------------------------------------------------------------
# Prediction column
# --------------------------------------------------------------------------
with col_predict:
    st.subheader("📊 Premium Prediction")

    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", VALID_REGIONS)

        submitted = st.form_submit_button("Predict Premium", use_container_width=True)

    if submitted:
        errors = validate_inputs(age, bmi, children)
        if errors:
            for err in errors:
                st.error(err)
        elif model is None:
            st.error(
                "Cannot generate a prediction because the model file is "
                "missing. Please add `health_insurance_model.pkl` to the "
                "project folder (see README.md)."
            )
        else:
            input_df = pd.DataFrame(
                [
                    {
                        "age": age,
                        "sex": sex,
                        "bmi": bmi,
                        "children": children,
                        "smoker": smoker,
                        "region": region,
                    }
                ]
            )
            try:
                prediction = model.predict(input_df)[0]
                prediction = max(prediction, 0)
                st.session_state.last_prediction = prediction

                st.success("Prediction complete!")
                st.metric("Estimated Premium", f"₹{prediction:,.0f}")

                # Simple, transparent explanation of key drivers
                reasons = []
                if smoker == "yes":
                    reasons.append("smoker status")
                if age >= 45:
                    reasons.append("age")
                if bmi >= 30:
                    reasons.append("higher BMI")
                if children >= 3:
                    reasons.append("number of children")

                if reasons:
                    st.info(
                        "This estimate is likely influenced upward by: "
                        + ", ".join(reasons)
                        + "."
                    )
                else:
                    st.info(
                        "Based on the details entered, none of the higher-risk "
                        "factors (older age, high BMI, smoking) strongly apply, "
                        "which tends to keep the estimate lower."
                    )
            except Exception as e:
                st.error(f"Something went wrong while generating the prediction: {e}")

    if st.session_state.last_prediction is not None:
        st.caption(
            f"Last estimated premium: ₹{st.session_state.last_prediction:,.0f} "
            "— ask the chatbot 'Why is my premium high?' to learn more."
        )

st.divider()

# --------------------------------------------------------------------------
# Footer / disclaimer
# --------------------------------------------------------------------------
st.subheader("ℹ️ Important Information")
st.write(
    "This application provides an ML-based estimate for **educational "
    "purposes only**. It is **not** an official insurance quote, and it is "
    "not financial or medical advice. Actual insurance premiums depend on "
    "many additional factors determined by insurance providers."
)
