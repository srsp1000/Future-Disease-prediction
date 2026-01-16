# 
import streamlit as st
import torch
import json
import os
import google.generativeai as genai
from transformers import AutoTokenizer
from model import TransformerForNextCond

st.set_page_config(page_title="Next Disease Predictor", layout="wide")

# Paths
ENCODER_PATH = "final_model/encoder" 
FULL_MODEL_STATE_PATH = "final_model/full_model_with_classifier.pt"
VOCAB_PATH = "vocab.json"
MAX_LEN = 256

GEMINI_API_KEY = "---------put gemini api key--------------"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

#  MAPPINGS 
with open("code_to_name.json", "r") as f:
    code_to_name = json.load(f)

# MODEL RESOURCES
@st.cache_resource
def load_model_resources():
    try:
        if not os.path.exists(ENCODER_PATH) or not os.path.exists(FULL_MODEL_STATE_PATH):
            st.error("Model files not found! Ensure 'final_model' folder is set up correctly.")
            return None, None, None

        tokenizer = AutoTokenizer.from_pretrained(ENCODER_PATH)
        
        with open(VOCAB_PATH, 'r') as f:
            idx_to_cond = json.load(f)
            idx_to_cond = {int(k): v for k, v in idx_to_cond.items()}

        num_labels = len(idx_to_cond)
        model = TransformerForNextCond(base_model_name=ENCODER_PATH, num_labels=num_labels)
        model.load_state_dict(torch.load(FULL_MODEL_STATE_PATH, map_location=torch.device('cpu')))
        model.eval()
        return tokenizer, model, idx_to_cond
    except Exception as e:
        st.error(f"Error loading model resources: {e}")
        return None, None, None

tokenizer, model, idx_to_cond = load_model_resources()

#NPUT FUNCTION 
def get_structured_input(user_text, full_vocab_str):
    try:
        llm = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
        Convert the natural language patient summary into a structured sequence of medical codes.
        ONLY use codes from the provided vocabulary.
        Format: [AGE_*] [SEX_*] COND_xxx COND_xxx ...

        VOCABULARY:
        {full_vocab_str}

        PATIENT SUMMARY:
        "{user_text}"
        ---
        STRUCTURED OUTPUT:
        """
        response = llm.generate_content([prompt])
        return response.text.strip().replace("\n", " ")
    except Exception as e:
        st.error(f"Error processing input: {e}")
        return None

def verify_prediction(structured_input, predictions, code_to_name):
    try:
        llm = genai.GenerativeModel("models/gemini-2.5-pro")
        prediction_names = [code_to_name.get(cond, cond) for cond in predictions]

        prompt = f"""
        Patient structured history:
        {structured_input}

        Model predicted next conditions:
        1. {prediction_names[0]}
        2. {prediction_names[1]}
        3. {prediction_names[2]}

        Pick the MOST likely condition, give a short explanation, and 1-2 precautions.

        Format:
        CONDITION: <name>
        EXPLANATION: <reasoning>
        PRECAUTIONS: <bulleted list>
        """
        response = llm.generate_content([prompt])
        return response.text.strip()
    except Exception as e:
        st.error(f"Error during verification: {e}")
        return None

# --- MODEL PREDICTION ---
def predict(structured_text):
    if not structured_text or not tokenizer or not model:
        return None
    inputs = tokenizer(structured_text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN)
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
    probabilities = torch.softmax(logits, dim=1)[0]
    top3_probs, top3_indices = torch.topk(probabilities, 3)

    predictions = []
    for i in range(top3_indices.size(0)):
        pred_idx = top3_indices[i].item()
        pred_condition = idx_to_cond.get(pred_idx, "Unknown Condition")
        predictions.append(pred_condition)
    return predictions

# --- STREAMLIT UI ---
st.title("🏥  Next Disease Predictor")
st.markdown("Enter a patient's medical history. The model will predict the top 3 most likely subsequent conditions and provide a recommended top condition with explanation and precautions.")

if idx_to_cond:
    all_codes = [v for v in idx_to_cond.values()]
    structural_tokens = ["[TIME_GAP_WEEK]", "[TIME_GAP_MONTH]", "[TIME_GAP_YEAR]", "[TIME_GAP_LONG]"]
    age_tokens = [f"[AGE_{i*10}-{(i+1)*10-1}]" for i in range(10)]
    sex_tokens = ["[SEX_M]", "[SEX_F]"]
    full_vocab_list = all_codes + structural_tokens + age_tokens + sex_tokens
    full_vocab_str = " ".join(full_vocab_list)
else:
    full_vocab_str = "Vocabulary not loaded."

patient_history = st.text_area("Enter Patient History:", height=200, placeholder="e.g., A 62-year-old female with diabetes and hypertension...")

if st.button("Predict Next Condition"):
    if not GEMINI_API_KEY:
        st.warning("API Key is missing. Cannot process.")
    elif not patient_history:
        st.warning("Please enter patient history.")
    elif not model or not tokenizer or not idx_to_cond:
        st.error("Model not loaded properly.")
    else:
        with st.spinner("Predicting..."):
            structured_input = get_structured_input(patient_history, full_vocab_str)
            if structured_input:
                st.subheader("Processed Input")
                st.info(f"`{structured_input}`")

                predictions = predict(structured_input)
                if predictions:
                    st.subheader("Top 3 Predicted Conditions (Model Output)")
                    for condition in predictions:
                        disease_name = code_to_name.get(condition, condition)
                        st.write(f"🔹 {disease_name}")

                    
                    top1_output = verify_prediction(structured_input, predictions, code_to_name)
                    if top1_output:
                        st.subheader("Recommended Condition with Explanation & Precautions")
                        st.success(top1_output)
