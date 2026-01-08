import streamlit as st
import numpy as np
import re
import joblib
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. LOAD ASSETS
# ---------------------------------------------------------
@st.cache_resource
def load_assets():

    clf_model = joblib.load('clf_model.pkl')
    reg_model = joblib.load('reg_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    
    return clf_model, reg_model, scaler, sentence_transformer

clf_model, reg_model, scaler, sentence_transformer = load_assets()

# ---------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------
LATEX_PATTERNS = [
    r'\$\$.*?\$\$',
    r'\$[^$]*\$',
    r'\\\(.+?\\\)',
    r'\\\[.+?\\\)'
]
LATEX_MAP = {
    r'\\leq': '<=',
    r'\\le': '<',
    r'\\ge': '>',
    r'\\geq': '>=',
    r'\\neq': '!=',
    r'\\times': '*',
    r'\\cdot': '*',
    r'\\dots': '...',
    r'\\ldots': '...',
    r'\\cdots': '...',
    r'\\\{': '{',
    r'\\\}': '}',
    r'\\,\s*': ''
}
def extract_latex(text):
    latex_blocks = []

    def repl(match):
        latex_blocks.append(match.group())
        return f" <LATEX_{len(latex_blocks)-1}> "

    for pattern in LATEX_PATTERNS:
        text = re.sub(pattern, repl, text)

    return text, latex_blocks



def clean_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



def normalize_fractions(expr):
    pattern = r'\\frac\{([^{}]+)\}\{([^{}]+)\}'
    while re.search(pattern, expr):
        expr = re.sub(pattern, r'(\1)/(\2)', expr)
    return expr

def normalize_latex(expr):
    expr = expr.strip()
    if expr.startswith("$$") and expr.endswith("$$"):
        expr = expr[2:-2]
    elif expr.startswith("$") and expr.endswith("$"):
        expr = expr[1:-1]
    elif expr.startswith("\\(") and expr.endswith("\\)"):
        expr = expr[2:-2]
    elif expr.startswith("\\[") and expr.endswith("\\]"):
        expr = expr[2:-2]

    expr = normalize_fractions(expr)

    for k, v in LATEX_MAP.items():
        expr = re.sub(k, v, expr)

    return expr.strip()



def reinsert(text, latex_blocks, normalize=False):
    for i, block in enumerate(latex_blocks):
        if normalize:
            block = normalize_latex(block)
        text = text.replace(f"<LATEX_{i}>", block)
    return text

def preprocess_text(text):
    protected_text, latex_blocks = extract_latex(text)
    cleaned = clean_text(protected_text)
    semantic_version = reinsert(cleaned, latex_blocks, normalize=True)
    return semantic_version

POWER_PATTERNS = [
    r'\b\w+\s*\^\s*\w+\b',
    r'\b\w+\s*\^\{\s*\w+\s*\}',
]

power_regex = re.compile("|".join(POWER_PATTERNS))

def has_power_notation(text):
    if not isinstance(text, str):
        return 0
    return int(bool(power_regex.search(text)))

def get_features(text, embedding_model):
    clean_text = preprocess_text(text)
    
    embedding = embedding_model.encode([clean_text]) 
    
    text_len = len(clean_text)
    math_count = sum(clean_text.count(c) for c in "<>=^*/+-")
    num_count = len(re.findall(r'\d+', clean_text))
    has_power = has_power_notation(clean_text)
    manual_feats = np.array([[has_power, num_count, math_count, text_len]])
    
    return np.hstack([embedding, manual_feats])

# ---------------------------------------------------------
# 3. USER INTERFACE
# ---------------------------------------------------------
st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("AutoJudge : Difficulty Rater", text_alignment='center')
st.markdown("**Predict both the category and the exact score**.", text_alignment='center')

col1, col2 = st.columns(2)
with col1:
    input_desc = st.text_area("Input Description", height=100)
with col2:
    output_desc = st.text_area("Output Description", height=100)

problem_text = st.text_area("Problem Description", height=200, placeholder="Paste the full problem text here")

if st.button("Analyze Difficulty", type="primary"):
    if not problem_text:
        st.warning("Please enter the problem text.")
    else:
        with st.spinner("Analyzing complexity..."):
            # 1. Prepare Data
            full_text = f"PROBLEM: {problem_text} INPUT_DESCRIPTION: {input_desc} OUTPUT_DESCRIPTION: {output_desc}"
            raw_features = get_features(full_text, sentence_transformer)
            scaled_features = scaler.transform(raw_features)
            
            # 2. Predict Classification
            pred_class_idx = clf_model.predict(scaled_features)[0]
            
            labels_map = {0: "Easy", 1: "Medium", 2: "Hard"} 
            
            pred_label = labels_map.get(int(pred_class_idx), "Unknown")
            
            # 3. Predict Regression
            pred_score = reg_model.predict(scaled_features)[0]
            # Assuming 1-10 scale based on typical complexity scores
            pred_score = max(0, min(10, pred_score))

            # 4. Display Results
            st.divider()
            
            r_col1, r_col2 = st.columns(2)
            
            with r_col1:
                st.subheader("Classification")
                color = "green" if pred_label == "Easy" else "orange" if pred_label == "Medium" else "red"
                st.markdown(f":{color}[**{pred_label}**]")
            
            with r_col2:
                st.subheader("Regression Score")
                st.metric(label="Difficulty Score", value=f"{pred_score:.2f} / 10")
                st.progress(pred_score / 10)