import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForTokenClassification, 
    AutoModelForQuestionAnswering, 
    pipeline
)
from rapidfuzz import process, fuzz
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import emoji 
import re
from collections import Counter


st.set_page_config(page_title="Laptop Review Analysis", layout="wide")


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    GPU_NAME = torch.cuda.get_device_name(0)
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    GPU_NAME = "Apple Silicon GPU (MPS)"
else:
    DEVICE = torch.device("cpu")
    GPU_NAME = "CPU (Slow)"

# --- Constants & Mappings ---
SC_MODEL_NAME = "roberta-base"
SC_STATE_DICT_PATH = r"roberta.pth"

AE_MODEL_NAME = "bert-base-uncased"
AE_STATE_DICT_PATH = r"bert_ae_best_model.pth"

QA_MODEL_PATH = r"cause_model" 

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {0: "negative", 1: "neutral", 2: "positive"}
tag2id = {"O": 0, "B-ASP": 1, "I-ASP": 2}
id2tag = {0: "O", 1: "B-ASP", 2: "I-ASP"}

CORE_ASPECTS = {
    "battery life", "battery", "screen", "display", "keyboard", "keys",
    "touchpad", "glass touchpad", "trackpad", "speaker", "speakers",
    "sound", "audio", "sound quality", "audio quality",
    "performance", "speed", "price", "value",
    "quality", "build quality", "fan", "noise", "graphics", "graphics card",
    "wifi", "wireless", "operating system", "system", "os",
    "hard drive", "drive", "storage", "warranty", "service", "warranty service",
    "laptop", "computer", "machine", "unit", "pc", "power", "monitor", 
    "typing", "cost", "money"
}

MANUAL_MAP = {
    "glass touchpad": "touchpad", "trackpad": "touchpad",
    "battery life": "battery", "charging": "battery",
    "screen": "display", "monitor": "display", "lcd": "display", "pixel": "display", "resolution": "display",
    "processor": "performance", "cpu": "performance", "speed": "performance", "boot time": "performance", "lag": "performance",
    "ssd": "storage", "hard drive": "storage", "disk": "storage", "ssd card slot": "storage", "sd card": "storage", "memory": "storage",
    "speakers": "audio", "sound": "audio", "sound quality": "audio", "audio quality": "audio",
    "chassis": "build quality", "case": "build quality", "hinge": "build quality", "quality": "build quality",
    "warranty service": "customer service", "service": "customer service", "support": "customer service",
    "os": "operating system", "system": "operating system", "software": "operating system"
}

TRAIN_ASPECTS_REF = list(CORE_ASPECTS)
GENERIC_STOPWORDS = {"this", "that", "my", "your", "the", "a", "an", "it", "laptop", "computer", "device", "unit", "machine"}

SARCASTIC_OVERRIDE = {
    "🙄": "but it is annoying", 
    "😒": "but it is bad", 
    "🙃": "ironically",
    "🤮": "but it is disgusting",
    "🤢": "but it is bad",
    "👎": "but it is bad",
    "🤡": "but it is ridiculous" # Stronger negative phrasing
}

# --- Model Loading ---
@st.cache_resource
def load_nlp_resources():
    # Now it will just load the model you manually installed
    nlp = spacy.load("en_core_web_sm")
    return nlp

import os

@st.cache_resource
def load_models():
    # 1. Sentiment
    sc_tokenizer = AutoTokenizer.from_pretrained(SC_MODEL_NAME)
    sc_model = AutoModelForSequenceClassification.from_pretrained(SC_MODEL_NAME, num_labels=3)
    if os.path.exists(SC_STATE_DICT_PATH):
        state_dict = torch.load(SC_STATE_DICT_PATH, map_location=torch.device('cpu'))
        sc_model.load_state_dict(state_dict)
    else:
        print(f"Warning: {SC_STATE_DICT_PATH} not found. Using untrained base model.")
    sc_model.to(DEVICE).eval()

    # 2. Aspect
    ae_tokenizer = AutoTokenizer.from_pretrained(AE_MODEL_NAME)
    ae_model = AutoModelForTokenClassification.from_pretrained(AE_MODEL_NAME, num_labels=len(tag2id))
    if os.path.exists(AE_STATE_DICT_PATH):
        ae_state_dict = torch.load(AE_STATE_DICT_PATH, map_location=torch.device('cpu'))
        ae_model.load_state_dict(ae_state_dict)
    else:
        print(f"Warning: {AE_STATE_DICT_PATH} not found. Using untrained base model.")
    ae_model.to(DEVICE).eval()

    # 3. QA
    try:
        qa_pipeline = pipeline("question-answering", model=QA_MODEL_PATH, tokenizer=QA_MODEL_PATH, 
                               device=DEVICE)
    except Exception as e:
        print(f"Warning: {QA_MODEL_PATH} weights missing or invalid. Falling back to default model.")
        qa_pipeline = pipeline("question-answering", device=DEVICE)
        
    return sc_tokenizer, sc_model, ae_tokenizer, ae_model, qa_pipeline

nlp = load_nlp_resources()
sc_tokenizer, sc_model, ae_tokenizer, ae_model, qa_pipeline = load_models()

# --- Helper Functions ---

def process_emojis(text):
    for emo, replacement in SARCASTIC_OVERRIDE.items():
        text = text.replace(emo, f" {replacement} ")
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace("_", " ")
    return text

def get_target_sentence(text, aspect):
    doc = nlp(text)
    potential_terms = [k for k, v in MANUAL_MAP.items() if v == aspect]
    potential_terms.append(aspect)
    for sent in doc.sents:
        sent_lower = sent.text.lower()
        if any(term in sent_lower for term in potential_terms):
            return sent.text
    return text

def extract_aspects_bert_raw(sentence):
    enc = ae_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        logits = ae_model(input_ids=input_ids, attention_mask=attention_mask).logits
    pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()
    tokens = ae_tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    aspects, current = [], []
    for tok, tid in zip(tokens, pred_ids):
        tag = id2tag[tid]
        if tag == "B-ASP":
            if current: aspects.append(current)
            current = [tok]
        elif tag == "I-ASP" and current:
            current.append(tok)
        else:
            if current: aspects.append(current)
            current = []
    if current: aspects.append(current)
    
    phrases = []
    for toks in aspects:
        phrase = ae_tokenizer.convert_tokens_to_string(toks).replace(" ##", "").strip()
        if len(phrase) > 2: phrases.append(phrase)
    return phrases

def extract_canonical_aspects(sentence):
    raw_spans = extract_aspects_bert_raw(sentence)
    cleaned = []
    for p in raw_spans:
        pl = p.lower().strip()
        pl_clean = re.sub(r'[^a-z0-9\s]', '', pl)
        if len(pl_clean.split()) > 2:
            doc = nlp(pl_clean)
            content = [t.text for t in doc if t.text not in GENERIC_STOPWORDS]
            cleaned.append(" ".join(content) if content else pl_clean)
        else:
            cleaned.append(pl_clean)
    return cleaned

def fix_ambiguous_aspects(sentence, aspects):
    fixed = []
    sent_lower = sentence.lower()
    for asp in aspects:
        if asp == "build quality":
            if "sound quality" in sent_lower or "audio quality" in sent_lower:
                fixed.append("audio")
            else:
                fixed.append(asp)
        else:
            fixed.append(asp)
    return list(set(fixed))

def predict_aspect_sentiment_with_conf(full_text, aspect):
    target_sent = get_target_sentence(full_text, aspect)
    text = f"aspect: {aspect} [SEP] sentence: {target_sent}"
    enc = sc_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = sc_model(**enc)
        probs = F.softmax(out.logits, dim=-1)
        conf, pred_id = torch.max(probs, dim=-1)
    return id2label[pred_id.item()], conf.item()

def extract_cause_dl(full_text, aspect, sentiment):
    if sentiment == "neutral": return None
    
    # 1. Get the specific sentence
    target_sent = get_target_sentence(full_text, aspect)
    
    if sentiment == "negative":
        questions = [f"What is wrong with the {aspect}?", f"Why is the {aspect} {sentiment}?"]
    else:
        questions = [f"What is good about the {aspect}?", f"What makes the {aspect} good?"]
    
    best_answer = None
    max_score = 0
    
    for question in questions:
        result = qa_pipeline(question=question, context=target_sent)
        
        if result['score'] > 0.05: # Threshold
            ans = result['answer'].lower().strip()
            
            
            if ans in [aspect.lower(), "quality", "laptop", "bad", "terrible", "good", "issue"]:
                continue

            
            is_contaminated = False
            for other_asp in CORE_ASPECTS:
                
                if other_asp in ans and len(other_asp) > 3:
                    if other_asp not in aspect.lower() and aspect.lower() not in other_asp:
                        is_contaminated = True
                        break
            
            if is_contaminated:
                continue

            
            if result['score'] > max_score:
                max_score = result['score']
                best_answer = result['answer']
                
    return best_answer

def analyze_full_pipeline(text, include_cause=True):
    processed_text = process_emojis(text)
    raw_aspects = extract_canonical_aspects(processed_text)
    
    processed_aspects = []
    for asp in raw_aspects:
        if asp.lower() in MANUAL_MAP:
            processed_aspects.append(MANUAL_MAP[asp.lower()])
            continue
        mapped = process.extractOne(asp, TRAIN_ASPECTS_REF, scorer=fuzz.token_set_ratio)
        mapped_candidate = mapped[0] if mapped else asp
        final_term = MANUAL_MAP.get(mapped_candidate, mapped_candidate)
        
        if (final_term in CORE_ASPECTS) or (final_term in MANUAL_MAP.values()):
            processed_aspects.append(final_term)
        else:
            processed_aspects.append(asp)
            
    final_aspects = fix_ambiguous_aspects(processed_text, processed_aspects)
    final_aspects = list(set(final_aspects))
    
    results = []
    for asp in final_aspects:
        sent, conf = predict_aspect_sentiment_with_conf(processed_text, asp)
        if include_cause:
            cause = extract_cause_dl(processed_text, asp, sent)
        else:
            cause = None
        
        results.append({
            "Aspect": asp.title(),
            "Sentiment": sent,
            "Confidence": round(conf, 4),
            "Cause": cause if cause else "N/A"
        })
    return results

# --- DATA PROCESSING ---

def process_batch_reviews(df, product_name):
    processed_rows = []
    cols = {c.lower(): c for c in df.columns}
    
    if 'review' not in cols or 'rating' not in cols:
        return None, "Error: CSV must contain 'review' and 'rating' columns."

    total = len(df)
    bar = st.progress(0)
    
    for i, row in df.iterrows():
        text = str(row[cols['review']])
        try: rating = float(row[cols['rating']])
        except: rating = 3.0
            
        analysis = analyze_full_pipeline(text, include_cause=False)
        
        for res in analysis:
            processed_rows.append({
                "Product": product_name,
                "Original_Text": text, 
                "Aspect": res['Aspect'].upper(),
                "Sentiment": res['Sentiment'],
                "Rating": rating
            })
        bar.progress(min((i + 1) / total, 1.0))
    
    bar.empty()
    return pd.DataFrame(processed_rows), None

# --- DETAILED COMPARISON REPORT ---

import plotly.graph_objects as go # <--- Add this import at the top

# --- NEW: Persona Weighting Logic ---
PERSONA_WEIGHTS = {
    "General Use": {}, # Default (all 1.0)
    "Student / Office": {
        "BATTERY": 2.0, "PRICE": 1.5, "KEYBOARD": 1.5, "PORTABILITY": 1.5,
        "PERFORMANCE": 0.8, "GAMING": 0.5
    },
    "Gamer": {
        "PERFORMANCE": 2.0, "GRAPHICS": 2.0, "DISPLAY": 1.5, "COOLING": 1.5,
        "BATTERY": 0.5, "PORTABILITY": 0.5
    },
    "Content Creator": {
        "DISPLAY": 2.0, "STORAGE": 1.5, "PERFORMANCE": 1.5,
        "AUDIO": 1.2
    }
}

def calculate_weighted_score(df, persona):
    """Calculates a single 0-100 score for a laptop based on the persona."""
    weights = PERSONA_WEIGHTS.get(persona, {})
    
    # Get mean NSS per aspect
    stats = df.groupby('Aspect')['Sentiment'].value_counts().unstack(fill_value=0)
    for c in ['positive', 'negative', 'neutral']:
        if c not in stats.columns: stats[c] = 0
    stats['Total'] = stats.sum(axis=1)
    stats['NSS'] = (stats['positive'] - stats['negative']) / stats['Total']
    
    total_weight = 0
    weighted_sum = 0
    
    for aspect, row in stats.iterrows():
        nss = row['NSS'] # Range -1 to 1
        
        # Apply specific weight if exists, else default to 1.0
        # We search broadly (e.g., 'BATTERY LIFE' matches 'BATTERY')
        w = 1.0
        for key, val in weights.items():
            if key in aspect: 
                w = val
                break
        
        # Normalize NSS (-1 to 1) -> (0 to 100)
        score_0_100 = (nss + 1) * 50 
        
        weighted_sum += score_0_100 * w
        total_weight += w
        
    if total_weight == 0: return 50
    return weighted_sum / total_weight

def plot_radar_chart(df, p1, p2):
    """Creates a professional Radar Chart using Plotly."""
    
    # Get Top 5 shared aspects for the chart
    counts = df['Aspect'].value_counts()
    top_aspects = counts.head(6).index.tolist()
    
    # Calculate NSS for these aspects
    def get_scores(product_name):
        scores = []
        for asp in top_aspects:
            subset = df[(df['Product'] == product_name) & (df['Aspect'] == asp)]
            if subset.empty:
                scores.append(50) # Neutral baseline
            else:
                pos = len(subset[subset['Sentiment']=='positive'])
                neg = len(subset[subset['Sentiment']=='negative'])
                total = pos + neg + len(subset[subset['Sentiment']=='neutral'])
                if total == 0: nss = 0
                else: nss = (pos - neg) / total
                scores.append((nss + 1) * 50) # Convert to 0-100 scale
        return scores

    vals1 = get_scores(p1)
    vals2 = get_scores(p2)
    
    # Close the loop for radar chart
    top_aspects.append(top_aspects[0])
    vals1.append(vals1[0])
    vals2.append(vals2[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals1, theta=top_aspects, fill='toself', name=p1, line_color='blue'))
    fig.add_trace(go.Scatterpolar(r=vals2, theta=top_aspects, fill='toself', name=p2, line_color='red'))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Aspect Performance Map (0-100)"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- UPDATED REPORT FUNCTION ---

def generate_professional_report(df, persona):
    products = df['Product'].unique()
    if len(products) != 2: st.warning("Need exactly 2 products."); return
    p1, p2 = products[0], products[1]
    
    st.divider()
    
    # 1. THE SCORECARD (Big Numbers)
    score1 = calculate_weighted_score(df[df['Product'] == p1], persona)
    score2 = calculate_weighted_score(df[df['Product'] == p2], persona)
    
    diff = score1 - score2
    winner = p1 if score1 > score2 else p2
    
    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        st.metric(label=f"{p1} Score", value=f"{score1:.1f}/100", delta=f"{diff:.1f}")
    with colB:
        st.metric(label=f"{p2} Score", value=f"{score2:.1f}/100", delta=f"{-diff:.1f}")
    with colC:
        if abs(diff) < 2:
            st.info(f"**Verdict:** It's a Tie! Both are great for {persona}s.")
        else:
            st.success(f"**Verdict:** The **{winner}** is the better choice for **{persona}s**.")

    # 2. RADAR CHART
    st.subheader(" Performance Radar")
    plot_radar_chart(df, p1, p2)
    
    # 3. PROS & CONS (With Grade Levels)
    # We use a helper to get grade from score
    def get_grade(nss):
        if nss >= 0.7: return "A+ (Excellent)"
        if nss >= 0.4: return "A (Good)"
        if nss >= 0.1: return "B (Okay)"
        if nss >= -0.2: return "C (Average)"
        return "F (Poor)"

    st.subheader(" Detailed Report Card")
    
    # Get stats again
    stats = df.groupby(['Product', 'Aspect'])['Sentiment'].value_counts().unstack(fill_value=0)
    for c in ['positive', 'negative', 'neutral']: 
        if c not in stats.columns: stats[c] = 0
    stats['Total'] = stats.sum(axis=1)
    stats['NSS'] = (stats['positive'] - stats['negative']) / stats['Total']
    stats = stats.reset_index()
    
    # Filter for table
    sig = stats[stats['Total'] >= 2]
    common = set(sig[sig['Product']==p1]['Aspect']).intersection(set(sig[sig['Product']==p2]['Aspect']))
    
    if common:
        data = []
        for asp in common:
            r1 = sig[(sig['Product']==p1) & (sig['Aspect']==asp)].iloc[0]
            r2 = sig[(sig['Product']==p2) & (sig['Aspect']==asp)].iloc[0]
            
            # Winner Logic
            if r1['NSS'] > r2['NSS'] + 0.1: win = p1
            elif r2['NSS'] > r1['NSS'] + 0.1: win = p2
            else: win = "Draw"
            
            data.append({
                "Feature": asp,
                f"{p1} Grade": get_grade(r1['NSS']),
                f"{p2} Grade": get_grade(r2['NSS']),
                "Winner": win
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True)

    # 4. DEAL BREAKER (The Lazy QA Part)
    # Only run this on the loser of the weighted score
    loser = p2 if score1 > score2 else p1
    
    st.subheader(f" Potential Deal Breakers for {loser}")
    fail = sig[(sig['Product']==loser) & (sig['NSS'] < 0)].sort_values('NSS').head(1)
    
    if not fail.empty:
        bad_aspect = fail.iloc[0]['Aspect']
        st.error(f"Users really disliked the **{bad_aspect}** on the {loser}.")
        
        # Lazy QA Trigger
        with st.spinner("Analyzing complaints..."):
            complaints = df[(df['Product']==loser) & (df['Aspect']==bad_aspect) & (df['Sentiment']=='negative')]
            causes = []
            for txt in complaints['Original_Text'].head(5):
                c = extract_cause_dl(txt, bad_aspect, "negative")
                if c and len(c)>3: causes.append(c)
            
            if causes:
                top_cause = Counter(causes).most_common(1)[0][0]
                st.write(f"**Why?** Most users mentioned: *\"{top_cause}\"*")
            else:
                st.write("**Why?** General dissatisfaction found in reviews.")
    else:
        st.write(f"Surprisingly, no major deal breakers found for {loser}!")

# --- PAGES ---

def page_eda_metrics():
    st.title(" EDA & Metrics")
     
    
    st.subheader("1. Handling Class Imbalance")
    st.markdown("""
    **The Challenge:** Real-world review data is heavily skewed. Most users leave positive reviews, 
    while neutral reviews are very rare. This causes models to ignore the neutral class.
    
    **The Solution:** We applied data augmentation strategies to the training set to balance the classes, 
    ensuring the model learns to identify 'Neutral' sentiments accurately.
    """)
    
    
    comparison_df = pd.DataFrame({
        "Sentiment": ["Negative", "Neutral", "Positive"] * 2,
        "Count": [866, 460, 789,   # Approximate 'Original' (Imbalanced)
                  789, 789, 789], # 'Balanced' (Training)
        "Dataset": ["Original (Raw)", "Original (Raw)", "Original (Raw)",
                    "Training (Balanced)", "Training (Balanced)", "Training (Balanced)"]
    })
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.caption("Class Distribution Strategy")
        st.dataframe(
            pd.DataFrame({
                "Class": ["Negative", "Neutral", "Positive"],
                "Original Count": ["866", "460", "789"],
                "Training Count": ["789", "789", "789"]
            }), 
            hide_index=True
        )
        
    with col2:
        # Side-by-Side Bar Chart
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=comparison_df, x="Sentiment", y="Count", hue="Dataset", palette="viridis", ax=ax)
        ax.set_title("Impact of Balancing Strategy")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), )
        st.pyplot(fig)

    st.divider()
    
    st.subheader("2. Model Performance (Test Set)")
    st.markdown("Metrics calculated on **held-out test data** (Unseen by the model).")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Sentiment Model (RoBERTa)")
        # Ensure these metrics are from your TEST set, not training set
        st.dataframe(pd.DataFrame({
            "Class": ["Negative", "Neutral", "Positive"], 
            "Precision": [0.86, 0.76, 0.88],
            "Recall": [0.91, 0.70, 0.87], 
            "F1-Score": [0.88, 0.73, 0.88]
        }), hide_index=True)
        st.metric("Overall Weighted F1", "0.85")
    with c2:
        st.markdown("###  Aspect Extraction (BERT)")
        metrics_ae = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1-Score"],
            "Value": [0.72, 0.84, 0.78]
        })
        st.dataframe(metrics_ae, hide_index=True)
        st.metric("Overall F1 Score", "0.78")

def page_inference():
    st.title(" Single Review Analysis")
     
    
    with st.form("inference"):
        text = st.text_area("Review:", "I love the battery 🙄. The resolution is fantastic but the SSD is slow.")
        if st.form_submit_button("Analyze"):
            with st.spinner("Processing..."):
                results = analyze_full_pipeline(text, include_cause=True)
                if not results:
                    st.warning("No aspects found.")
                else:
                    df = pd.DataFrame(results)
                    st.subheader("Results")
                    def style(v): 
                        c = '#d4edda' if v == 'positive' else '#f8d7da' if v == 'negative' else '#fff3cd'
                        return f'background-color: {c}; color: black'
                    st.dataframe(df.style.map(style, subset=['Sentiment']).format({"Confidence": "{:.2%}"}), use_container_width=True)
                    
                    fig, ax = plt.subplots(figsize=(6, 2))
                    sns.countplot(y="Sentiment", data=df, palette="coolwarm", order=['positive', 'neutral', 'negative'], ax=ax)
                    st.pyplot(fig)

def page_comparison():
    st.title("Laptop Comparison")
    st.markdown("Upload reviews to see a Head-to-Head battle.")
    
    # Persona Selector in Main Area
    persona = st.selectbox(" Select User Persona", list(PERSONA_WEIGHTS.keys()))
    
    col1, col2 = st.columns(2)
    with col1:
        name1 = st.text_input("Laptop A", "Laptop A")
        file1 = st.file_uploader("Reviews A", type=["csv"], key="f1")
    with col2:
        name2 = st.text_input("Laptop B", "Laptop B")
        file2 = st.file_uploader("Reviews B", type=["csv"], key="f2")
        
    if st.button(" Run Analysis") and file1 and file2:
        try:
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
            
            st.write("Reading reviews...")
            # Use include_cause=False for speed
            r1, _ = process_batch_reviews(df1, name1)
            r2, _ = process_batch_reviews(df2, name2)
            
            full = pd.concat([r1, r2], ignore_index=True)
            
            generate_professional_report(full, persona)
            
        except Exception as e:
            st.error(f"Error: {e}")


st.sidebar.title("ABSA Dashboard")


page = st.sidebar.radio("Go To:", ["EDA & Metrics", "Single Inference", "Comparison"])

if page == "EDA & Metrics": page_eda_metrics()
elif page == "Single Inference": page_inference()
else: page_comparison()