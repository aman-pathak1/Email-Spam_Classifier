import streamlit as st
import joblib
import string
import spacy
import spacy.cli
import time

st.set_page_config(page_title="Spam Email Detector", page_icon="✉️")

# -------------------- Load spaCy safely --------------------
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# -------------------- Load ML model --------------------
@st.cache_resource(ttl=3600)
def load_model():
    model = joblib.load('checkpoints/spam_detection_model.pkl')
    vectorizer = joblib.load('checkpoints/count_vectorizer.pkl')
    return model, vectorizer

# -------------------- Text preprocessing --------------------
def clean_text(s): 
    for cs in s:
        if cs not in string.ascii_letters:
            s = s.replace(cs, ' ')
    return s.strip()

def remove_little(s): 
    return ' '.join([w for w in s.split() if len(w) > 2])

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc)

def preprocess(text):
    return lemmatize_text(remove_little(clean_text(text)))

def classify_email(model, vectorizer, email):
    return model.predict(vectorizer.transform([email]))

# -------------------- Streamlit UI --------------------
def main():
    st.title("Spam Email Detector")

    output = st.empty()
    status_bar = st.empty()

    user_input = st.text_area(
        "Enter the email text:",
        placeholder="Congratulations!! You have won Rs. 100000.\nClick here to Redeem!!"
    )

    if st.button("Check for Spam"):
        output.empty()
        status_bar.empty()

        if user_input.strip():
            with status_bar.status("Loading the model...", expanded=True) as status:
                model, vectorizer = load_model()
                time.sleep(1)

                status.update(label="Analyzing the email...", state="running")
                processed_text = preprocess(user_input)
                time.sleep(1)

                status.update(label="Checking for Spam...", state="running")
                prediction = classify_email(model, vectorizer, processed_text)
                time.sleep(1)

                status.update(label="Detection Completed!", state="complete", expanded=False)

            status_bar.empty()
            if prediction[0] == 1:
                output.error("🚨 Spam Detected!")
            else:
                output.success("✅ Not Spam")
        else:
            output.warning("Kindly enter the text to detect!")

if __name__ == "__main__":
    main()
