import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# UI
st.title(" Sentiment Analysis App")

user_input = st.text_area("Enter your text here:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        pred_class = torch.argmax(probs, dim=1).item()

        label = model.config.id2label[pred_class]
        score = probs[0][pred_class].item()

        st.subheader("Result:")
        st.write(f"Label: {label}")
        st.write(f"Confidence: {score:.4f}")