import streamlit as st
from model1utils import predict_model_1
from model2utils import predict_model_2
from spacyutils import predict_spacy
from stanzautils import predict_stanza

# Streamlit UI Setup
st.set_page_config(page_title="NER Models", layout="wide")

# Set a background image or color
st.markdown("""
    <style>
    .reportview-container {
        background: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)

# Page title with some style
st.title("Named Entity Recognition (NER) Models")
st.markdown("<h5>Predict named entities from various NER models</h5>", unsafe_allow_html=True)

# Input text area for sentence input
st.markdown("<br>", unsafe_allow_html=True)
sentence_input = st.text_area("Enter a text to analyze:", height=100)


# If the user clicks the "Predict" button
if st.button("Predict"):
    if sentence_input:
        # Using columns to display predictions side by side
        col1, col2, col3,col4 = st.columns(4)
        
        with col1:
            st.subheader("Model 1 Predictions")
            predictions_1 = predict_model_1(sentence_input)
            st.write(predictions_1)

        with col2:
            st.subheader("Model 2 Predictions")
            predictions_2 = predict_model_2(sentence_input)
            st.write(predictions_2)

        with col3:
            st.subheader("Spacy Model Predictions")
            predictions_spacy = predict_spacy(sentence_input)
            st.write(predictions_spacy)

        with col4:
            st.subheader("Stanza Model Predictions")
            predictions_stanza = predict_stanza(sentence_input)
            st.write(predictions_stanza)

    else:
        # Warning message when no sentence is entered
        st.warning("Please enter a sentence to analyze.", icon="⚠️")

# Add some padding and footer for a nice touch
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h5 style='color:#555555;'>NER Model Comparison</h5>", unsafe_allow_html=True)
st.markdown("Compare the predictions from different NER models and explore how each one identifies entities.", unsafe_allow_html=True)

# You can also include a footer with a custom message or link to the source code
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Made by [ZAALI Mohamed](https://www.linkedin.com/in/m-zaali/) and [SEKAL Douaâ](https://www.linkedin.com/in/douaa-sekal/)", unsafe_allow_html=True)
