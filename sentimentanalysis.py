import streamlit as st
from transformers import pipeline

# Load the sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Streamlit app
def main():
    st.title("🎭 Sentiment Analysis App")

    # User input
    user_input = st.text_input("📝 Enter a sentence:", placeholder="Example: You are an amazing soul")

    if user_input:
           # Perform sentiment analysisS
           result = classifier(user_input)

           # Display results
           st.write("Sentiment:", result[0]['label'])
           st.write("Confidence:", result[0]['score'])

 
# calling main function so that complete code get executed
if __name__ == "__main__":
    main()
