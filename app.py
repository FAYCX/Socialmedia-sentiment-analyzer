import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

alt.themes.enable("dark")

st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown('<style>div.block-container{padding-top:0.5rem;}</style>', unsafe_allow_html=True)

white_color = "#fff"
h1 = "1.8rem"
h2 = "1.5rem" 
h3 = "1.1rem" 
p = "0.9rem"

font_css = f"""
<style>
    body {{
        margin: 0;
        padding: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }}
    .title-box {{
        padding: 10px;
        margin: 10px;
        background-color: #333;
        color: {white_color};
        border-radius: 10px;
        box-shadow: 0.4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
    }}
    h1 {{
        font-family: 'Helvetica Neue', Arial, sans-serif !important;
        font-size: {h1};
        font-weight: bold;
        font-stretch: condensed;
        margin: 0;
        letter-spacing: 0.08rem;
    }}
    h2 {{
        font-family: 'Helvetica Neue', Arial, sans-serif !important;
        font-size: {h2};
        font-weight: bold;
        font-stretch: condensed;
        letter-spacing: 0.02rem;
    }}
    h3 {{
        font-family: 'Helvetica Neue', Arial, sans-serif !important;
        font-size: {h3};
        font-weight: bold;
        font-stretch: condensed;
        letter-spacing: 0.01rem;
        color: #edcce8;
    }}
    h5, p {{
        font-family: 'Helvetica Neue', Arial, sans-serif !important;
        color: #85888c;
        font-size: {p};
    }}
    @media (max-width: 480px) {{
        .title-box {{
            padding: 10px;
            margin: 10px;
        }}
        h1 {{
            font-size: 1.4rem;
        }}
    }}
    h2 {{
        font-family: 'Helvetica Neue', Arial, sans-serif !important;
        font-size: {h2};
        font-weight: bold;
        font-stretch: condensed;
        letter-spacing: 0.02rem;
    }}
    h3 {{
        font-family: 'Helvetica Neue', Arial, sans-serif !important;
        font-size: {h3};
        font-weight: bold;
        font-stretch: condensed;
        letter-spacing: 0.01rem;
    }}
    h5, p {{
        font-family: 'Helvetica Neue', Arial, sans-serif !important;
        color: #dbdbdb;
        font-size: {p};
    }}
</style>
"""

st.markdown(font_css, unsafe_allow_html=True)

# Load your trained model
pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_Mar29_2024.pkl", "rb"))

# Prediction Function
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]  # Return the first and only item

# Get Prediction Probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Emoji Dictionary for Emotions
emotions_emoji_dict = {"anger": "😤", "disgust": "🤢", "fear": "😨", "joy": "😸", "surprise": "😻", "neutral": "😶", "sadness": "😭", "shame": "🫣"}

def main():
    st.title("Social Media Sentiment Analyzer AI-Bot")
    st.subheader("Collection of ML Projects Created by [Fay Cai](https://www.faycai.com)")
    st.write("🤖:'Trying my best to understand human emotion - [Info about my Training Data](https://www.faycai.com/data-science/the-mosaic-mind-of-ai-app)'")

    for _ in range(3):
        st.write("")

    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area("Please Type Any Text Here")
        submit_text = st.form_submit_button(label='submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Your Text")
            st.write(raw_text)

            st.success("My Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "😶")  # Safely get the emoji, default to neutral if not found
            st.write(f"{prediction} {emoji_icon}")
            st.write("My Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
