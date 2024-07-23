import sys
import subprocess
import os

# Print Python executable path
#print("Python executable:", sys.executable)

# List installed packages
#subprocess.run(["pip", "list"])


#print("Current working directory:", os.getcwd())
#print("Files in the current directory:", os.listdir())

import streamlit as st

st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)


import pandas as pd
import numpy as np
import joblib
import altair as alt

alt.themes.enable("dark")

st.markdown('<style>div.block-container{padding-top:0.5rem;}</style>',unsafe_allow_html=True)

white_color = "#fff"
h1 = "1.8rem"
h2 = "1.5rem" 
h3 = "1.1rem" 
p = "0.9rem"

font_css = f"""
<style>
    /* Base styles */
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
        background-color: #333;  /* Dark background for the title box */
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

    /* Adjustments for smaller screens */
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


# def get_image_as_base64(path):
#     """ Convert image file to base64 string """
#     with open(path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode()

# # Load the image file and convert to base64
# logo_base64 = get_image_as_base64("AI_logo.png")

# # Define custom HTML to display the image
# logo_html = f"""
# <style>
# #logo {{
#     position: absolute;
#     top: 1vw;
#     right:0;
#     height:5w;
#     max-height:150px;
#     width:auto;
#     z-index: 999;
#     transition: top 0.3s;
# }}

# @media (max-width:768px){{
#     #logo {{
#     top: 0.5vw;
#     height:8vw;

#     }}
# }}
# </style>
# <img id="logo" src="data:image/png;base64,{logo_base64}" alt="App logo" style="height: 150px; width: auto;">
# """

# # Display the logo
# st.markdown(logo_html, unsafe_allow_html=True)



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
emotions_emoji_dict = {"anger": "😤", "disgust": "🤢", "fear": "😨", "joy": "😸", "surprise": "😻", "neutral": "😶", "sadness": "😭", "shame":"🫣"}




# Sidebar
with st.sidebar:
    st.header('Distant Reading with AI Mind')
    st.sidebar.write('''
    ---
    Collection of ML Projects Created 
    with ❤ by [Fay Cai](https://www.faycai.com)
    ''')
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")


    #st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    menu = ["Sentiment Analyzer", "About"]
    choice = st.sidebar.selectbox("Sentiment Analyzer", menu)

    #create_choice = st.sidebar.selectbox("Generate", create)
    st.markdown(" ")


  

def main():
    
    if choice == "Sentiment Analyzer":
        st.title("Social Media Sentiment Analyzer AI-Bot")
        st.write("🤖:'Trying my best to understand human emotion'")


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
                # You might want to improve this to display it better
                #st.write(probability)
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                #st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                st.altair_chart(fig,use_container_width=True)
        


    else:
        st.title("About")
        st.subheader("About the Data")
        #annotation
        st.write('''
        ---
        This emotion analyzer ML model is trained on `2 MB of real user-generated posts on Twitter platform`, using `logistic regression` algorithms. The model achieves an overall accuracy of 0.8, however, after multiple tests, we found the model exhibits `strong biases`: for instance, the word 'queen' and 'woman' as categorized as joy, while 'man' and 'king' are associated with anger and sadness. This highlights a significant bias presented in the selected posts from the Twitter platform. For more info, please check: [The Mosaic Mind of AI](https://www.faycai.com/data-science/the-mosaic-mind-of-ai-app)
        ''')

        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")

        st.subheader("About the Creator")
        #annotation
        st.write('''
        ---
        [Fay Cai](https://www.faycai.com) is a product innovator armed with an interdisciplinary skill set in Interactive Technology, Data Science, Product Design, complemented by seasoned professional experience as a Brand Marketing Manager, and advanced studies in Cognitive Science. Through advanced studies and research, she uncovers profound insights into both human and artificial intelligence, transcending unconventional awareness to enhance product design. She envision a sustainable future where AI augments and enhances human capabilities, ultimately improving our overall well-being.
        ''')






if __name__ == '__main__':
    main()