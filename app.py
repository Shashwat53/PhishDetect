import streamlit as st
from keras.models import load_model 
import numpy as np 
import os
from streamlit_lottie import st_lottie

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from keras.preprocessing import sequence

class phishing_ai():
    """
    Wrapper class to add custom prediction function to the model.
    """

    def __init__(self, model_name='phishing_detection_model'):
        with open(f'{model_name}_settings.json', "r") as msf:
            self.model_settings = json.load(msf)
        self.char2idx = {u:i for i, u in enumerate(self.model_settings["vocab"])}
        self.model = tf.keras.models.load_model(model_name)

    def text_to_int(self, text):
        return np.array([self.char2idx[c] for c in text])

    def is_phishing(self, url):
        """
        Returns True if phishing, False if not-phishing
        """
        encoded_text = sequence.pad_sequences([self.text_to_int(url)], self.model_settings["max_seq_len"])
        result = self.model.predict(encoded_text)
        return result[0][0] > self.model_settings["threshold"]


st.set_page_config(page_title="Phishing Website Detector", page_icon=":tada:", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style/style.css")

st.title("Welcome to Phishing Website Detection")
st.write(
        "TRUE: Given URL is Phishing"
        )
st.write(        
        "FALSE: Given URL is not Phishing"
    )

url = st.text_input("Enter the URL to Check")

btn = st.button("Check Safety")

if btn:
    #st.text("Safe")
    tf.config.set_visible_devices([], 'GPU')
    phishing_ai = phishing_ai()
    print("Prediction on url:", url, phishing_ai.is_phishing(url))
    st.subheader(phishing_ai.is_phishing(url))

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("PROJECT DETAILS")
        #st.write("##")
        st.write(
            """
            A PROJECT ON PHISHING WEBSITE DETECTION UNDER THE GUIDANCE OF:
            - PROF. CHANDRA MOHAN B
            TEAM MEMBERS:
            - KUSHAJ ARORA 20BCI0047
            - SHASHWAT KUMAR 20BCE2818
            - ANANNYA CHULI 20BCE2046
            """
        )
