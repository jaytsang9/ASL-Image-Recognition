import streamlit as st
from PIL import Image
import torchvision
from torchvision import transforms
import torch
import os
from neural_net import prediction
import time
import numpy as np

from matplotlib import pyplot as plt

st.title('ASL Alphabet Image Classifier')

tab_names = ["Take Photo", "Upload", "Learn ASL"]
tab1, tab2, tab3 = st.tabs(tab_names)

file_types = ['jpg', 'jpeg', 'png']

def make_predictions(img):
    labels, metrics = prediction(img) 
    with st.spinner('Predicting...'):
        time.sleep(0.7)
        
        st.header('Top 3 Results')
        for label, metric in zip(labels, metrics):
            st.metric(label, f'{metric:.2f}%')

        st.success('Done!')

def lay_columns(tab):
    upload = None
    if tab == "Take Photo":
        upload = st.camera_input("Take a picture")
    if tab == "Upload":
        upload = st.file_uploader("Choose a file", type=file_types, accept_multiple_files=False)

    if upload is not None:    
        image = Image.open(upload).convert('RGB')
        st.image(image, caption=f'{upload.name}', width=128)
        with st.sidebar:
            make_predictions(image)

with tab1:
    #lay_columns(tab_names[0])
    upload = st.camera_input("Take a picture")
    if upload is not None:    
        image = Image.open(upload).convert('RGB')
        st.image(image, caption=f'{upload.name}', width=128)
        with st.sidebar:
            make_predictions(image)

with tab2:
    #lay_columns(tab_names[1])
    upload = st.file_uploader("Choose a file", type=file_types, accept_multiple_files=False)

    if upload is not None:    
        image = Image.open(upload).convert('RGB')
        st.image(image, caption=f'{upload.name}', width=128)
        with st.sidebar:
            make_predictions(image)

with tab3:
    st.header("Learn the American Sign Language Alphabet")
    st.image("images/asl_alphabet.png", width=700)



