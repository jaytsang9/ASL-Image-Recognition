import streamlit as st
from PIL import Image
from torchvision import transforms
import torch
import os
from neural_net import prediction

from matplotlib import pyplot as plt


st.title('ASL Alphabet Image Classifier', anchor=None)

tab1, tab2, tab3 = st.tabs(["Camera", "Upload","Learn ASL",])
col1, col2 = st.columns(2)
file_types = ['jpg', 'jpeg', 'png']

with tab1:
    st.header("Take a Picture")
    cam_upload = st.camera_input("Take a picture")
    img = Image.open(cam_upload).convert('RGB')
    labels, metrics = prediction(img) 
    with col1:
        st.image(img, caption=f'{cam_upload.name}', width=256)
    with col2:
        st.header('Top 3 Results')
        for label, metric in zip(labels, metrics):
            st.metric(label, f'{metric:.2f}%')


with tab2:
    st.header("Upload an Image")
    upload = st.file_uploader("Choose a file", type=file_types, accept_multiple_files=True)

    for u in upload:
        img = Image.open(u).convert('RGB')
        labels, metrics = prediction(img)
    with col1:
        st.image(img, caption=f'{u.name}', width=256)
    with col2:
        st.header('Top 3 Results')
        for label, metric in zip(labels, metrics):
            st.metric(label, f'{metric:.2f}%')
        

with tab3:
    st.header("Learn the American Sign Language Alphabet")
    st.image("/images/asl_alphabet.png", width=200)

