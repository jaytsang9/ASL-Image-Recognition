import streamlit as st
from PIL import Image
from torchvision import transforms
import torch
import os
from neural_net import prediction

from matplotlib import pyplot as plt


st.title('Welcome to ASL Image Recognition!')

file_types = ['jpg', 'jpeg', 'png']

option = st.selectbox(
     'Camera or Upload?',
     ('Choose an Option', 'Camera', 'Upload'))

if option == 'Camera':
    upload = st.camera_input("Take a picture")
elif option == 'Upload':
    upload = st.file_uploader("Choose a file", type=file_types, accept_multiple_files=True)
else:
    upload = None



# upload = st.camera_input("Take a picture")
col1, col2 = st.columns(2)

if upload is not None:
    # To read file as bytes:
    if option == 'Upload':
        for u in upload:
            image = Image.open(u).convert('RGB')
            labels, metrics = prediction(image)
            with col1:
                st.image(image, caption=f'{u.name}', width=256)
            with col2:
                st.header('Top 3 Results')
                for label, metric in zip(labels, metrics):
                    st.metric(label, f'{metric:.2f}%')

    else:
        image = Image.open(upload).convert('RGB')
        labels, metrics = prediction(image) 
        with col1:
            st.image(image, caption=f'{upload.name}', width=256)
        with col2:
            st.header('Top 3 Results')
            for label, metric in zip(labels, metrics):
                st.metric(label, f'{metric:.2f}%')