import streamlit as st
from PIL import Image
from torchvision import transforms
import torch
import os
from neural_net import prediction
import time

from matplotlib import pyplot as plt

def make_predictions():
    with st.spinner('Wait for it...'):
        time.sleep(1)
    
    st.header('Top 3 Results')
    for label, metric in zip(labels, metrics):
        st.metric(label, f'{metric:.2f}%')

    st.success('Done!')

    


st.title('ASL Alphabet Image Classifier', anchor=None)

tab1, tab2 = st.tabs(["Upload","Learn ASL",])
col1, col2 = st.columns(2)
file_types = ['jpg', 'jpeg', 'png']

with tab1:
    option = st.selectbox('Choose an Option:', ('Camera', 'Upload'))

    with col1:
        if option == 'Camera':
            upload = st.camera_input("Take a picture")
        elif option == 'Upload':
            upload = st.file_uploader("Choose a file", type=file_types, accept_multiple_files=True)
        else:
            upload = None

        if option == 'Upload' and upload is not None:
            for u in upload:
                image = Image.open(u).convert('RGB')
                labels, metrics = prediction(image)
                with col1:
                    st.image(image, caption=f'{u.name}', width=256)
                with col2:
                    '''
                    st.header('Top 3 Results')
                    for label, metric in zip(labels, metrics):
                        st.metric(label, f'{metric:.2f}%')
                    '''
                    make_predictions()

        if option == 'Camera' and upload is not None:    
            image = Image.open(upload).convert('RGB')
            labels, metrics = prediction(image) 
            with col2:
                make_predictions()


with tab2:
    st.header("Learn the American Sign Language Alphabet")
    st.image("images/asl_alphabet.png", width=700)