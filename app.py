import streamlit as st
import time
import numpy as np
from PIL import Image
from neural_net import prediction

def main():
    st.title('ASL Image Classifier')

    tab_names = ["Take Photo", "Upload Image", "Learn ASL"]
    tab1, tab2, tab3 = st.tabs(tab_names)

    with tab1:
        layout_tabs(tab_names[0])

    with tab2:
        layout_tabs(tab_names[1])

    with tab3:
        st.header("Learn the American Sign Language Alphabet")
        st.image("images/asl_alphabet.png", width=700)


def make_predictions(img):
    labels, metrics = prediction(img) 
    with st.spinner('Analyzing...'):
        time.sleep(0.7)
        
        st.header('Top 3 Results')
        for label, metric in zip(labels, metrics):
            st.metric(label, f'{metric:.2f}%')

        st.success('Done!')

def layout_tabs(tab):
    col1, col2 = st.columns(2)
    file_types = ['jpg', 'jpeg', 'png']
    upload = None
    with st.container():
        if tab == "Take Photo":
            upload = st.camera_input("")
        if tab == "Upload Image":
            upload = st.file_uploader("", type=file_types)   
    if upload is not None:    
        with col1:
            image = Image.open(upload).convert('RGB')
            st.image(image, caption=f'{upload.name}', width=200)
        with col2:
            make_predictions(image)


if __name__ == '__main__':
    main()
