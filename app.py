import streamlit as st
import tensorflow as tf 
import numpy as np 
import streamlit.components.v1 as components
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding',False)

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('./flower_sequential_model_trained.hdf5')
    return model

def predict_class(image,model):
    image=tf.cast(image,tf.float32)
    image=tf.image.resize(image,[180,180])
    image=np.expand_dims(image,axis=0)
    prediction = model.predict(image)
    return prediction

def image_output(test_image):
    st.image(test_image,caption="input Image",width=400)
    pred=predict_class(np.asarray(test_image),model)
    class_names=['daisy','dandelion','rose','sunflower','tulip']
    result=class_names[np.argmax(pred)]
    output='The image is a '+result
    st.success(output)
    col1,col2,col3=st.columns(3)
    with col1:
        google=st.button("check google")
    with col2:
        wiki=st.button('check wiki')
    with col3:
        st.button('clear')
    if google:
        components.iframe(f'https://www.google.com/search?igu=1&ei=&q={"how to plant "+result}',height=1000)
    if wiki:
        components.iframe(f'https://en.wikipedia.org/wiki/{result}',height=1000)
    


model=load_model()
st.title('Flower Classifier')

file=st.file_uploader("upload an image of flower",type=['jpg','png','jpeg'])

if file is None:
    st.text('Waiting for upload....')
    test_image=st.camera_input("Take a picture")
    if test_image:
        test_image=Image.open(test_image)

else:
    slot=st.empty()
    slot.text('running inference')
    test_image=Image.open(file)

if test_image:
    st.write(test_image)
    image_output(test_image)



