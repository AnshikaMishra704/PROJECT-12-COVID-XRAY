import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(
    page_title="COVID-19 Chest X-Ray Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 COVID-19 Detection from Chest X-Ray")
st.write("Upload a Chest X-Ray image to predict whether it is COVID or NORMAL.")

@st.cache_resource
def load_my_model():
    return load_model("model.h5")

model = load_my_model()

uploaded_file = st.file_uploader(
    "Choose a Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((299, 299))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    threshold = 0.5

    if prediction[0][0] > threshold:
        label = "COVID"
    else:
        label = "NORMAL"

    st.subheader("Prediction")
    st.success(label)

    st.write("Raw Model Output:")
    st.write(prediction)
