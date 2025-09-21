import streamlit as st
import requests

API_URL = "http://fastapi:8000/predict"


st.title("Object Detection using CNN")
st.write("Upload an image to get predictions from the CNN model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Prepare the file for upload
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

    # Make the request to the API
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        result = response.json()
        st.success("Prediction successful!")
        st.json(result)
    else:
        st.error("Could not connect to the FastAPI server, make sure it's running over port 8000.")