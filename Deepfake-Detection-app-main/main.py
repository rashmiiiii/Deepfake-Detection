import streamlit as st
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
from tensorflow.lite.python.interpreter import Interpreter
from PIL import Image
from datetime import datetime
import os

# Load the TFLite model
interpreter = Interpreter(model_path="deepfake_detection_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Function to preprocess video frames
def preprocess_frames(frames):
    preprocessed_frames = []
    for frame in frames:
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preprocessed_frames.append(frame)
    return np.array(preprocessed_frames)

# Function to extract frames from video
def extract_frames(video_bytes, num_frames=10):
    frames = []
    with NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file_path = tmp_file.name
    
    cap = cv2.VideoCapture(tmp_file_path)
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # Delete temporary file
    os.unlink(tmp_file_path)
    
    return frames

# Function to predict if the video is real or deepfake
def predict_video(uploaded_file, status_log):
    video_bytes = uploaded_file.read()  # Read byte data from file object
    frames = extract_frames(video_bytes, num_frames=10)
    preprocessed_frames = preprocess_frames(frames)
    preprocessed_frames = preprocessed_frames.reshape((1, 10, 224, 224, 3))

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frames.astype('float32'))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    accuracy = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    result = "Deepfake" if prediction[0][0] > 0.5 else "Real"

    status_log.text(f"[{timestamp}] - The video is {result} with {accuracy * 100:.2f}% accuracy.\n")

    return result, accuracy

def main():
    st.title("Deepfake Detection App")
    st.image('cover.jpg', use_column_width=True)

    # Section 1: Title and Navigation Bar
    st.sidebar.title("Navigation")
    if st.sidebar.button("Home"):
        page = "Home"
    elif st.sidebar.button("About"):
        page = "About"
    else:
        page = "Home"  # Default to Home page if no button is clicked

    # Section 2: Video and Model Detection
    if page == "Home":
        st.header("Upload a Video for Deepfake Detection")
        uploaded_file = st.file_uploader("Choose a video file (.mp4):", type=["mp4"])

        # Logs Section
        st.subheader("Logs:")
        status_log = st.empty()

        if uploaded_file is not None:
            st.subheader("Uploaded Video")
            st.video(uploaded_file)

            if st.button("Detect Deepfake"):
                with st.spinner("Detecting deepfake..."):
                    result, accuracy = predict_video(uploaded_file, status_log)
                    status_log.success(f"The video is {result} with {accuracy * 100:.2f}% accuracy.")

    # Section 3: Log Section
    elif page == "About":
        st.header("About Deepfake Detection")
        st.write(
            "This is a Streamlit app for detecting deepfake videos using a pre-trained machine learning model. "
            "Deepfakes are synthetic media where a person in a video or image is replaced with someone else's likeness, "
            "usually using deep learning techniques. Deepfake detection aims to identify these manipulated videos "
            "and images to combat misinformation and preserve trust in digital media."
        )
        st.subheader("Model Information")
        st.write(
            "The deepfake detection model used in this app is a machine learning model trained to classify videos "
            "as either real or fake. The model is based on deep neural networks and is capable of detecting subtle "
            "artifacts and inconsistencies introduced by deepfake generation techniques."
        )
        st.subheader("System Logs")
        st.write("System logs will be displayed here.")

if __name__ == "__main__":
    main()
