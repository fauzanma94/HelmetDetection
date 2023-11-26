import streamlit as st
from ultralytics import YOLO
import cv2
import PIL
import numpy as np
import os

model_path = 'D:\Perkuliahan\Semester 7\Deep Learning\webapp\webapp2\models\helmetdetection.pt'
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="./images/object.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Creating sidebar
with st.sidebar:
    st.header("Image/Video Config")
    source_option = st.radio("Select source", ("Image", "Video"))

    if source_option == "Image":
        source_file = st.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    elif source_option == "Video":
        source_file = st.file_uploader(
            "Choose a video...", type=("mp4", "avi", "mkv"))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("Object Detection using YOLOv8")

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Load YOLO model
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Function to display detected frames
def display_detected_frames(image, confidence, model):
    image = cv2.resize(image, (720, int(720*(9/16))))
    res = model.predict(image, conf=confidence)
    res_plotted = res[0].plot()
    return res_plotted


# Detect objects based on user's choice (Image or Video)
if st.sidebar.button('Detect Objects'):
    if source_option == "Image":
        if source_file:
            uploaded_image = PIL.Image.open(source_file)

            # Display the uploaded image on the left
            col1.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

            with st.spinner('Detecting objects...'):
                # Perform object detection
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

            # Display the detected image on the right
            col2.image(res_plotted, caption='Detected Image', use_column_width=True)
            
            # Display detection results
            with col2.expander("Detection Results"):
                try:
                    for box in boxes:
                        st.write(box.xywh)
                except Exception as ex:
                    st.write("No objects detected.")

    elif source_option == "Video":
        if source_file:
            # Convert the FileUploader object to a readable format for VideoCapture
            file_bytes = source_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)

            # Create a temporary video file
            temp_video_path = "temp_video.mp4"
            with open(temp_video_path, "wb") as temp_file:
                temp_file.write(file_bytes)

            # Open the temporary video file
            cap = cv2.VideoCapture(temp_video_path)
            st_frame = col1.empty()  # Display the video frame on the left

            with st.spinner('Detecting objects...'):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    detected_frame = display_detected_frames(frame, confidence, model)
                    st_frame.image(detected_frame, caption='Detected Video', channels="BGR", use_column_width=True)

                cap.release()

                # Remove the temporary video file
                os.remove(temp_video_path)
