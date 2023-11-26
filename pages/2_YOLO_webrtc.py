import cv2
from ultralytics import YOLO
import av
import supervision as sv
import numpy as np
from streamlit_webrtc import webrtc_streamer
import streamlit as st


st.set_page_config(
    page_title="Object Detection using YOLOv8",  
    page_icon="./images/camera.png",     
    layout="wide",      
    initial_sidebar_state="expanded"    
)

model = YOLO('D:\Perkuliahan\Semester 7\Deep Learning\webapp\webapp2\models\helmetdetection.pt')
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    result = model(img)[0]
    detections = sv.Detections.from_yolov8(result)
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _
        in detections
    ]

    annotated_frame = box_annotator.annotate(
        scene=img, 
        detections=detections,
        labels=labels)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Call webrtc_streamer
webrtc_streamer(key="example", 
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video":True,"audio":False})
