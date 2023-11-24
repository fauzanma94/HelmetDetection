import streamlit as st 

st.set_page_config(page_title="Home",
                   layout='wide',
                   page_icon='./images/home.png')

st.title("YOLO V8 Object Detection App")
st.caption('This web application demostrate Detection of Riders Without Helmets and Object Counting Using YOLOv8')

# Content
st.markdown("""
### This App detects objects from Images
- Automatically detects 2 Object From Image
- [Click here for App](/YOLO_for_image/)  

Below give are the object the our model will detect
1. Berhelm : Riders with Helmet
2. TidakBerhelm : Riders without Helmet
        
            """)