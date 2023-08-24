import cv2 as cv
import numpy as np
import streamlit as st 
import PIL

# gray conversion
def convert_to_gray(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray

# Edges detect
def detect_edges(img):
    edges = cv.Canny(img, 100, 200)
    return edges

# Faces detect
def detect_faces(img):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 4)
        
    return img

# Title
st.title("Image processing using OpenCV")
    
# Upload file
upload_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if upload_file is not None:
    # Convert file to cv
    file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, 1)
    
    # Display original image
    st.image(img, channels="BGR", use_column_width=True)
    
    # Gray button
    if st.button('Convert to Grayscale'):
        img_gray = convert_to_gray(img)
        st.image(img_gray, use_column_width=True)
            
    # Edge detection
    if st.button('Edges Detection'):
        img_edges = detect_edges(img)
        st.image(img_edges, use_column_width=True)
             
    # Faces detection
    if st.button('Face detection'):
        img_faces = detect_faces(img)
        st.image(img_faces, use_column_width=True)
