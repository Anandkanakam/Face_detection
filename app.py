import cv2
import streamlit as st
import numpy as np

# Load the cascade
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to capture video frames and detect faces
def capture_and_detect_faces():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1500)  # set the width
    cap.set(4, 1600)  # set the height

    frame_placeholder = st.empty()

    while not st.session_state['stop']:
        ret, img = cap.read()
        if not ret:
            st.warning("Failed to read from the webcam. Please check if the webcam is connected properly.")
            break

        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.5, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Convert the frame to bytes for displaying in Streamlit
        frame_bytes = cv2.imencode('.jpg', img)[1].tobytes()
        frame_placeholder.image(frame_bytes, channels='BGR')

    cap.release()

# Main function to run the Streamlit app
def main():
    st.title("Face Detection with OpenCV")

    if 'stop' not in st.session_state:
        st.session_state['stop'] = True  # Initialize stop flag as True

    start_button = st.button("Start Camera")
    stop_button = st.button("Stop Camera")

    if start_button:
        st.session_state['stop'] = False
        capture_and_detect_faces()

    if stop_button:
        st.session_state['stop'] = True

if __name__ == "__main__":
    main()





