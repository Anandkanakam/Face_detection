import cv2
import streamlit as st
import numpy as np

# Load the cascade
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to capture video frames and detect faces
def capture_and_detect_faces():
    try:
        # Open the webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Failed to open webcam.")
        
        cap.set(3, 1500)  # set the width
        cap.set(4, 1600)  # set the height

        # Placeholder for video frames
        frame_placeholder = st.empty()

        while not st.session_state['stop']:
            ret, img = cap.read()
            if not ret:
                st.warning("Failed to read from the webcam. Please check if the webcam is connected properly.")
                break

            # Flip the image for natural viewing
            img = cv2.flip(img, 1)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = cascade.detectMultiScale(gray, 1.5, 5)

            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Convert the frame to bytes for displaying in Streamlit
            frame_bytes = cv2.imencode('.jpg', img)[1].tobytes()
            frame_placeholder.image(frame_bytes, channels='BGR')

        cap.release()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Main function to run the Streamlit app
def main():
    st.title("Face Detection with OpenCV and Streamlit")

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






