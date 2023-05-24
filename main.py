import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
import streamlit as st
import tempfile

# Create a YOLOv8 object and load the pre-trained weights
yolo = YOLO("best.pt")

def main():
    st.title("YOLOv8 Object Detection")
    
    file_uploaded = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png", "mp4"])

    if file_uploaded is not None:
        # Check the file extension
        file_extension = file_uploaded.name.split(".")[-1]
        if file_extension in ["jpg", "jpeg", "png"]:
            # If the file is an image, perform object detection on the image
            image = Image.open(file_uploaded)
            results = yolo(image, conf=0.25)
            num_objects = len(results[0])
            st.write(f"Number of objects detected: {num_objects}")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        elif file_extension == "mp4":
            # If the file is a video, save it to a temporary location and run object detection on the video
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file_uploaded.read())
                video_path = temp_file.name
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter("output.mp4", fourcc, 25.0, (width, height))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = yolo(frame, save=True)
                num_objects = len(results[0])
                cv2.putText(frame, f"Number of objects detected: {num_objects}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                res_plotted = results[0].plot()
                cv2.imshow("result", res_plotted)
                out.write(res_plotted)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            with open("output.mp4", "rb") as f:
                video_bytes = f.read()
            #st.video(video_bytes)
            st.download_button(
                label="Download output video",
                data=video_bytes,
                file_name="output.mp4",
                mime="video/mp4"
            )

if __name__ == '__main__':
    main()
