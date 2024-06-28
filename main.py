import numpy as np
import streamlit as st
from PIL import Image
from object_detection import process_image, annotate_image


def main():
    st.title('Object Detection for Images')
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])

    if file is not None:
        st.image(file, caption='Uploaded Image')
        image = Image.open(file)
        image = np.array(image)
        detections = process_image(image)
        processed_image = annotate_image(image, detections)
        st.image(processed_image, caption='Processed Image')


if __name__ == '__main__':
    main()
