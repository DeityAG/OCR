import streamlit as st
import cv2
import numpy as np
import easyocr
import re
from langdetect import detect_langs
from PIL import Image
import io

def load_easyocr_reader():
    return easyocr.Reader(['hi', 'en'], gpu=False)

def preprocess_image(image):
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated

def perform_easyocr(image, reader):
    preprocessed_image = preprocess_image(image)
    results = reader.readtext(preprocessed_image, paragraph=True, detail=0, 
                              contrast_ths=0.2, adjust_contrast=0.5, 
                              add_margin=0.1, width_ths=0.7, height_ths=0.7)
    extracted_text = ' '.join(results)
    return extracted_text

# ... (keep the detect_languages, fallback_language_check, and highlight_text functions as they are)

def main():
    st.title("OCR for Hindi and English")

    easyocr_reader = load_easyocr_reader()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Perform OCR'):
            with st.spinner('Processing...'):
                extracted_text = perform_easyocr(image, easyocr_reader)
                
                st.subheader("Extracted Text:")
                st.write(extracted_text)
                
                languages_detected = detect_languages(extracted_text)
                if languages_detected:
                    st.write("Detected languages:", ', '.join(languages_detected))
                else:
                    st.write("No languages detected.")

        st.subheader("Search in Extracted Text")
        search_query = st.text_input("Enter keywords to search:")
        if search_query:
            keywords = search_query.split()
            highlighted_text = highlight_text(extracted_text, keywords)
            st.markdown(highlighted_text)

if __name__ == "__main__":
    main()
