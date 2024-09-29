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

def detect_languages(text):
    cleaned_text = re.sub(r'[^a-zA-Z\u0900-\u097F\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    if not cleaned_text:
        return []
    try:
        langs = detect_langs(cleaned_text)
        detected = []
        for lang in langs:
            if lang.lang == 'hi' and lang.prob > 0.1:
                detected.append('Hindi')
            elif lang.lang == 'en' and lang.prob > 0.1:
                detected.append('English')
        return detected
    except:
        return fallback_language_check(cleaned_text)

def fallback_language_check(text):
    hindi_range = range(0x0900, 0x097F)
    english_range = range(0x0041, 0x007A)
    
    has_hindi = any(ord(char) in hindi_range for char in text)
    has_english = any(ord(char) in english_range for char in text)
    
    detected = []
    if has_hindi:
        detected.append('Hindi')
    if has_english:
        detected.append('English')
    
    return detected

def highlight_text(text, keywords):
    for keyword in keywords:
        text = text.replace(keyword, f"**{keyword}**")
    return text

def main():
    st.title("OCR for Hindi and English")

    easyocr_reader = load_easyocr_reader()

    # Initialize session state variables
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Perform OCR'):
            with st.spinner('Processing...'):
                st.session_state.extracted_text = perform_easyocr(image, easyocr_reader)
                
                st.subheader("Extracted Text:")
                st.write(st.session_state.extracted_text)
                
                languages_detected = detect_languages(st.session_state.extracted_text)
                if languages_detected:
                    st.write("Detected languages:", ', '.join(languages_detected))
                else:
                    st.write("No languages detected.")

        if st.session_state.extracted_text:  # Check if OCR has been performed
            st.subheader("Search in Extracted Text")
            search_query = st.text_input("Enter keywords to search:", "")
            if search_query:
                keywords = search_query.split()
                highlighted_text = highlight_text(st.session_state.extracted_text, keywords)
                st.markdown(highlighted_text)

if __name__ == "__main__":
    main()

