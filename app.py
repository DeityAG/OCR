import streamlit as st
import cv2
import numpy as np
import easyocr
import re
import torch
from transformers import AutoModel, AutoTokenizer
from langdetect import detect_langs
from PIL import Image
import io

@st.cache_resource
def load_got_model():
    tokenizer = AutoTokenizer.from_pretrained('stepfun-ai/GOT-OCR2_0', trust_remote_code=True)
    model = AutoModel.from_pretrained('stepfun-ai/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='auto', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
    model.eval()
    return model, tokenizer

@st.cache_resource
def load_easyocr_reader():
    reader = easyocr.Reader(['hi', 'en'], gpu=False)  
    return reader

def preprocess_image(image):
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated

def perform_got_ocr(image, model, tokenizer, ocr_type='ocr'):
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        with torch.no_grad():
            res = model.chat(tokenizer, img_byte_arr, ocr_type=ocr_type)
        return res
    except Exception as e:
        st.error(f"Error during GOT OCR: {e}")
        return None

# Rest of the functions remain the same

def main():
    st.title("OCR for Hindi and English")

    got_model, tokenizer = load_got_model()
    easyocr_reader = load_easyocr_reader()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Perform OCR'):
            with st.spinner('Processing...'):
                extracted_text = perform_got_ocr(image, got_model, tokenizer)
                
                if not extracted_text or not any('\u0900' <= char <= '\u097F' for char in extracted_text):
                    st.warning("Falling back to EasyOCR for Hindi text extraction.")
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
