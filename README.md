# OCR Web Application with Keyword Search

This web application allows users to upload images containing text in both Hindi and English, extract text using Optical Character Recognition (OCR), and perform keyword searches on the extracted text. It is built with Streamlit, utilizes the EasyOCR library for text recognition, and is deployed on Hugging Face Spaces.

[Live Demo](https://huggingface.co/spaces/Adignite/OCR) <!-- Add your live demo link here -->
[Drive with demo and Zip File](https://drive.google.com/drive/folders/1ltHyPLnd1xyn76YMsvl9t8iI0XliOQK9)

## Features

- **Multi-language OCR**: Supports text extraction from images containing Hindi and English.
- **Keyword Search**: Allows users to search for keywords within the extracted text.
- **Language Detection**: Uses langdetect to identify Hindi and English in the extracted text.
- **Streamlit Web App**: Simple and interactive web interface for uploading images and performing OCR.

## Additional Implementations

I have also implemented the GOT-OCR 2.0 model (stepfun-ai/GOT-OCR2_0) alongside EasyOCR. However, the GOT model consistently crashes in the Streamlit app, so it couldn't be deployed. You can check out the implementation here:

- [GOT-EasyOCR on GitHub](https://github.com/DeityAG/OCR/blob/main/got-easyocr.ipynb) <!-- Add your GitHub link here -->
- [GOT-EasyOCR on Kaggle](https://www.kaggle.com/code/adignite/got-easyocr) <!-- Add your Kaggle link here -->

The EasyOCR is used because the GOT model was not very good at extracting Hindi Texts from the images, but the EasyOCR library does a much better job.

## Setup Instructions

### 1. Clone the Repository

To run this project locally, clone the repository:

```bash
git clone https://github.com/DeityAG/OCR.git
cd OCR
```

### 2. Set Up the Environment

Make sure you have Python 3.x installed. You can set up the required environment using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key Dependencies:
- streamlit (for web interface)
- easyocr (for OCR processing)
- opencv-python-headless (for image preprocessing)
- langdetect (for detecting the language in the extracted text)

Alternatively, you can install the required libraries manually:

```bash
pip install streamlit easyocr opencv-python-headless numpy langdetect Pillow
```

### 3. Running the Application Locally

After setting up the environment, you can run the Streamlit app locally using:

```bash
streamlit run app.py
```

Open the link provided by Streamlit in your browser, and you can start uploading images to perform OCR.

## Deployment

The web app is currently deployed on Hugging Face Spaces. If you'd like to deploy it yourself on platforms like Streamlit Sharing or Hugging Face, follow these steps:

### Hugging Face Deployment:

1. Create a space on Hugging Face Spaces.
2. Upload your project files to the space.
3. Ensure you have a `requirements.txt` file to install dependencies automatically.

### Streamlit Sharing Deployment:

1. Push your code to a GitHub repository.
2. Visit Streamlit Sharing.
3. Deploy your app by linking it to your GitHub repository.

## Files

- `app.py`: The main script that handles image upload, OCR processing, language detection, and keyword search.
- `requirements.txt`: Lists all the dependencies needed to run the project.
- `got-easyocr.ipynb`: Jupyter notebook containing the implementation of the GOT-OCR model, available on GitHub and Kaggle.

## License

This project is licensed under the MIT License.
