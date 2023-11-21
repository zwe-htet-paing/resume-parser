import os
import time
import math
import json
import sys
import subprocess

import re
import shutil
import string

from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from IPython.display import Image, display

from roboflow import Roboflow
from ultralytics import YOLO

from pdf2image import convert_from_path

# Tesseract OCR
import pytesseract

# Easy OCR
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def get_roboflow_model(api_key="S2g9XxR8N5xMQ5Ntxro8"):
    """
    Get the Roboflow model for resume parsing.
    
    :param api_key: API key of Roboflow account.
    :return: The Roboflow model.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("resume-parsing-odjft")
    model = project.version(3).model
    return model

def load_yolo_model(model_path='./runs/detect/resume-parser2/weights/best.pt'):
    """
    Get the YOLOv8 model for resume parsing.

    :param model_path: Path of pretrained YOLOv8 model.
    :returns: YOLOv8 model.
    """
    model = YOLO(model_path)
    return model

class LibreOfficeError(Exception):
    """
    Custom exception for LibreOffice errors.
    """
    def __init__(self, output):
        self.output = output

def convert_to_pdf(docx_path, output_path, timeout=None):
    """
    Convert a document to PDF using LibreOffice.

    :param output_path: The output output_path for the converted PDF.
    :param docx_path: The path to the docx_path document.
    :param timeout: Timeout for the LibreOffice conversion process.
    :return: The name of the converted PDF file.
    :raises LibreOfficeError: If the conversion fails.
    """
    args = [libreoffice_exec(), '--headless', '--convert-to', 'pdf', '--outdir', output_path, docx_path]

    process = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    filename_match = re.search('-> (.*?) using filter', process.stdout.decode())

    if filename_match is None:
        raise LibreOfficeError(process.stdout.decode())
    else:
        pdf_filename = filename_match.group(1)
        return pdf_filename

def libreoffice_exec():
    """
    Get the path to the LibreOffice executable based on the platform.

    :return: The path to the LibreOffice executable.
    """
    if sys.platform == 'darwin':
        return '/Applications/LibreOffice.app/Contents/MacOS/soffice'
    return 'libreoffice'

def get_image_from_pdf(pdf_path):
    """
    Convert a PDF file to images.
    
    :param pdf_path (str): The path to the input PDF file.
    :return: images list
    """
    # Convert PDF to images
    images = convert_from_path(pdf_path, fmt='JPEG')
    
    # # Remove existing images if they exist and create a new directory
    # if os.path.exists("./pdf2image"):
    #     shutil.rmtree("./pdf2image")
    # os.makedirs("./pdf2image")
        
    # for i in range(len(images)):
    #     images[i].save('./pdf2image/page' + str(i) + '.jpg', 'JPEG')
    
    return images
       
def perform_ocr_roboflow(image, predictions, ocr_engine='easyocr'):
    """
    Perform OCR on the cropped regions of the image based on predictions.

    :param image: PIL Image object.
    :param predictions: List of dictionaries containing object detection results.
    :param ocr_engine: String indicating the OCR engine to use ('tesseract' or 'easyocr').
    :return: A defaultdict(list) with class names and corresponding extracted texts.
    """
    text_dict = defaultdict(list)
    origin_image = image.convert('RGB')

    for item in predictions:
        x, y, width, height = item["x"], item["y"], item["width"], item["height"]
        class_name = item["class"]

        x1, y1, x2, y2 = x - (width / 2), y - (height / 2), x + (width / 2), y + (height / 2)

        # Crop the image
        cropped_image = origin_image.crop((x1, y1, x2, y2))
        
        # Draw bounding box on the original image
        draw = ImageDraw.Draw(origin_image)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        # Specify the font and size
        font = ImageFont.load_default()
        # font = ImageFont.truetype("font.ttf", 20)
        draw.text((x1, y1 - 10), class_name, fill="green", font=font)

        # Convert the image to grayscale (often improves OCR accuracy)
        gray_image = cropped_image.convert("L")

        # Perform text detection using the selected OCR engine
        if ocr_engine == 'tesseract':
            text = pytesseract.image_to_string(gray_image)
        elif ocr_engine == 'easyocr':
            text_results = reader.readtext(np.array(gray_image))
            text = ' '.join([txt[1] for txt in text_results])
        else:
            raise ValueError("Invalid OCR engine. Use 'tesseract' or 'easyocr'.")

        text_dict[class_name].append(text)
        
    return text_dict, origin_image

def perform_ocr_yolo(image, predictions, ocr_engine='easyocr'):
    """
    Perform OCR on the cropped regions of the image based on predictions.

    :param image: PIL Image object.
    :param predictions: List of dictionaries containing object detection results.
    :param ocr_engine: String indicating the OCR engine to use ('tesseract' or 'easyocr').
    :return: A defaultdict(list) with class names and corresponding extracted texts.
    """
    text_dict = defaultdict(list)
    origin_image = image.convert('RGB')

    result = predictions[0]
    
    for box in result.boxes:  
        x1, y1, x2, y2 = [
            round(coord) for coord in box.xyxy[0].tolist()
        ]
        class_name = model.names[box.cls[0].item()]

        # Crop the image using PILLOW
        cropped_image = origin_image.crop((x1, y1, x2, y2))
        
        # Draw bounding box on the original image
        draw = ImageDraw.Draw(origin_image)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        # Specify the font and size
        font = ImageFont.load_default()
        # font = ImageFont.truetype("font.ttf", 20)
        draw.text((x1, y1 - 10), class_name, fill="green", font=font)

        # Convert the image to grayscale (often improves OCR accuracy)
        gray_image = cropped_image.convert("L")

        # Perform text detection using the selected OCR engine
        if ocr_engine == 'tesseract':
            text = pytesseract.image_to_string(gray_image)
        elif ocr_engine == 'easyocr':
            text_results = reader.readtext(np.array(gray_image))
            text = ' '.join([txt[1] for txt in text_results])
        else:
            raise ValueError("Invalid OCR engine. Use 'tesseract' or 'easyocr'.")

        text_dict[class_name].append(text)

    return text_dict, origin_image

def process_images(images, model_type='roboflow', ocr_engine='easyocr'):
    """
    Process a list of images using the Roboflow model and extract text from detected regions.

    :param images: List of PIL Image objects.
    :param model_type: String indicating the Text Detection model to use ('roboflow' or 'yolo').
    :param ocr_engine: String indicating the OCR engine to use ('tesseract' or 'easyocr').
    :return: A list of defaultdict(list) with class names and corresponding extracted texts.
    """
    
    result_texts = []
    result_images = []
    start_time = time.time()
    for image in images:
        # Convert image to numpy array for Roboflow model prediction
        image_array = np.array(image)

        if model_type == 'roboflow':
            # Obtain predictions from the Roboflow model
            predictions = roboflow_model.predict(image_array, confidence=30, overlap=30)

            # Perform OCR and store the results
            text_dict, annotated_image = perform_ocr_roboflow(image, predictions, ocr_engine)
            
        elif model_type == 'yolo':
            # Obtain predictions from the Roboflow model
            predictions = model.predict(image_array, conf=0.20)
            
            # Perform OCR and store the results
            text_dict, annotated_image = perform_ocr_yolo(image, predictions, ocr_engine)
        
        result_texts.append(text_dict)
        result_images.append(annotated_image)
        
    execution_time  = time.time() - start_time
    print(f"Execution time {model_type}: {execution_time:.5f} seconds")

    return result_texts, result_images

if __name__ == "__main__":
    # Main Program
    try:
        # Load models
        model = load_yolo_model()
        roboflow_model = get_roboflow_model()

        # Process resumes
        resume_path = './sample_data/Alice Clark CV.docx'
        fname = resume_path.split('/')[-1].split('.')[0]
        output_path = './sample_data/generated_data/'

        # Chosses model for Detector and OCR
        model_type = 'roboflow'
        ocr_engine="easyocr"

        if resume_path.split('.')[-1] == 'pdf':
            images = get_image_from_pdf(resume_path)
        elif resume_path.split('.')[-1] == 'docx':
            pdf_path = convert_to_pdf(resume_path, output_path)
            images = get_image_from_pdf(pdf_path)
            
        result_texts, result_images = process_images(images, model_type, ocr_engine)
        print(len(result_texts))
        print(result_texts)
        
    except LibreOfficeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
