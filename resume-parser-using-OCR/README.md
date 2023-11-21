# Resume Parser using OCR

To extract relavent information from resumes.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Potential Enhancements](#potential-enhancements)
- [Installation](#installation)
- [Conclusion](#conclusion)


## Project Overview

The Resume Parser using OCR project aims to develop a system that automates the extraction of relevant information from resumes in various formats, such as DOCX and PDF. The system leverages object detection models for locating and identifying key sections in a resume, followed by Optical Character Recognition (OCR) to extract text from these sections. The parsed information can be used for tasks such as candidate screening, data entry, and recruitment process automation.

## Key Features

1. Document Conversion:

    Converts different resume formats (DOCX, PDF) into a standardized format for processing.

2. Object Detection:

    Utilizes object detection models (e.g., YOLOv8, Roboflow) to identify key sections in a resume, such as contact information, education, work experience, and skills.

3. Optical Character Recognition (OCR):

    Implements OCR engines (e.g., Tesseract, EasyOCR) to extract text from the identified sections.

4. Modular Architecture:

    Organizes the codebase into modular functions for loading models, converting documents, performing OCR, and processing images. This enhances code readability, maintainability, and extensibility.

5. Error Handling:

    Incorporates robust error handling mechanisms to handle exceptions gracefully and provide meaningful error messages to users.

6. Configuration and Constants:

    Centralizes configuration parameters and constants at the top of the script, making it easy to modify and adapt the system to different environments.

7. Performance Metrics:

    Tracks and displays the execution time of each processing step, helping users assess the system's performance.

8. Support for Multiple OCR Engines:

    Provides flexibility by supporting multiple OCR engines (Tesseract, EasyOCR) and allowing users to choose the engine based on their requirements.

## Usage

1. Users input resumes in DOCX or PDF format.

2. The system converts the documents to a common format (e.g., PDF).

3. Object detection models identify and highlight key sections in the resume.

4. OCR engines extract text from the highlighted sections.

5. The parsed information is stored or further processed as needed.

## Technologies Used

- Programming Language: Python

- Libraries/Frameworks: OpenCV, NumPy, Pillow, Matplotlib, PyTesseract, EasyOCR, PDF2Image, YOLO, Roboflow

- External Services: Roboflow API

## Potential Enhancements

1. Named Entity Recognition (NER):

    - Implement NER techniques to categorize and extract specific entities such as names, dates, and locations.

2. Natural Language Processing (NLP):

    - Apply NLP techniques to analyze the extracted text for sentiment analysis or to identify relevant keywords.

3. Web Interface:

    - Develop a user-friendly web interface for uploading resumes, displaying parsed information, and interacting with the system.

4. Database Integration:

    - Integrate with a database to store and manage parsed information, enabling search and retrieval functionalities.

5. Multi-language Support:

    - Extend the system to support resumes in multiple languages by incorporating language-specific OCR models.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:
```
python inference.py
```

## Conclusion

The Resume Parser project streamlines the process of extracting valuable information from resumes, providing a powerful tool for recruiters, HR professionals, and organizations looking to automate and optimize their hiring processes.
