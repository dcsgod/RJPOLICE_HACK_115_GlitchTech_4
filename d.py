from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os

# Set the path to the Tesseract OCR executable (change this according to your installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_pdf(pdf_path):
    # Convert PDF to images
    images = convert_from_path(pdf_path)

    # Create a folder to store the extracted text
    output_folder = "extracted_text"
    os.makedirs(output_folder, exist_ok=True)

    extracted_text = ""

    for i, image in enumerate(images):
        # Save image temporarily
        image_path = os.path.join(output_folder, f"temp_page_{i}.png")
        image.save(image_path, "PNG")

        # Extract text from the image using Pytesseract
        text = pytesseract.image_to_string(Image.open(image_path))

        # Append the extracted text
        extracted_text += text

        # Remove the temporary image file
        os.remove(image_path)

    return extracted_text

# Example usage:
pdf_path = "FIR-309-2023.pdf"
text_data = extract_text_from_pdf(pdf_path)

# Display the extracted text
print(text_data)
