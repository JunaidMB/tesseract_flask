from PIL import Image
import pytesseract

# Configuration of Tesseract
pytesseract.pytesseract.tesseract = './.virtualenvs/safa_prototype/lib/python3.8/site-packages'

def ocr_core(filename):
    """
    This function will handle the core OCR processing
    """

    text = pytesseract.image_to_string(Image.open(filename))

    return text

#print(ocr_core("demo.png"))
