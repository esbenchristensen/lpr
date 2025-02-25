import sys
import pytesseract
from PIL import Image

# If Tesseract is not found in PATH, uncomment and set the correct path:
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def test_tesseract(image_path):
    """
    Opens an image with PIL and runs Tesseract OCR on it.
    Prints the recognized text to confirm Tesseract is working.
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Could not find file '{image_path}'")
        return

    # Perform OCR
    text = pytesseract.image_to_string(img)
    print("Tesseract OCR Result:")
    print("---------------------")
    print(text)

if __name__ == "__main__":
    # Check if user provided an image path as an argument
    if len(sys.argv) < 2:
        print("Usage: python test_tesseract.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    test_tesseract(image_path)