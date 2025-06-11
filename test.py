import fitz  # PyMuPDF
import os 

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

folder_path = "/Users/christopherperezlebron/Documents/Resumes"
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        print(f"Extracting from: {filename}")
        text = extract_text_from_pdf(file_path)
        # Do something with the text here - for example, print or save it
        print(text)  # Print first 500 characters as a sample
      