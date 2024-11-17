import pdfplumber
from omegaconf import OmegaConf

def extract_pdf_as_string(file_path):
    with pdfplumber.open(file_path) as pdf:
        full_text = ''.join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return full_text

config = OmegaConf.load("../configs/inference.yml")
pdf_text = extract_pdf_as_string(config.file_path)

print(pdf_text)