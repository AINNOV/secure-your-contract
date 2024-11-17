import os
from PyPDF2 import PdfReader

def extract_text_after_phrase(pdf_folder, output_folder, phrase):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, file_name)
            txt_file_name = os.path.splitext(file_name)[0] + '.txt'
            txt_path = os.path.join(output_folder, txt_file_name)
            
            try:
                reader = PdfReader(pdf_path)
                text = ""
                phrase_found = False  
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    
                    if phrase in page_text:
                        phrase_found = True

                        text += page_text.split(phrase, 1)[1]
                    elif phrase_found:

                        text += page_text

                if text.strip():  
                    with open(txt_path, 'w', encoding='utf-8') as txt_file:
                        txt_file.write(text)
                    print(f"{file_name} -> {txt_file_name} saved")
                else:
                    print(f"In {file_name}, text after '{phrase}' is not found")
            except Exception as e:
                print(f"{file_name} processing error: {e}")

pdf_folder = "../contracts_SYC" 
output_folder = "../data/pair/prompt"  
phrase = "You get all of the benefits of a lawyer at a fraction of the cost." # wonder.legal/us provides contracts sample with the prefix text ending with this phrase
extract_text_after_phrase(pdf_folder, output_folder, phrase)