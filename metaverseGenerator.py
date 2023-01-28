import fitz
from contextSync import ContextSync
import io
import pdfplumber
from PyPDF2 import PdfReader

class MetaverseGenerator():
    def __init__(self, pdf_url):
        self.pdf = fitz.open(pdf_url) #  pdfreader(pdf) 
        self.sync = ContextSync()
        self.pdf_url = pdf_url
        #self.remote_file = urlopen(pdf_url).read()
        #self.memory_file = io.BytesIO(self.remote_file)
        #self.pdf = pdftotext.PDF(self.memory_file)

    # Extract the text per page
    def extract_text(self):
        #with pdfplumber.open(r'C:\Users\Asus ZenBook\Desktop\UCL\One day long ago.pdf') as pdf:
        #    page = pdf.pages[0]
        #    text_in_page = page.extract_text()

        #with fitz.open(r"C:\Users\Asus ZenBook\Desktop\DISSERTATION\Alice in Wonderland.pdf") as doc:
        #    for page in doc:
        #        text= page.get_text()
        #        print(text)
        #        self.sync.contextSync(text)

        reader = PdfReader(r'C:\Users\Asus ZenBook\Desktop\DISSERTATION\Alice in Wonderland.pdf')
        page = reader.pages[0]
        print(page.extract_text())
                