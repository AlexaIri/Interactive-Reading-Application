import fitz
from contextSync import ContextSync

class MetaverseGenerator():
    def __init__(self, pdf):
        self.pdf = fitz.open(pdf)
        self.sync = ContextSync()

    # Extract the text per page
    def extract_text(self):
        for page in self.pdf:
          #text_in_page = page.get_text()
          print(page.get_text())
          #self.sync.contextSync(text_in_page)
         

       