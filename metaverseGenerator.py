import fitz
from contextSync import ContextSync
import re

class MetaverseGenerator():
    def __init__(self, pdf_url):
        self.pdf = pdf_url  
        self.sync = ContextSync()
        
    def clean_text(self, text):
        #text = text.replace("Alice’s Adventures in Wonderland", '')
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'Free eBooks at Planet eBook.com','', text)
        return text


    # Extract the text per page and per chapter
    def extract_text(self):

        with fitz.open(self.pdf) as doc:
            #    text = "pleasure"
            #    text_instances = page.search_for(text)

            #    for inst in text_instances:
            #        highlight = page.add_highlight_annot(inst)
            #        #highlight.set_colors({"stroke":(0, 0, 1), "fill":(0.75, 0.8, 0.95)})
            #        highlight.update()
            # print(page.get_text("words"))

            story_text = ''
            for page in doc:
                story_text += page.get_text()

            story_text = story_text[story_text.find('Chapter') + len('Chapter') + 2:]
            story_text = self.clean_text(story_text)

            chapter_no = -1 
            while(story_text.find('Chapter')!=-1):
                chapter = story_text[:story_text.find('Chapter')]
                chapter_no += 1
                #print("Start of a new chapter\n\n", chapter, '\n')
                story_text = story_text[story_text.find('Chapter') + len('Chapter') + 2:]
                
                # Generate Metaverse and Proceed with the Context Analyis
                self.sync.contextSyncForMetaverse(chapter, chapter_no)
               

             # Last chapter 
            print("Last chapter\n", story_text)
            self.sync.contextSyncForMetaverse(story_text, chapter_no+1)

            
       




        #page = self.pdf.load_page(0)  # number of page
        #pix = page.get_pixmap()
        #print(type(pix))
        #output = "outfile.png"
        #pix.save(output)
        ##img = pyautogui.screenshot(output)
        ##print(type(img))
        ##img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
       
        #data = pytesseract.image_to_data(Image.open(output), output_type = pytesseract.Output.DICT, lang="eng")
        #indexes_to_del = self.clean_text(data['text'])

        #for key in data.keys():
        #    if key!='text':
        #        self.delete_from_text(indexes_to_del, data[key])
                
        #contiguous_text = ' '.join(data['text'])
        #print("FMMMMM", contiguous_text, '\n')
        #self.sync.contextSync(contiguous_text)
      