import fitz
from contextSync import ContextSync
import re
import pyautogui
import pytesseract
import numpy as np
import cv2
import nltk
from phraseSync import PhraseSync
import os
import json
from PIL import Image
from pathlib import Path

class MetaverseGenerator():
    def __init__(self, pdf_url):
        self.pdf = pdf_url  
        self.sync = ContextSync()
        nltk.download('stopwords')
        self.json_data_dir = Path.cwd() / "json_image_filestore"        
        self.png_data_dir = Path.cwd() / "png_image_filestore"
        
    def clean_text(self, text):
        #text = text.replace("Alice’s Adventures in Wonderland", '')
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'Free eBooks at Planet eBook.com','', text)
        return text

    def delete_from_text(self, list_indexes, tokens):
        for index in sorted(list_indexes, reverse=True):
            del tokens[index]

    def process_text(self, splitted_text_with_punctuation):

        junk_words = ['Notepad', 'File', 'Edit', 'Format', 'View', 'Help', 'Windows', '(CRLF)', 'Ln', 'Col', 'PM', 'AM', 'Adobe', 'Reader', 'Acrobat', 'Microsoft', 'AdobeReader', 'html', 'Tools', 'Fill', 'Sign','Comment', 'Bookmarks', 'Bookmark', 'History', 'Soren', 'Window', 'ES', 'FQ', '(SECURED)', 'pdf', 'de)', 'x', 'wl']
        def match_words_with_punctuation(word):
            return bool(re.match('^[.,;\-?!()\""]?[a-zA-Z]+[.,;\-?!()\""]*',word))
  
        stop_words = nltk.corpus.stopwords.words("english")
        stop_words.append("I")
  
        def eliminate_miscellaneous_chars(tokens):
            indexes_to_del = set()
            for ind,token in enumerate(tokens):
                if token not in stop_words and (token.isnumeric() or token in junk_words or match_words_with_punctuation(token)==False or (len(token)==1 and token.isalpha())) :
                    indexes_to_del.add(ind)
            return indexes_to_del
    
      
        indexes = eliminate_miscellaneous_chars(splitted_text_with_punctuation)
        self.delete_from_text(indexes, splitted_text_with_punctuation)
        return indexes  


    # Extract the text per page and per chapter
    def process_text_load_metaverse(self):

        with fitz.open(self.pdf) as doc:
                spoken_phrase = "pleasure of making a daisy-chain would".split()
                first_word, last_word = spoken_phrase[0], spoken_phrase[-1]
                page = doc[0]
                words = page.get_text("words")

               
                rl = page.search_for("pleasure of making a daisy-chain would", quads = True)
                # mark all found quads with one annotation
                page.add_squiggly_annot(rl)
                
                #for tuple in words:
                #    if first_word == tuple[4]:
                #        first_word_ind = tuple[:4]
                #    if last_word == tuple[4]:
                #        last_word_ind = tuple[:4]
                #        break
                #first_word_ind, last_word_ind = page.search_for(first_word), page.search_for(last_word)
                
                #for inst in text_instances:
                #    highlight = page.add_highlight_annot(inst)
                #    #highlight.set_colors({"stroke":(0, 0, 1), "fill":(0.75, 0.8, 0.95)})
                #    highlight.update()
                
                
                story_text = ''
                for page_no, page in enumerate(doc):
                    text = page.get_text()
                    text = text[text.find('Chapter') + len('Chapter') + 3:]
                    text = self.clean_text(text)
                    self.sync.contextSyncForMetaverse(text, page_no)

                #    story_text += page.get_text()
                   
                #story_text = story_text[story_text.find('Chapter') + len('Chapter') + 2:]
                #story_text = self.clean_text(story_text)

                #chapter_no = -1 
                #while(story_text.find('Chapter')!=-1):
                #    chapter = story_text[:story_text.find('Chapter')]
                #    chapter_no += 1
                #    #print("Start of a new chapter\n\n", chapter, '\n')
                #    story_text = story_text[story_text.find('Chapter') + len('Chapter') + 2:]
                
                #    # Generate Metaverse and Proceed with the Context Analyis
                #    self.sync.contextSyncForMetaverse(chapter, chapter_no)
                

                # # Last chapter 
                #print("Last chapter\n", story_text)
                #self.sync.contextSyncForMetaverse(story_text, chapter_no+1)

            

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

    def read_json(self, target_page, target_phrase_ind):
                print(os.listdir(self.png_data_dir)[0])
                for ind_json_file, json_file in enumerate(os.listdir(self.json_data_dir)): # binary search
                    with open(self.json_data_dir/json_file, "r",  encoding="utf-8") as file_name:
                        data = json.load(file_name)
                    if data['page'] == target_page:
                        if data['start_ind_phrase'] <= target_phrase_ind <= data['end_ind_phrase']:
                            png_subfolder = os.listdir(self.png_data_dir)[ind_json_file]
                            image = os.listdir(self.png_data_dir/png_subfolder)[0]
                            print(ind_json_file, image)
                            img = Image.open(self.png_data_dir/png_subfolder/image)
                            img.show()
                            cv2.waitKey(0)
                            #sys.exit() # to exit from all the processes


    def contextSyncForReading(self, phrase_spoken):
        img = pyautogui.screenshot()
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
                
        data = pytesseract.image_to_data(img, output_type = pytesseract.Output.DICT, lang="eng")
        indexes_to_del = self.process_text(data['text'])

        for key in data.keys():
            if key!='text':
                self.delete_from_text(indexes_to_del, data[key])
                            
        input_text_for_sync = data['text']
        co_ord_list = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))
        print(phrase_spoken)
        _, index_sentence = PhraseSync(phrase_spoken).phrase_sync(input_text_for_sync, co_ord_list)
        self.read_json(0, index_sentence)
            