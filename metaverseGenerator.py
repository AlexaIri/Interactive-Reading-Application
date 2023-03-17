import fitz
from contextSync import ContextSync
import re
import pyautogui
import pytesseract
import cv2
import nltk
from phraseSync import PhraseSync
import json
from PIL import Image
from pathlib import Path
import sys
import subprocess
import os


class MetaverseGenerator():
    def __init__(self, story_book_path, selected_story_book):
        self.story_book_path = story_book_path  
        self.selected_story_book = selected_story_book
        self.sync = ContextSync()
        nltk.download('stopwords')
        self.json_data_dir = Path.cwd() / "JSON Images Filestore" / selected_story_book       
        self.png_data_dir = Path.cwd() / "PNG Images Filestore" / selected_story_book
        
    def clean_text(self, text):
        text = re.sub(self.selected_story_book, "", text)
        text = re.sub(r'Stories', "", text)
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'Free eBooks at Planet eBook.com','', text)
        text = re.sub(r'^(www)?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) # nu merge
        text = re.sub(r'www.islamicoccasions.com', "", text)
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
        with fitz.open(self.story_book_path) as doc:
                page = doc.load_page(page_id=4)
                words = page.get_text("words") # the words with their coordinates, word_index, block_index, paragraph_index, respectively
                print(words)

                spoken_phrase = "honor my father and mother".split()
                first_word, last_word = spoken_phrase[0], spoken_phrase[-1]
                first_word_ind, last_word_ind = page.search_for(first_word), page.search_for(last_word)

                #self.highlight(first_word_ind[0][0], first_word_ind[0][1], last_word_ind[0][2], last_word_ind[0][3])
                #print("BLOCCKKK", page.get_text("blocks", sort=False))
                #print(page)
                
               
                #rl = page.search_for(spoken_phrase, quads = True)
                
                #page = doc.reload_page(page)
                
                shape = page.new_shape()
                rectangular_annotator = fitz.Rect(40.821800231933594, first_word_ind[0][1], 557.24462890625, last_word_ind[0][3])
                #rectangular_annotator = fitz.Rect(first_word_ind[0][0], first_word_ind[0][1],  last_word_ind[0][2], last_word_ind[0][3])

                print(rectangular_annotator)
                shape.draw_rect(rectangular_annotator)
                shape.finish(color = (1, 0, 0))
                shape.commit()
                save_file_with_annotation = "pnm.pdf"
                doc.save(save_file_with_annotation) #, incremental=True,encryption=fitz.PDF_ENCRYPT_KEEP)
                self.open_pdf_at_page(save_file_with_annotation, 4)
                
    def highlight(self, word1_x, word1_y, word2_width, word2_height):
        #word_placement = [(value, count) for count, value in enumerate(self.co_ord_list)]
        print("The highlighting on page starts\n")
        pyautogui.click(x=word1_x, y=word1_y,  duration = 0.1)
        pyautogui.keyDown('shift') # press the key
        pyautogui.keyDown('ctrl')
        pyautogui.dragTo(x=word2_width, y=word2_height, duration = 0.5)
        pyautogui.click()
        pyautogui.keyUp('shift') # release the key
        pyautogui.keyUp('ctrl')

    def read_image_json(self, target_page, target_phrase_ind):
        print(os.listdir(self.png_data_dir)[0])
        #pyautogui.prompt(text="#" + " #".join(data['keywords']), title='Image keywords' , default='')

        for ind_json_file, json_file in enumerate(os.listdir(self.json_data_dir)):
            with open(self.json_data_dir/json_file, "r",  encoding="utf-8") as file_name:
                data = json.load(file_name)
            if data['page'] == target_page:
                if data['start_ind_phrase'] <= target_phrase_ind <= data['end_ind_phrase']:
                    png_subfolder = os.listdir(self.png_data_dir)[ind_json_file]
                    image = os.listdir(self.png_data_dir/png_subfolder)[0]
                    print(ind_json_file, image)
                    # print keywords - data['keywords']
                    img = Image.open(self.png_data_dir/png_subfolder/image)
                    img.show() 
                    img.close()

    def open_pdf_at_page(self, pdf_file, page_no):     
        PATH_TO_STORY_BOOK =  os.path.abspath(pdf_file)
        print(PATH_TO_STORY_BOOK)
        PATH_TO_ACROBAT_READER = os.path.abspath("C:\Program Files (x86)\Adobe\Reader 11.0\Reader\AcroRd32.exe") 

        # this will open our story book on page #page_no
        process = subprocess.Popen([PATH_TO_ACROBAT_READER, '/A', 'page={}'.format(page_no+1), PATH_TO_STORY_BOOK], shell=False, stdout=subprocess.PIPE)
        process.wait()

    def generateMetaverse(self, phrase_spoken, input_text_for_sync, co_ord_list):
        #img = pyautogui.screenshot()
        #img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
                
        #data = pytesseract.image_to_data(img, output_type = pytesseract.Output.DICT, lang="eng")
        #indexes_to_del = self.process_text(data['text'])

        #for key in data.keys():
        #    if key!='text':
        #        self.delete_from_text(indexes_to_del, data[key])
          
        #input_text_for_sync = data['text']
        #co_ord_list = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))
        print(phrase_spoken)
        try:
            _, index_sentence = PhraseSync(phrase_spoken).phrase_sync(input_text_for_sync, co_ord_list)
            self.read_image_json(0, index_sentence)
        except:
            print("Keep reading to see the metaverse getting displayed!")
            