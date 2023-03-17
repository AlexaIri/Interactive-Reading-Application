import fitz
import re
import pytesseract
import nltk
from reading_tracker import ReadingTracker
import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import subprocess
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np


class MetaverseGenerator():
    def __init__(self, story_book_path, selected_story_book):
        self.story_book_path = story_book_path  
        self.selected_story_book = selected_story_book
        nltk.download('stopwords')
        self.json_data_dir = Path.cwd() / "JSON Images Filestore" / selected_story_book       
        self.png_data_dir = Path.cwd() / "PNG Images Filestore" / selected_story_book
       
    def clean_text(self, splitted_text_with_punctuation):

        junk_words = ['Kids.pdf', 'Moral', 'Stories', 'Kids', '.pdf', 'Notepad', 'File', 'Edit', 'Format', 'View', 'Help', 'Windows', '(CRLF)', 'Ln', 'Col', 'PM', 'AM', 'Adobe', 'Reader', 'Acrobat', 'Microsoft', 'AdobeReader', 'html', 'Tools', 'Fill', 'Sign','Comment', 'Bookmarks', 'Bookmark', 'History', 'Soren', 'Window', 'ES', 'FQ', '(SECURED)', 'pdf', 'de)', 'x', 'wl']
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


    def read_story_by_page(self):
        with fitz.open(self.story_book_path) as doc:
                page = doc.load_page(page_id=0)
                page_pix = page.get_pixmap()
                page_pix.save(f"{page.number}.png")
                data = pytesseract.image_to_data(f"{page.number}.png", output_type = pytesseract.Output.DICT, lang="eng")
                words = page.get_text("words") # the words with their coordinates, word_index, block_index, paragraph_index, respectively
                indexes_to_del = self.clean_text(data['text'])
                for key in data.keys():
                    if key!='text':
                        self.delete_from_text(indexes_to_del, data[key])

                print("NEWWWWWWWWWW data generated from ss:", data)
                input_text_for_sync = data['text']
                co_ord_list = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))

                self.metaverse_generator("One day, he went for a trip to some distant areas of his country.", input_text_for_sync, co_ord_list)
                

    def read_image_json(self, target_page, target_phrase_ind):
        print(os.listdir(self.png_data_dir)[0])

        for ind_json_file, json_file in enumerate(os.listdir(self.json_data_dir)):
            with open(self.json_data_dir/json_file, "r",  encoding="utf-8") as file_name:
                data = json.load(file_name)

            if data['page'] == target_page:
                if data['start_ind_phrase'] <= target_phrase_ind <= data['end_ind_phrase']:
                    png_subfolder = os.listdir(self.png_data_dir)[ind_json_file]
                    image = os.listdir(self.png_data_dir/png_subfolder)[0]
                    print(ind_json_file, image)
                    
                    # display image
                    img = Image.open(self.png_data_dir/png_subfolder/image)

                    # get the size of the image
                    width, height = img.size
                    print(width, height)

                    # print keywords 
                    keywords = "#" + " #".join(data['keywords'])
                    font = ImageFont.truetype("arial.ttf", 25)
                    id = ImageDraw.Draw(img)
                    id.text((int(39*width/100), int(95*height/100)), keywords, fill=(255, 255, 255), font = font)

                    img.show() 
                    img.close()

    def open_pdf_at_page(self, pdf_file, page_no):     
        PATH_TO_STORY_BOOK =  os.path.abspath(pdf_file)
        print(PATH_TO_STORY_BOOK)
        PATH_TO_ACROBAT_READER = os.path.abspath("C:\Program Files (x86)\Adobe\Reader 11.0\Reader\AcroRd32.exe") 

        # this will open our story book on page #page_no
        process = subprocess.Popen([PATH_TO_ACROBAT_READER, '/A', 'page={}'.format(page_no+1), PATH_TO_STORY_BOOK], shell=False, stdout=subprocess.PIPE)
        process.wait()

    def metaverse_generator(self, page_no, phrase_spoken, input_text_for_sync, co_ord_list):
        print(phrase_spoken)
        try:
            _, index_sentence = ReadingTracker(phrase_spoken).reading_tracker(input_text_for_sync, co_ord_list)
            self.read_image_json(page_no, index_sentence)
        except:
            print("Keep reading to see the metaverse getting displayed!")

    def create_img_grid():
        
        im1 = np.arange(100).reshape((10, 10))
        im2 = im1.T
        im3 = np.flipud(im1)
        im4 = np.fliplr(im2)

        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, [im1, im2, im3, im4]):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)

        plt.show()

            