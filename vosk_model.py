import queue
import sounddevice as sd
import vosk
vosk.SetLogLevel(-1)
import sys
import json
import pyautogui
import os
import time
import pytesseract
import re
import numpy as np
import cv2
import spacy
import nltk
import nltk, re
from nltk.collocations import *
from pathlib import Path
from pronunciation_checker import PronunciationChecker
from reading_tracker import ReadingTracker
from metaverse_generator import MetaverseGenerator
import fitz

TESSERACT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'AppData\Local\Tesseract-OCR','tesseract'))
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Class following the structure of the VoskModel calss from the MotionInput codebase in vosk_model.py file
class VoskModel():
    def __init__(self, lang, model_path, mode='transcription', safety_word='stop'):
        self.model_path = model_path  # self._get_model_path()
        self.q = queue.Queue()
        self.previous_line = ""
        self.previous_jump = 0
        self.mode = mode
        self.safety_word = safety_word
        self.change_page = "next page"
        self.lang = lang
        self.rec = ""
        self.text_dict = {}
        self.co_ord_list = []
        self.last_index = -1
        self.first_index = -1
        self.match = False
        self.nlp = spacy.load("en_core_web_sm")
        
        self.json_data_dir = Path.cwd() / "json_image_filestore"        
        self.png_data_dir = Path.cwd() / "png_image_filestore"

    # Helper method taken from the FunReaders from MotionInput codebase in vosk_model.py file
    def setUp(self):
        
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')

        if not os.path.exists(self.model_path):
            print(f"and unpack into {self.model_path}.")
        print("model's path:" + self.model_path)
        device_info = sd.query_devices(kind='input')
        samplerate = int(device_info['default_samplerate'])
        location = self.model_path+ "\\"+ self.lang 
        
        model = vosk.Model(location)

        rec = vosk.KaldiRecognizer(model, samplerate)
        print('#' * 80)
        print('Press Ctrl+C to stop the recording')
        print('#' * 80)
        return rec,samplerate

    # Helper method taken from the FunReaders from MotionInput codebase in vosk_model.py file
    def _get_model_path(self):
        full_path = os.path.realpath(__file__)
        file_dir = os.path.dirname(full_path)
        model_path = os.path.join(file_dir, 'models/vosk') 
        return model_path

    # Helper method taken from the FunReaders from MotionInput codebase in vosk_model.py file
    def _callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
            sys.stdout.flush()
        self.q.put(bytes(indata))

    ##################### TEXT CLEANING AND PRE-PROCESSING ##########################

    def delete_from_text(self, list_indexes, tokens):
        for index in sorted(list_indexes, reverse=True):
            del tokens[index]


    def get_sentences(self, text):
        about_text = self.nlp(text) # text is a string including the whole screenshotted story, with the punctuation included
        return list(about_text.sents)
    

    def clean_text(self, splitted_text_with_punctuation, blocks, target_block_to_del, book_title):
        junk_words = ['for Kids.pdf', "for", "kidspdf", 'Moral', 'Stories', 'Kids', '.pdf', 'Notepad', 'File', 'Edit', 'Format', 'View', 'Help', 'Windows', '(CRLF)', 'Ln', 'Col', 'PM', 'AM', 'Adobe', 'Reader', 'Acrobat', 'Microsoft', 'AdobeReader', 'html', 'Tools', 'Fill', 'Sign','Comment', 'Bookmarks', 'Bookmark', 'History', 'Soren', 'Window', 'ES', 'FQ', '(SECURED)', 'pdf', 'de)', 'x', 'wl']
        junk_words.extend(book_title.split())# delete the title of the book from the document
        def match_words_with_punctuation(word):
            return bool(re.match('^[.,;\-?!()\""]?[a-zA-Z]+[.,;\-?!()\""]*', word))
  
        stop_words = nltk.corpus.stopwords.words("english")
        stop_words.append("I")
  
        def eliminate_miscellaneous_chars(tokens):
            indexes_to_del = set()
            for ind,token in enumerate(tokens):
                if token not in stop_words and (token.isnumeric() or token in junk_words or match_words_with_punctuation(token)==False or (len(token)==1 and token.isalpha())) :
                    indexes_to_del.add(ind)
            return indexes_to_del
        # Code for fixing the bug of highlighting the navigation bar/taskbar, the book title alongside the first sentence
        #def eliminate_start_junk_tokens(blocks, target_block_to_del):
        #    indexes_to_del = set()
        #    for ind, b in enumerate(blocks):
        #        if b == target_block_to_del:
        #            indexes_to_del.add(ind)
        #    return indexes_to_del
        indexes = eliminate_miscellaneous_chars(splitted_text_with_punctuation)        
        #indexes.update(eliminate_start_junk_tokens(blocks, target_block_to_del))
        self.delete_from_text(indexes, splitted_text_with_punctuation)
        return indexes  

    ################### LIVE READING #######################
    def live_reading(self, sync_algorithm, selected_story_book): 
        rec,samplerate = self.setUp()
        try: 
            with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=None, dtype='int16', channels=1, callback=self._callback):
                time_of_prev_ss = time.perf_counter()
                img = pyautogui.screenshot()
                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

                data = pytesseract.image_to_data(img, output_type = pytesseract.Output.DICT, lang="eng")
                indexes_to_del = self.clean_text(data['text'], data['block_num'], data['block_num'][0], selected_story_book)
                indexes_to_del.add(0)
                del data['text'][0]
                for key in data.keys():
                    if key!='text':
                        self.delete_from_text(indexes_to_del, data[key])

                # Code for fixing the bug of highlighting the navigation bar/taskbar, the book title alongside the first sentence
                #indexes_to_del = self.eliminate_start_junk_tokens(data['block_num'], data['block_num'][0])
                #for key in data.keys():
                #    if key!='text':
                #        self.delete_from_text(indexes_to_del, data[key])

                print("Data generated from Tesseract OCR after pre-proceesing:", data)
                input_text_for_sync = data['text']
                self.co_ord_list = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))

                # Pronunciation Checker
                if sync_algorithm == "pronunciation_checker":
                   while True:
                        data = self.q.get()
                        if rec.AcceptWaveform(data):
                            my_data = json.loads(rec.Result())
                        else:
                            my_data = json.loads(rec.PartialResult())
                        for key in my_data.keys():
                            if my_data[key]:
                                if my_data[key] != self.previous_line or key == 'text':
                                
                                    if 'text' in my_data: 
                                        PronunciationChecker(cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR), my_data, input_text_for_sync, self.co_ord_list).pronunciation_checker()
                                       
                                    if my_data[key] == self.safety_word : return
                                    self.previous_line = my_data[key]

                # Karaoke Reading Tracker
                elif sync_algorithm == "reading_tracker":
                    while True:
                        # Take a new screenshot by comparing the timings of the previous and the current screenshots
                        # in case of scrolling down the story book to a new page 
                        time_of_latest_ss = time.perf_counter()
                        if(time_of_latest_ss - time_of_prev_ss > 5):
                            img = pyautogui.screenshot()
                            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

                            data = pytesseract.image_to_data(img, output_type = pytesseract.Output.DICT, lang="eng")

                            indexes_to_del = self.clean_text(data['text'], data['block_num'], data['block_num'][0], selected_story_book)
                            indexes_to_del.add(0)
                            del data['text'][0]
                            for key in data.keys():
                                if key!='text':
                                    self.delete_from_text(indexes_to_del, data[key])
                
                            input_text_for_sync = data['text']
                            self.co_ord_list = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))
                        
                            time_of_prev_ss = time_of_latest_ss

                        # Record a newly spoken sequence of words and highlight it accordingly by applying a text analysis algorithm 
                        data = self.q.get()
                        if rec.AcceptWaveform(data):
                            my_data = json.loads(rec.Result())
                        else:
                            my_data = json.loads(rec.PartialResult())
                        for key in my_data.keys():
                            if my_data[key]:
                                if my_data[key] != self.previous_line or key == 'text':
                                
                                    if 'text' in my_data: # read what is stored in the jsons, change the paths
                                        ReadingTracker(my_data['text']).reading_tracker(input_text_for_sync, self.co_ord_list)
                                        
                                    if my_data[key] == self.safety_word : return
                                    self.previous_line = my_data[key]

                # Metaverse Generator
                else:
                    
                    MAIN_FOLDER_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__))))
                    PDF_STORY_PATH = os.path.join(MAIN_FOLDER_PATH,'Story Library', selected_story_book)
                                 
                    with fitz.open(PDF_STORY_PATH) as doc:
                        # For each page of a book, read until saying "next page" to alert the system the page changed
                        # To exit the algorithm, say "stop"
                        for page_no, page in enumerate(doc):
                            page = doc.load_page(page_id=page_no)
                            page_pix = page.get_pixmap()
                            page_pix.save("page.png")

                            # Alternative approach to perform Tesseract OCR on a page
                            #data = pytesseract.image_to_data("page.png", output_type = pytesseract.Output.DICT, lang="eng")

                            img = pyautogui.screenshot()
                            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

                            data = pytesseract.image_to_data(img, output_type = pytesseract.Output.DICT, lang="eng")

                            indexes_to_del = self.clean_text(data['text'], data['block_num'], data['block_num'][0], selected_story_book)
                            indexes_to_del.add(0)
                            del data['text'][0]
                            for key in data.keys():
                                if key!='text':
                                    self.delete_from_text(indexes_to_del, data[key])

                
                            self.co_ord_list = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))
                        
                            page_changed = False
                            print("Data generated from screenshotting the page:", data)
                            input_text_for_sync = data['text']
                            co_ord_list = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))
                            while True:
                                data = self.q.get()
                                if rec.AcceptWaveform(data):
                                    my_data = json.loads(rec.Result())
                                else:
                                    my_data = json.loads(rec.PartialResult())
                                for key in my_data.keys():
                                    if my_data[key]:
                                        if my_data[key] != self.previous_line or key == 'text':
                                
                                            if 'text' in my_data: 
                                                MetaverseGenerator(PDF_STORY_PATH, selected_story_book).metaverse_generator(page_no, my_data['text'],                                                                                                    input_text_for_sync, co_ord_list)
                                        
                                            if my_data[key] == self.safety_word : return
                                            if my_data[key] == self.change_page:
                                               page_changed = True
                                               break
                                            self.previous_line = my_data[key]
                                if page_changed:
                                    break # proceed to the next page

        except KeyboardInterrupt:
            print('\nDone -- KEYBOARDiNTERRUPT')
        except Exception as e:
            print('exception', e)
