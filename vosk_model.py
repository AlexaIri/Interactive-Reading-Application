import queue
from pygments import highlight
import sounddevice as sd
import vosk
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import  QCoreApplication
from LiveSubtitleWidget import LiveSubtitleWidget
vosk.SetLogLevel(-1)
import sys
import json
import pyautogui
import os
import time
import pytesseract
from PIL import Image
import re
import numpy as np
import cv2
import spacy
from itertools import islice
import nltk
from gpt3Api import ImageGenerator
import nltk, re
from nltk.collocations import *
from pathlib import Path
import math
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.text import Text
from pronunciation_checker import PronunciationChecker
from reading_tracker import ReadingTracker
from metaverseGenerator import MetaverseGenerator
import fitz

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Asus ZenBook\AppData\Local\Tesseract-OCR\tesseract'

class VoskModel():
    def __init__(self, lang, model_path, mode='transcription', safety_word='stop'):
        self.model_path = model_path  # self._get_model_path()
        self.q = queue.Queue()
        self.previous_line = ""
        self.previous_length = 0
        self.mode = mode
        self.safety_word = safety_word
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

    def setUp(self):

        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')

        if not os.path.exists(self.model_path):
            print("Please download a model for your language from https://alphacephei.com/vosk/models")
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

    def _get_model_path(self):
        full_path = os.path.realpath(__file__)
        file_dir = os.path.dirname(full_path)
        model_path = os.path.join(file_dir, 'models/vosk') 
        return model_path

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

    def clean_story_text(self, text):
        #text = text.replace("Alice’s Adventures in Wonderland", '')
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'Free eBooks at Planet eBook.com','', text)
        return text


    ################### KARAOKE READING #######################

    def karaoke_reading(self, sync_algorithm, selected_story_book): 
        rec,samplerate = self.setUp()
        try: 

            with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=None, dtype='int16', channels=1, callback=self._callback):
                time_of_prev_ss = time.perf_counter()
                img = pyautogui.screenshot()
                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

                data = pytesseract.image_to_data(img, output_type = pytesseract.Output.DICT, lang="eng")
                print("data generated from ss:", data)

                indexes_to_del = self.clean_text(data['text'])
                for key in data.keys():
                    if key!='text':
                        self.delete_from_text(indexes_to_del, data[key])

                print("NEWWWWWWWWWW data generated from ss:", data)
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
                                
                                    if 'text' in my_data: # read what is stored in the jsons
                                        PronunciationChecker(cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR), my_data, input_text_for_sync, self.co_ord_list).pronunciation_checker()
                                        #self.word_sync(cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR), d, input_text_for_sync) 
                                       
                                    if my_data[key] == self.safety_word : return
                                    self.previous_line = my_data[key]

                # Reading Tracker
                elif sync_algorithm == "reading_tracker":
                
                    while True:
                        # Take a new screenshot by comparing the timings of the previous and the current screenshots
                        # in case of scrolling down the story book to a new page 
                        time_of_latest_ss = time.perf_counter()
                        if(time_of_latest_ss - time_of_prev_ss > 30):
                            img = pyautogui.screenshot()
                            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

                            data = pytesseract.image_to_data(img, output_type = pytesseract.Output.DICT, lang="eng")
                            #print("data generated from ss:", data)

                            indexes_to_del = self.clean_text(data['text'])

                            for key in data.keys():
                                if key!='text':
                                    self.delete_from_text(indexes_to_del, data[key])

                            #print("NEWWWWWWWWWW data generated from ss:", data)
                
                            input_text_for_sync = data['text']
                            self.co_ord_list = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))
                        
                            time_of_prev_ss = time_of_latest_ss

                        # Record a new spoken sequence of words and highlight it accordingly by applying a text analysis algorithm (PhraseSync) 
                        data = self.q.get()
                        if rec.AcceptWaveform(data):
                            my_data = json.loads(rec.Result())
                        else:
                            my_data = json.loads(rec.PartialResult())
                        for key in my_data.keys():
                            if my_data[key]:
                                if my_data[key] != self.previous_line or key == 'text':
                                
                                    if 'text' in my_data: # read what is stored in the jsons, change the paths
                                        #self.phrase_sync(my_data, input_text_for_sync)
                                        ReadingTracker(my_data).reading_tracker(input_text_for_sync, self.co_ord_list)
                                        
                                    if my_data[key] == self.safety_word : return
                                    self.previous_line = my_data[key]

                # Sync Reading by Context
                else:
                    
                    MAIN_FOLDER_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__))))
                    PDF_STORY_PATH = os.path.join(MAIN_FOLDER_PATH,'Story Library', selected_story_book)
                                       
                    while True:
                        data = self.q.get()
                        if rec.AcceptWaveform(data):
                            my_data = json.loads(rec.Result())
                        else:
                            my_data = json.loads(rec.PartialResult())
                        for key in my_data.keys():
                            if my_data[key]:
                                if my_data[key] != self.previous_line or key == 'text':
                                
                                    if 'text' in my_data: # read what is stored in the jsons, change the paths
                                        MetaverseGenerator(PDF_STORY_PATH, selected_story_book).generateMetaverse(my_data, input_text_for_sync, self.co_ord_list)
                                        
                                    if my_data[key] == self.safety_word : return
                                    self.previous_line = my_data[key]
                    #with fitz.open(PDF_STORY_PATH) as doc:
                    #    change_page = False 
                    #    for page_no, page in enumerate(doc):
                    #        print("I M LA NAIBA SA MA IA: ", page_no, change_page)

                    #        if change_page == True: # the first page of the book is already screenshotted

                    #            img = pyautogui.screenshot()
                    #            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

                    #            data = pytesseract.image_to_data(img, output_type = pytesseract.Output.DICT, lang="eng")
                    #            #print("data generated from ss:", data)

                    #            indexes_to_del = self.clean_text(data['text'])
                    #            for key in data.keys():
                    #                if key!='text':
                    #                    self.delete_from_text(indexes_to_del, data[key])

                    #            input_text_for_sync = data['text']
                    #            self.co_ord_list = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))
                    #            change_page = False
                            
                    #        text = page.get_text()
                    #        text = text[text.find('Chapter') + len('Chapter') + 3:]
                    #        #print(text, "\n")
                    #        text = self.clean_story_text(text)
                    
                    #        while True:
                    #            data = self.q.get()
                    #            if rec.AcceptWaveform(data):
                    #                my_data = json.loads(rec.Result())
                    #            else:
                    #                my_data = json.loads(rec.PartialResult()) 
                    #            for key in my_data.keys():
                    #                if my_data[key]:
                    #                    if my_data[key] != self.previous_line or key == 'text':
                                
                    #                        if 'text' in my_data: # read what is stored in the jsons
                    #                           self.generateMetaverse(page_no, my_data, input_text_for_sync, self.co_ord_list)
                                           
                    #                        if my_data[key] == self.safety_word : return
                    #                        elif my_data[key] == "go to the next page":
                    #                            change_page = True
                    #                            break 

                    #                        self.previous_line = my_data[key]

                    #            if change_page:
                    #                break



        except KeyboardInterrupt:
            print('\nDone -- KEYBOARDiNTERRUPT')
        except Exception as e:
            print('exception', e)

    def read_json(self, target_page, target_phrase_ind):
        print(os.listdir(self.png_data_dir)[0])
        for ind_json_file, json_file in enumerate(os.listdir(self.json_data_dir)):
            with open(self.json_data_dir/json_file, "r",  encoding="utf-8") as file_name:
                data = json.load(file_name)
            if data['page'] == target_page:
                if data['start_ind_phrase'] <= target_phrase_ind <= data['end_ind_phrase']:
                    png_subfolder = os.listdir(self.png_data_dir)[ind_json_file]
                    image = os.listdir(self.png_data_dir/png_subfolder)[0]
                    print(ind_json_file, image)
                    
                    img = Image.open(self.png_data_dir/png_subfolder/image)
                    img.show() 
                    img.close()

    def generateMetaverse(self, page_number, phrase_spoken, input_text_for_sync, co_ord_list):
        print(phrase_spoken)
        try:
            _, index_sentence = PhraseSync(phrase_spoken).phrase_sync(input_text_for_sync, co_ord_list)
            self.read_json(page_number, index_sentence)
        except:
            print("Keep reading to see the metaverse getting displayed!")
