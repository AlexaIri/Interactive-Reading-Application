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
from wordSync import WordSync
from phraseSync import PhraseSync
from contextSync import ContextSync

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


    ################### KARAOKE READING #######################

    def karaoke_reading(self, sync_algorithm): 
        rec,samplerate = self.setUp()
        try: 

            with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=None, dtype='int16', channels=1, callback=self._callback):
                time_of_prev_ss = time.perf_counter()
                img = pyautogui.screenshot()
                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

                words = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
                data = pytesseract.image_to_data(img, output_type = pytesseract.Output.DICT, lang="eng")
                print("data generated from ss:", data)

                indexes_to_del = self.clean_text(data['text'])

                for key in data.keys():
                    if key!='text':
                        self.delete_from_text(indexes_to_del, data[key])

                print("NEWWWWWWWWWW data generated from ss:", data)
                

                input_text_for_sync = data['text']
                contiguous_text = ' '.join(input_text_for_sync)
                #self.contextSync(contiguous_text)
                sentences = self.get_sentences(contiguous_text)
                first_sentence_from_prev_screenshot, last_sentence_from_prev_screenshot = sentences[0], sentences[-1]
                
                self.co_ord_list = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))
                 
                while True:
                    current_sentences = self.get_sentences(contiguous_text)
                    time_of_latest_ss = time.perf_counter()
                    #print("first sent first ss\n", first_sentence_from_prev_screenshot)
                    #print("first sent current ss\n", current_sentences[0])
                    #if (first_sentence_from_prev_screenshot != current_sentences[0] and last_sentence_from_prev_screenshot != current_sentences[-1]):
                    #if(time_of_latest_ss - time_of_prev_ss > 30):
                    #    img = pyautogui.screenshot()
                    #    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

                    #    words = pytesseract.image_to_string(img, lang="eng", config="--psm 6") #"eng" needs to be replaced by variable
                    #    #print("words generated from ss:", words)
                    #    data = pytesseract.image_to_data(img, output_type = pytesseract.Output.DICT, lang="eng")
                    #    #print("data generated from ss:", data)

                        
                    #    indexes_to_del = self.clean_text(data['text'])

                    #    for key in data.keys():
                    #        if key!='text':
                    #            self.delete_from_text(indexes_to_del, data[key])

                    #    print("NEWWWWWWWWWW data generated from ss:", data)
                

                    #    input_text_for_sync = data['text']
                    #    contiguous_text = ' '.join(input_text_for_sync)
                    #    #self.contextSync(contiguous_text)
                    #    sentences = self.get_sentences(contiguous_text)

                    #    self.co_ord_list = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))
                        
                    #    first_sentence_from_prev_screenshot, last_sentence_from_prev_screenshot = current_sentences[0], current_sentences[-1]
                    #    time_of_prev_ss = time_of_latest_ss

                    #MAYBE TAKE A NEW SCREENSHOT IF U SCROLL THE PAGE, aka RETAIN THE FIRST SENTENCE ON EACH PAGE, IF IT IS DIFFERENT, THEN TAKE A NEW SS
                    data = self.q.get()
                    if rec.AcceptWaveform(data):
                        my_data = json.loads(rec.Result())
                    else:
                        my_data = json.loads(rec.PartialResult())
                    for key in my_data.keys():
                        if my_data[key]:
                            if my_data[key] != self.previous_line or key == 'text':
                                
                                if 'text' in my_data:
                                    if sync_algorithm == "word_sync":
                                        WordSync(cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR), my_data, input_text_for_sync, self.co_ord_list).word_sync()
                                        #self.word_sync(cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR), d, input_text_for_sync) 
                                    elif sync_algorithm == "phrase_sync":
                                        #self.phrase_sync(my_data, input_text_for_sync)
                                        PhraseSync(my_data, input_text_for_sync, self.co_ord_list).phrase_sync()
                                    else:
                                        ContextSync().contextSync(my_data)
                                    
                                if my_data[key] == self.safety_word : return
                                self.previous_line = my_data[key]

        except KeyboardInterrupt:
            print('\nDone -- KEYBOARDiNTERRUPT')
        except Exception as e:
            print('exception', e)
