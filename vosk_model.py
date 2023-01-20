import queue
from random import sample
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
#import turtle
from itertools import islice
import nltk

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Asus ZenBook\AppData\Local\Tesseract-OCR\tesseract'

class VoskModel():
    def __init__(self, lang, model_path, mode='transcription', safety_word='stop' ):
        self.model_path = model_path # self._get_model_path()
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
        # nltk.download()
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
    
    def karaoke_reading(self): # split this into 3 methods: wordSync, phraseSync, contextSync
        rec,samplerate = self.setUp()
        try: 

            with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=None, dtype='int16', channels=1, callback=self._callback):
                # time.sleep(6)
                img = pyautogui.screenshot()
                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

                words = pytesseract.image_to_string(img, lang="eng", config="--psm 6") #"eng" needs to be replaced by variable
                #print("words generated from ss:", words)
                data = pytesseract.image_to_data(img, output_type = pytesseract.Output.DICT, lang="eng")
                print("data generated from ss:", data)
                input_text_for_phrase_sync = data['text']

                initial_sentences = self.text_cleaning_and_sentence_detection(input_text_for_phrase_sync)
                first_sentence_from_prev_screenshot, last_sentence_from_prev_screenshot = initial_sentences[0], initial_sentences[-1]

                # regex matching: digits/alphabetical characters ("\W"); white spaces ("\S")
                words_clean = re.sub(r'[^\w\s]', '', words).lower() # cleans up the string generated from the screenshot, removing all the special characters apart from _ and returning the lower-case text without punctuation
                #print("words_clean\n", words_clean)
                splitted = words.split() # split the string according to the whitespaces into a list of words including the punctuation and the special characters 
                splitted_clean = words_clean.split()  # split the string according to the whitespaces into a list of words without the punctuation and the special characters 
                #print("The screenshot read the following-splitted clean \n",splitted_clean)
                #print("The indexes associated with each word from ss for phrase sync:\n", [(value, count) for count, value in enumerate(input_text_for_phrase_sync)])
                     
                    #dict(zip(splitted_clean, [i for i in range(len(splitted_clean))]))
                
                self.co_ord_list = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))
                co_ord_list_len = len(self.co_ord_list)
                index = 0
                while index<co_ord_list_len:
                    if self.co_ord_list[index][0].replace(" ", "") == "": 
                        self.co_ord_list.pop(index) # delete the whitespaces
                        co_ord_list_len = len(self.co_ord_list)
                    else:
                        index+=1
                 

                while True:
                    current_sentences = self.text_cleaning_and_sentence_detection(input_text_for_phrase_sync)
                    print("first sent first ss\n", first_sentence_from_prev_screenshot)
                    print("first sent current ss\n", current_sentences[0])
                    if (first_sentence_from_prev_screenshot != current_sentences[0] and last_sentence_from_prev_screenshot != current_sentences[-1]):

                        img = pyautogui.screenshot()
                        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

                        words = pytesseract.image_to_string(img, lang="eng", config="--psm 6") #"eng" needs to be replaced by variable
                        #print("words generated from ss:", words)
                        data = pytesseract.image_to_data(img, output_type = pytesseract.Output.DICT, lang="eng")
                        #print("data generated from ss:", data)
                        input_text_for_phrase_sync = data['text']

                        # regex matching: digits/alphabetical characters ("\W"); white spaces ("\S")
                        words_clean = re.sub(r'[^\w\s]', '', words).lower() # cleans up the string generated from the screenshot, removing all the special characters apart from _ and returning the lower-case text without punctuation
                        #print("words_clean\n", words_clean)
                        splitted = words.split() # split the string according to the whitespaces into a list of words including the punctuation and the special characters 
                        splitted_clean = words_clean.split()  # split the string according to the whitespaces into a list of words without the punctuation and the special characters 
                        #print("The screenshot read the following-splitted clean \n",splitted_clean)
                        #print("The indexes associated with each word from ss for phrase sync:\n", [(value, count) for count, value in enumerate(input_text_for_phrase_sync)])
                                #dict(zip(splitted_clean, [i for i in range(len(splitted_clean))]))
                
                        self.co_ord_list = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))
                        co_ord_list_len = len(self.co_ord_list)
                        index = 0
                        while index<co_ord_list_len:
                            if self.co_ord_list[index][0].replace(" ", "") == "": 
                                self.co_ord_list.pop(index) # delete the whitespaces
                                co_ord_list_len = len(self.co_ord_list)
                            else:
                                index+=1
                        first_sentence_from_prev_screenshot, last_sentence_from_first_screenshot = current_sentences[0], current_sentences[-1]
                    # MAYBE TAKE A NEW SCREENSHOT IF U SCROLL THE PAGE, aka RETAIN THE FIRST SENTENCE ON EACH PAGE, IF IT IS DIFFERENT, THEN TAKE A NEW SS
                    data = self.q.get()
                    if rec.AcceptWaveform(data):
                        d = json.loads(rec.Result())
                    else:
                        d = json.loads(rec.PartialResult())
                    for key in d.keys():
                        if d[key]:
                            if d[key] != self.previous_line or key == 'text':
                                
                              
                                if d[key] == self.safety_word : return
                                self.previous_line = d[key]

        except KeyboardInterrupt:
            print('\nDone -- KEYBOARDiNTERRUPT')
        except Exception as e:
            print('exception', e)
        

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

