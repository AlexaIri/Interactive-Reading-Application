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
from metaverseGenerator import MetaverseGenerator

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

                    #    data = pytesseract.image_to_data(img, output_type = pytesseract.Output.DICT, lang="eng")
                    #    #print("data generated from ss:", data)

                        
                    #    indexes_to_del = self.clean_text(data['text'])

                    #    for key in data.keys():
                    #        if key!='text':
                    #            self.delete_from_text(indexes_to_del, data[key])

                    #    print("NEWWWWWWWWWW data generated from ss:", data)
                
                    #    input_text_for_sync = data['text']
                    #    contiguous_text = ' '.join(input_text_for_sync)
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
                                        PhraseSync(my_data).phrase_sync(input_text_for_sync, self.co_ord_list)
                                    else:
                                        MetaverseGenerator(r"C:\Users\Asus ZenBook\Desktop\UCL\Alice_removed.pdf").contextSyncForReading(my_data)

                                    
                                if my_data[key] == self.safety_word : return
                                self.previous_line = my_data[key]

        except KeyboardInterrupt:
            print('\nDone -- KEYBOARDiNTERRUPT')
        except Exception as e:
            print('exception', e)

    ######################## PHRASE SYNC #################################

    def phrase_sync(self, phrase_spoken, splitted_text_with_punctuation):  # say a few words and see from which phrase they come from, and highlight everything from the start of the sentence 
                                                                           # to where it ends which is where the punctuation mark is ('.!?;:')
        phrase_spoken = phrase_spoken['text']
        print("phrase spoken\n", phrase_spoken)

        # Step 1: eliminate trailing whitespaces from the text
        def eliminate_leading_whitespaces(text):
            ind = 0
            while ind<len(text) and (text[ind].isspace() or text[ind] == ""):
                ind += 1
            return ind
       
        # Step 2: an enter = 3 consecutive whitespaces (" "); find the first occurence of exactly three consecutive whitespaces to see which/where the first word of the first phrase is/starts from
        # Why is it needed? (Challenge behind) when screenshotting the pdf/html/word document, every textual element gets added in the string (e.g., Notepad, Adobe, Acrobat. File, Edit, Save, Exit etc) which
        # is not neccessarily part of the story, therefore these need to be ignored and start the text processing and analysis from the first 'true' word of the story 

        # find the index of the first 'true' word of the story 
        def index_first_word_of_story(whitespace_indexes): 
            for i in range(len(whitespace_indexes)-2):
                if(whitespace_indexes[i+1]-whitespace_indexes[i] == 1 and whitespace_indexes[i+2]-whitespace_indexes[i+1] == 1):
                    if i and whitespace_indexes[i]-whitespace_indexes[i-1]!=1 and i+3<len(whitespace_indexes) and whitespace_indexes[i+3]-whitespace_indexes[i+2]!=1:
                        return i
  
        def get_splitted_clean_text(splitted_text):
            index = 0
            while index<len(splitted_text):
                if splitted_text[index].replace(" ", "") == "": 
                    splitted_text.pop(index) # delete the whitespaces
                else:
                    index+=1
            return splitted_text

        story_text = ' '.join(splitted_text_with_punctuation)

  
        # Step 3: Sentence Detection via spacy, store the sentences in a list; crate a dictionary with keys as sentence indexes and (start_index_of_sentence, end_index_of_sentence) as values

        def get_sentences(text):
            about_text = self.nlp(text) # text is a string including the whole screenshotted story, with the punctuation included
            return list(about_text.sents)
        sentences = get_sentences(story_text)
  
        print("sentences\n:")
        for count, sentence in enumerate(sentences):
            print("Sentence number", count, " ", sentence, "\n")

        def get_dictio_sentences(phrases):
            sentence_metadata = dict()
            for i in range(len(phrases)):
                number_of_words_in_phrase = len(phrases[i].text.split())
                if not i:
                    sentence_metadata[i] = (i, number_of_words_in_phrase-1)
                    prev_end = number_of_words_in_phrase
                else:
                    sentence_metadata[i] = (prev_end, prev_end+number_of_words_in_phrase-1)
                    prev_end += number_of_words_in_phrase
            return sentence_metadata
  
        sentence_metadata =  get_dictio_sentences(sentences)
        print("sentence_metadata\n", sentence_metadata)
  
        # Step 4: Detect the Longest Common Contiguous Subsequence of the phrase_said and the sentences generated via spacy to determine the sentence which the spoken phrase is part of

        def is_sublist(source, target):
            slen = len(source)
            return any(all(item1 == item2 for (item1, item2) in zip(source, islice(target, i, i+slen))) for i in range(len(target) - slen + 1))
  
        def long_substr_by_word(data): # tb refacuta ca e copiata https://stackoverflow.com/questions/47099802/longest-common-sequence-of-words-from-more-than-two-strings
            subseq = []
            data_seqs = [s.split(' ') for s in data]
            if len(data_seqs) > 1 and len(data_seqs[0]) > 0:
                for i in range(len(data_seqs[0])):
                    for j in range(len(data_seqs[0])-i+1):
                        if j > len(subseq) and all(is_sublist(data_seqs[0][i:i+j], x) for x in data_seqs):
                            subseq = data_seqs[0][i:i+j]
            return (' '.join(subseq), len(subseq))
    
        def belong_to_which_sentence(phrase_spoken, sentences):
            max_length_similarity, detected_sentence = 0, ''
            for ind_sentence, sentence in enumerate(sentences):
                data = [phrase_spoken, sentence.text]
                print(long_substr_by_word(data)[1])
                if max_length_similarity<long_substr_by_word(data)[1]:
                    max_length_similarity, detected_sentence = long_substr_by_word(data)[1], (sentence, ind_sentence)
            if(detected_sentence == ''):
                return -1
           
            return detected_sentence
        # phrase_spoken = ' '.join(phrase_spoken)
        if( belong_to_which_sentence(phrase_spoken, sentences) != -1):

            detected_sentence, index_sentence = belong_to_which_sentence(phrase_spoken, sentences) # if the long substr by word does not return anything, this line will fail because there will be nothing to unpack, put a try catch or something
        
            #gpt3 = MetaverseGenerator(detected_sentence.text)
            #gpt3.retrieve_image_from_gpt3OpenAI()

            print("The sentence to be highlighted: ", detected_sentence)
            print(sentence_metadata[index_sentence][0], sentence_metadata[index_sentence][1])
            self.highlight(sentence_metadata[index_sentence][0], sentence_metadata[index_sentence][1])
        else:
            print("Nothing to highlight in the story! Try reading something else!")
        
    
    ###################### HIGHLIGHT A SEQUENCE OF WORDS ##############################
    def highlight(self, first_word_index, last_word_index):
        listuta = [(value, count) for count, value in enumerate(self.co_ord_list)]
        print("vreau sa vad\n", listuta)
        print("THE HIGHLIGHTING STARTS")
        pyautogui.moveTo(self.co_ord_list[first_word_index][1], self.co_ord_list[first_word_index][2], duration = 0.1)
        pyautogui.click()
        pyautogui.keyDown('shift') # press the key
        pyautogui.keyDown('ctrl')
        pyautogui.dragTo(self.co_ord_list[last_word_index][1]+self.co_ord_list[last_word_index][3], 
            self.co_ord_list[last_word_index][2]+self.co_ord_list[last_word_index][4], duration = 0.3)
        pyautogui.keyUp('shift') # release the key
        pyautogui.keyUp('ctrl')