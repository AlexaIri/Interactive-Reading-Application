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
from gpt3Api import MetaverseGenerator

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Asus ZenBook\AppData\Local\Tesseract-OCR\tesseract'

class VoskModel():
    def __init__(self, lang, model_path, mode='transcription', safety_word='stop' ):
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

    def _callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
            sys.stdout.flush()
        self.q.put(bytes(indata))

    ##################### text_cleaning_and_sentence_detection ##########################

    def text_cleaning_and_sentence_detection(self, splitted_text_with_punctuation):
         # Step 1: eliminate trailing whitespaces from the text
        def eliminate_leading_whitespaces(text):
            ind = 0
            while ind<len(text) and (text[ind].isspace() or text[ind] == ""):
                ind += 1
            return ind

        splitted_text_with_punctuation = splitted_text_with_punctuation[eliminate_leading_whitespaces(splitted_text_with_punctuation):]
       
        # Step 2: an enter = 3 consecutive whitespaces (" "); find the first occurence of exactly three consecutive whitespaces to see which/where the first word of the first phrase is/starts from
        # Why is it needed? (Challenge behind) when screenshotting the pdf/html/word document, every textual element gets added in the string (e.g., Notepad, Adobe, Acrobat. File, Edit, Save, Exit etc) which
        # is not neccessarily part of the story, therefore these need to be ignored and start the text processing and analysis from the first 'true' word of the story 

        # create a list with all the indexes of the whitespaces in the list of strings 
        def get_whitespace_indexes(iterable, object):
            return (index for index, element in enumerate(iterable) if element == object)
  
        whitespace_indexes = list(get_whitespace_indexes(splitted_text_with_punctuation, ''))

        # find the index of the first 'true' word of the story 
        def index_first_word_of_story(whitespace_indexes): 
            for i in range(len(whitespace_indexes)-2):
                if(whitespace_indexes[i+1]-whitespace_indexes[i] == 1 and whitespace_indexes[i+2]-whitespace_indexes[i+1] == 1):
                    if i and whitespace_indexes[i]-whitespace_indexes[i-1]!=1 and i+3<len(whitespace_indexes) and whitespace_indexes[i+3]-whitespace_indexes[i+2]!=1:
                        return i
  
        index_first_word = whitespace_indexes[index_first_word_of_story(whitespace_indexes)]
        how_much_got_removed = index_first_word - 1 # teoretic ar tb sa fie index_first_word - eliminate_leading_whitespaces(splitted_text_with_punctuation)
        splitted_text = splitted_text_with_punctuation[(index_first_word+3):]

        def get_splitted_clean_text():
            index = 0
            while index<len(splitted_text):
                if splitted_text[index].replace(" ", "") == "": 
                    splitted_text.pop(index) # delete the whitespaces
                else:
                    index+=1
            return splitted_text
        splitted_clean_text = get_splitted_clean_text()

        clean_story_text = ' '.join(splitted_clean_text)

        print(how_much_got_removed)
        print("The indexes of everything after I cleaned the text:\n", [(value, count) for count, value in enumerate(splitted_clean_text)])

  
        # Step 3: Sentence Detection via spacy, store the sentences in a list; crate a dictionary with keys as sentence indexes and (start_index_of_sentence, end_index_of_sentence) as values

        def get_sentences(text):
            about_text = self.nlp(text) # text is a string including the whole screenshotted story, with the punctuation included
            return list(about_text.sents)
        sentences = get_sentences(clean_story_text)
        #sentences = [sentence for sentence in sentences if not sentence.text.isspace()]
        
        print(sentences)
  
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

        return sentences



    ################### KARAOKE READING #######################
    

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
                    #current_sentences = self.text_cleaning_and_sentence_detection(input_text_for_phrase_sync)
                    #print("first sent first ss\n", first_sentence_from_prev_screenshot)
                    #print("first sent current ss\n", current_sentences[0])
                    #if (first_sentence_from_prev_screenshot != current_sentences[0] and last_sentence_from_prev_screenshot != current_sentences[-1]):

                    #    img = pyautogui.screenshot()
                    #    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

                    #    words = pytesseract.image_to_string(img, lang="eng", config="--psm 6") #"eng" needs to be replaced by variable
                    #    #print("words generated from ss:", words)
                    #    data = pytesseract.image_to_data(img, output_type = pytesseract.Output.DICT, lang="eng")
                    #    #print("data generated from ss:", data)
                    #    input_text_for_phrase_sync = data['text']

                    #    # regex matching: digits/alphabetical characters ("\W"); white spaces ("\S")
                    #    words_clean = re.sub(r'[^\w\s]', '', words).lower() # cleans up the string generated from the screenshot, removing all the special characters apart from _ and returning the lower-case text without punctuation
                    #    #print("words_clean\n", words_clean)
                    #    splitted = words.split() # split the string according to the whitespaces into a list of words including the punctuation and the special characters 
                    #    splitted_clean = words_clean.split()  # split the string according to the whitespaces into a list of words without the punctuation and the special characters 
                    #    #print("The screenshot read the following-splitted clean \n",splitted_clean)
                    #    #print("The indexes associated with each word from ss for phrase sync:\n", [(value, count) for count, value in enumerate(input_text_for_phrase_sync)])
                    #            #dict(zip(splitted_clean, [i for i in range(len(splitted_clean))]))
                
                    #    self.co_ord_list = list(zip(data['text'], data['left'], data['top'], data['width'], data['height']))
                    #    co_ord_list_len = len(self.co_ord_list)
                    #    index = 0
                    #    while index<co_ord_list_len:
                    #        if self.co_ord_list[index][0].replace(" ", "") == "": 
                    #            self.co_ord_list.pop(index) # delete the whitespaces
                    #            co_ord_list_len = len(self.co_ord_list)
                    #        else:
                    #            index+=1
                    #    first_sentence_from_prev_screenshot, last_sentence_from_first_screenshot = current_sentences[0], current_sentences[-1]
                    # MAYBE TAKE A NEW SCREENSHOT IF U SCROLL THE PAGE, aka RETAIN THE FIRST SENTENCE ON EACH PAGE, IF IT IS DIFFERENT, THEN TAKE A NEW SS
                    data = self.q.get()
                    if rec.AcceptWaveform(data):
                        d = json.loads(rec.Result())
                    else:
                        d = json.loads(rec.PartialResult())
                    for key in d.keys():
                        if d[key]:
                            if d[key] != self.previous_line or key == 'text':
                                
                                
                                if 'text' in d:
                                    #self.word_sync(cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR), d, splitted_clean) 
                                    self.phrase_sync(d, input_text_for_phrase_sync)

                              
                                if d[key] == self.safety_word : return
                                self.previous_line = d[key]

        except KeyboardInterrupt:
            print('\nDone -- KEYBOARDiNTERRUPT')
        except Exception as e:
            print('exception', e)
        
    ######################## WORD SYNC #################################

    
    def word_sync(self,img, phrase_said, cleaned_words):

        """ if partial and self.match = false:
        continue
    if text = self.match = false
    if previous first and current first match - self.match = true previous len and current length = number of indices you have to highlight and current highlight function can do one at a time
        """
        phrase_said = phrase_said['text'].split()
        
        cleaned_list_of_words = [s.replace("?","").replace(",","").replace(".","").replace('"','').replace("!","").replace(":","").replace(";","") for s in cleaned_words]
        
        print("FINAL indexes associated with each word from ss:\n", [(value, count) for count, value in enumerate(cleaned_list_of_words)])
        print("we are comparing the phrase you just said:", phrase_said)
        
        matched_indexes_of_words = []
        indexes_of_correctly_pronounced_words = [] 
        for word_index in range(len(phrase_said)): 
            if phrase_said[word_index] in cleaned_list_of_words:
                matched_indexes_of_words.append([i for i,val in enumerate(cleaned_list_of_words) if val==phrase_said[word_index]])
               
                indexes_of_correctly_pronounced_words.append(word_index)
        print("indexes of our text ss: ", matched_indexes_of_words)
        print("indexes of our phrase: ", indexes_of_correctly_pronounced_words)
        first, last = self.get_indexes(matched_indexes_of_words, indexes_of_correctly_pronounced_words, len(phrase_said))
        print("first and last index:", first, last, '\n')

        if first!= -1:
            print("WHAT YOU JUST READ")
            print("this will be highlited from FIRST to LAST:" + " ".join(cleaned_words[first:last]))
            #self.highlight(first, last-1) 
            self.box_words(img, first, last-1, indexes_of_correctly_pronounced_words) 
            
            
    def get_indexes(self,list_of_indexes, corresponding_indexes, length):
        if len(list_of_indexes) == 0:
            return -1, -1
        if len(list_of_indexes) == 1:
            if len(list_of_indexes[0])==1: 
                #calculate beginning and end, and return to do this, need to know index of the current one, otherwise assume it is not enough
                
                first = list_of_indexes[0][0] - corresponding_indexes[0]
                last = first + length
                return first, last
        
        initial = -1
        found = -1
        increments = 0
        for index in range(len(list_of_indexes)-1):
            
            if found==-1:
                for x in list_of_indexes[index]:
                    if x+1 in list_of_indexes[index+1]:
                        if initial == -1:
                            initial = x
                        found = x+1
                        increments+=1
                        break
            else:
                if found+1 in list_of_indexes[index+1]:
                    found+=1
                    increments+=1
                else:
                   
                    found=-1

        if increments >1:
            first = initial - corresponding_indexes[0]
            last = first + length
            if length < corresponding_indexes[len(corresponding_indexes)-1]:
                last = first + corresponding_indexes[len(corresponding_indexes)-1]
            
            return first, last
        else:
            return -1, -1


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
        splitted_text_with_punctuation = splitted_text_with_punctuation[eliminate_leading_whitespaces(splitted_text_with_punctuation):]
       
        # Step 2: an enter = 3 consecutive whitespaces (" "); find the first occurence of exactly three consecutive whitespaces to see which/where the first word of the first phrase is/starts from
        # Why is it needed? (Challenge behind) when screenshotting the pdf/html/word document, every textual element gets added in the string (e.g., Notepad, Adobe, Acrobat. File, Edit, Save, Exit etc) which
        # is not neccessarily part of the story, therefore these need to be ignored and start the text processing and analysis from the first 'true' word of the story 

        # create a list with all the indexes of the whitespaces in the list of strings 
        def get_whitespace_indexes(iterable, object):
            return (index for index, element in enumerate(iterable) if element == object)
  
        whitespace_indexes = list(get_whitespace_indexes(splitted_text_with_punctuation, ''))

        # find the index of the first 'true' word of the story 
        def index_first_word_of_story(whitespace_indexes): 
            for i in range(len(whitespace_indexes)-2):
                if(whitespace_indexes[i+1]-whitespace_indexes[i] == 1 and whitespace_indexes[i+2]-whitespace_indexes[i+1] == 1):
                    if i and whitespace_indexes[i]-whitespace_indexes[i-1]!=1 and i+3<len(whitespace_indexes) and whitespace_indexes[i+3]-whitespace_indexes[i+2]!=1:
                        return i
  
        index_first_word = whitespace_indexes[index_first_word_of_story(whitespace_indexes)]
        how_much_got_removed = index_first_word - 1 # teoretic ar tb sa fie index_first_word - eliminate_leading_whitespaces(splitted_text_with_punctuation)
        splitted_text = splitted_text_with_punctuation[(index_first_word+3):]

        def get_splitted_clean_text():
            index = 0
            while index<len(splitted_text):
                if splitted_text[index].replace(" ", "") == "": 
                    splitted_text.pop(index) # delete the whitespaces
                else:
                    index+=1
            return splitted_text
        splitted_clean_text = get_splitted_clean_text()

        clean_story_text = ' '.join(splitted_clean_text)

        print(how_much_got_removed)
        print("The indexes of the clean shit:\n", [(value, count) for count, value in enumerate(splitted_clean_text)])

  
        # Step 3: Sentence Detection via spacy, store the sentences in a list; crate a dictionary with keys as sentence indexes and (start_index_of_sentence, end_index_of_sentence) as values

        def get_sentences(text):
            about_text = self.nlp(text) # text is a string including the whole screenshotted story, with the punctuation included
            return list(about_text.sents)
        sentences = get_sentences( ' '.join(splitted_text_with_punctuation)) #get_sentences(clean_story_text)
        #sentences = [sentence for sentence in sentences if not sentence.text.isspace()]
  
        print("sentences\n:")
        for count, sentence in enumerate(sentences):
            print("Sent number", count, " ", sentence, "\n")

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
            return detected_sentence
        # phrase_spoken = ' '.join(phrase_spoken)
  
        detected_sentence, index_sentence = belong_to_which_sentence(phrase_spoken, sentences) # if the long substr by word does not return anything, this line will fail because there will be nothing to unpack, put a try catch or something
        
        gpt3 = MetaverseGenerator(detected_sentence.text)
        gpt3.retrieve_image_from_gpt3OpenAI()

        print("The sentence to be highlighted: ", detected_sentence)
        print(sentence_metadata[index_sentence][0] + how_much_got_removed, sentence_metadata[index_sentence][1] + how_much_got_removed)
        self.highlight(sentence_metadata[index_sentence][0] + how_much_got_removed, sentence_metadata[index_sentence][1] + how_much_got_removed)
        return 
    
        
    
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


    ###################### BORDER WORDS AROUND BOXES BASED ON YOUR PRONOUNTIATION ##############################

    def box_words(self,img, first_word_index, last_word_index, list_of_correctly_pronounced_words): # method accompanying the word_sync() method
        print("THE BOXING STARTS")
        print(list_of_correctly_pronounced_words)
        print(first_word_index, last_word_index)
        rgb_green, rgb_red = (0, 154, 0), (0, 0, 204) 
        for ind in range(last_word_index-first_word_index+1):
            if ind in list_of_correctly_pronounced_words:
                image = cv2.rectangle(img, (self.co_ord_list[ind+first_word_index][1], self.co_ord_list[ind+first_word_index][2]), 
                                (self.co_ord_list[ind+first_word_index][1]+self.co_ord_list[ind+first_word_index][3], 
                                 self.co_ord_list[ind+first_word_index][2]+self.co_ord_list[ind+first_word_index][4]), 
                                 color=rgb_green, thickness= 2) # add green box for correct pronounciation 
            else:
                image = cv2.rectangle(img, (self.co_ord_list[ind+first_word_index][1], self.co_ord_list[ind+first_word_index][2]), 
                                (self.co_ord_list[ind+first_word_index][1]+self.co_ord_list[ind+first_word_index][3], 
                                 self.co_ord_list[ind+first_word_index][2]+self.co_ord_list[ind+first_word_index][4]),
                                color=rgb_red, thickness=2) # add red box for wrong pronounciation 
        cv2.imshow('image', image)
        cv2.waitKey(10000) # change 0 to 60000 for the window to be deleted after 1 min = 60sec

        

