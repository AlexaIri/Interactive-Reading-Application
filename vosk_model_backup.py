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
    def karaoke_reading(self): 
        rec,samplerate = self.setUp()
        try: 

            with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=None, dtype='int16', channels=1, callback=self._callback):
                initial = time.perf_counter()
                img = pyautogui.screenshot()
                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

                words = pytesseract.image_to_string(img, lang="eng", config="--psm 6") #"eng" needs to be replaced by variable
                #print("words generated from ss:", words)

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
                    #if(time_of_latest_ss - initial > 300):
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
                    #    initial = time_of_latest_ss
                    #MAYBE TAKE A NEW SCREENSHOT IF U SCROLL THE PAGE, aka RETAIN THE FIRST SENTENCE ON EACH PAGE, IF IT IS DIFFERENT, THEN TAKE A NEW SS
                    data = self.q.get()
                    if rec.AcceptWaveform(data):
                        d = json.loads(rec.Result())
                    else:
                        d = json.loads(rec.PartialResult())
                    for key in d.keys():
                        if d[key]:
                            if d[key] != self.previous_line or key == 'text':
                                
                                if 'text' in d:
                                    #WordSync(cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR), d, input_text_for_sync, self.co_ord_list).word_sync()
                                    #self.word_sync(cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR), d, input_text_for_sync) 
                                    self.phrase_sync(d, input_text_for_sync)
                                    #PhraseSync(d, input_text_for_sync).phrase_sync()

                              
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
        cleaned_words = ' '.join(cleaned_words).lower().split() 
        print("brfrg", cleaned_words)
        
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







    ######################## CONTEXT SYNC ALGORITHM #################################


    def contextSync(self, text_with_punctuation):
        gpt_max_tokens = 1500
        '''
        Preprocess the text for a foundational step in information extraction, namely entity detection.
        To achieve this, we will follow three initial procedures: (1) sentence segmentation, (2) word tokenization, and (3) part-of-speech tagging process.
        ''' 
        def tokenization_and_pos_tagging(text):
            sentences = nltk.sent_tokenize(text)
            tokens = [nltk.word_tokenize(sentence) for sentence in sentences] 
            return sentences, tokens
    
        def pos_tagging(tokens):
            return [nltk.pos_tag(token) for token in tokens]

        sentences, tokens = tokenization_and_pos_tagging(text_with_punctuation)

        # The lexical richness metric shows us the percentage of distinct words in  the text
        def lexical_diversity(text): 
            return len(set(text)) / len(text) 

        lexical_diversity = lexical_diversity(text_with_punctuation)
        print(f'{format(lexical_diversity*100,".2f")}% unique words')
  
        text_without_punctuation = ''.join(filter(lambda x: x not in '".,;!?-', text_with_punctuation))
        splitted_text_without_punctuation = [word for word in text_without_punctuation.split() if word.isalpha()]

        # The stop words usually occur frequently at any text level, yet they can negatively impact the context analysis and information extraction since they do not possess a powerful meaning
        def stop_words(words_without_punct, words_with_punct):
            sents_for_collocation_check = []
            stop_words = nltk.corpus.stopwords.words("english")
            without_punct =  [w for w in words_without_punct if w.lower() not in stop_words]
            for sent in words_with_punct:
                sents_for_collocation_check.append(' '.join([w for w in sent if w.lower() not in stop_words]))
            return without_punct, sents_for_collocation_check
    
        text_without_stop_words, sents_for_collocation_check = stop_words(splitted_text_without_punctuation, tokens)
  
        print("\n The passage tokens after we eliminated the stop words for a meaningful textual analysis:\n", text_without_stop_words, '\n')

        def textual_metrics(words, sentences):
            average_word_length = sum(map(len, words))/len(words)
            average_sentence_length = sum(map(len, sentences))/len(sentences)
            avg_number_words_per_sentence = len(splitted_text_without_punctuation)/len(sentences)
            word_frequency = Counter(text_without_stop_words) # we will not take into consideration the stopwords since they are the most frequently occurring, yet counting their occurence does not bolster the analysis
            return average_word_length, average_sentence_length, avg_number_words_per_sentence, word_frequency
  
        average_word_length, average_sentence_length, number_words_per_sentence, word_frequency = textual_metrics(splitted_text_without_punctuation, sentences)
  
        print("Average Word Length: ", average_word_length)
        print("Average Sentence Length: ", average_sentence_length)
        print("Average Number of Words per Sentence: ", number_words_per_sentence)
        print("Word Frequency: ", word_frequency, '\n')
        threshold_to_input_to_gpt = math.floor(gpt_max_tokens/number_words_per_sentence)
        print("Maximum number of phrases to input into the GPT3 algorithm: ", threshold_to_input_to_gpt,'\n')

        # Determine the most frequently utilised nouns 
        frequent_nouns = set()
        pos_tagging_tokens = pos_tagging(tokens)
        for sent_no, splitted_words in enumerate(pos_tagging_tokens):
            for token in splitted_words:
                if token[1] in ['NN', 'NNS'] and word_frequency[token[0]]>=2:
                    frequent_nouns.add(token[0])

        print("Most most frequently utilised nouns: ", frequent_nouns, '\n')
  
        '''
        Implement Entity Recognition 
  
        Extract the subjects, objects, and actions from the text based on the word frequency dictionary excluding stop words
  
        Create a subjects and objects dictionary whose keys are the indexes of each phrase, and the values are two lists, first representing the subjects of the phrase and the second representing the objects of the phrase
        '''
  
        subjects_and_objects = defaultdict(list) 
  
        def extract_subjects_from_sents(sents): # redo because it's copied fromhttps://subscription.packtpub.com/book/data/9781838987312/2/ch02lvl1sec16/extracting-subjects-and-objects-of-the-sentence
            for sent_no, sentence in enumerate(sents):
                sentence = self.nlp(sentence)
                subjects = []
                for token in sentence:
                    if ("subj" in token.dep_):
                        subtree = list(token.subtree)
                        subjects.append(sentence[(subtree[0].i):(subtree[-1].i + 1)]) # there might be multiple subjects in a phrase
                subjects_and_objects[sent_no].append(subjects)
    
        extract_subjects_from_sents(sentences)

        def extract_objects_from_sents(sents): # redo it's copied
            for sent_no, sentence in enumerate(sents):
                sentence = self.nlp(sentence)
                objects = []
                for token in sentence:
                    if ("dobj" in token.dep_):
                        subtree = list(token.subtree)
                        objects.append(sentence[subtree[0].i:(subtree[-1].i + 1)]) # there might be multiple objects in a phrase
                subjects_and_objects[sent_no].append(objects)
    
        extract_objects_from_sents(sentences)
  
        print("Subjects and objects per sentence:\n", subjects_and_objects, '\n')
    
        # Implement Relation Extraction 
  
        # Extract the co-references between the nouns and the pronouns in the text - https://stackoverflow.com/questions/62735456/understanding-and-using-coreference-resolution-stanford-nlp-tool-in-python-3-7
  
        # Extract concordance and collocations with bi-grams for context comprehension 
        collocations, concordances = [], []

        def extract_collocations(text, filter):
            finder = nltk.collocations.BigramCollocationFinder.from_words(text) 
            filter = lambda *w: noun not in w
            finder.apply_ngram_filter(filter) # apply filter based on the most frequent nouns
            return(finder.ngram_fd.most_common(3))
    

        def extract_concordance(text, filter):
            text_conc = Text(word_tokenize(text))
            return(text_conc.concordance(filter))

        for noun in frequent_nouns:
            collocations.extend(extract_collocations(text_without_stop_words, noun))
        # concordances.extend(extract_concordance(text_without_stop_words,noun))

        collocations = [' '.join(elem[0]) for elem in collocations]
    
        print("Collocations:\n", collocations)
        print("\nConcordance:\n", concordances)

        # Input text generator for the GPT3 Metaverse Algorithm
        step = math.floor(threshold_to_input_to_gpt/(4*average_word_length))
  
        for ind in range(0, len(sentences), step):
            start_ind_phrase = ind
            coll_score = 0 
            if(ind+step>len(sentences)):
                end_ind_phrase = start_ind_phrase + len(sentences) % step
            else:
                end_ind_phrase = start_ind_phrase + step - 1
            passage = ' '.join(sentences[start_ind_phrase:end_ind_phrase])
            print("The excerpt to be fed into the GPT3 Algorithm is: \n", passage, '\n')

            # Find keywords for image metadata
            keywords = set(word for word in frequent_nouns if(passage.find(word))!=-1)

            image_metadata = (start_ind_phrase, end_ind_phrase, keywords)
            print("Image metadata: ", image_metadata, '\n')

            gpt3 = ImageGenerator(passage)
            gpt3.retrieve_image_from_gpt3OpenAI()

            collocated_text = ' '.join(sents_for_collocation_check[start_ind_phrase:end_ind_phrase])
            for collocation in collocations:
                if(collocated_text.find(collocation)>-1):
                    coll_score += 1 
            coll_score *= 100/len(collocations)

            if(coll_score>=30):
                print(f"The collocation strength score ({coll_score}) showcases a meaningful text excerpt. The GPT3 images can be generated successfully!\n")
                # call GPT3
            else:
                print(f"The collocation strength score ({coll_score}) showcases that we should merge two consecutive context extractions or generate a series of images instead of just 1. The GPT3 images can be generated successfully!\n")
                # call GPT3