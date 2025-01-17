import pyautogui
from itertools import islice
from nltk.collocations import *
import spacy

######################## KARAOKE READING TRACKER API #################################

class ReadingTracker():
    def __init__(self, phrase_spoken):
        self.phrase_spoken = phrase_spoken
        self.nlp = spacy.load("en_core_web_sm")
        
    def reading_tracker(self, splitted_text_with_punctuation, co_ord_list):  # say a few words and see from which phrase they come from, and highlight everything from the start of the sentence to where it ends which is where the punctuation mark is ('.!?;:')
        print("phrase spoken\n", self.phrase_spoken)

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

  
        # Step 3: Sentence Detection via spacy, store the sentences in a list; 
        # create a dictionary with keys as sentence indexes and (start_index_of_sentence, end_index_of_sentence) as values

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
  
        # Step 4: Detect the Longest Common Contiguous Subsequence of the phrase_said and the
        # sentences generated via spacy to determine the sentence which the spoken phrase pertains

        def is_sublist(source, target):
            slen = len(source)
            return any(all(item1 == item2 for (item1, item2) in zip(source, islice(target, i, i+slen))) for i in range(len(target) - slen + 1))

        def longest_common_sequence(data):
            data_seqs = [s.split(' ') for s in data]
            subseq = find_longest_subsequence(data_seqs)
            return (' '.join(subseq), len(subseq))

        def find_longest_subsequence(data_seqs):
            if len(data_seqs) == 0 or len(data_seqs[0]) == 0:
                return []
            longest_subseq = []
            for i in range(len(data_seqs[0])):
                for j in range(len(data_seqs[0])-i+1):
                    subseq = data_seqs[0][i:i+j]
                    if j > len(longest_subseq) and all(is_sublist(subseq, x) for x in data_seqs):
                        longest_subseq = subseq
            return longest_subseq

        def is_sublist(subseq, seq):
            n = len(subseq)
            return any((subseq == seq[i:i+n]) for i in range(len(seq)-n+1))

    
        def belong_to_which_sentence(phrase_spoken, sentences):
            max_jump_similarity, detected_sentence = 0, ''
            for ind_sentence, sentence in enumerate(sentences):
                data = [phrase_spoken, sentence.text]
                print(longest_common_sequence(data)[1])
                if max_jump_similarity<longest_common_sequence(data)[1]:
                    max_jump_similarity, detected_sentence = longest_common_sequence(data)[1], (sentence, ind_sentence)
            if(detected_sentence == ''):
                return -1
            return detected_sentence

        try:
            detected_sentence, index_sentence = belong_to_which_sentence(self.phrase_spoken, sentences) 
            print("The sentence to be highlighted: ", detected_sentence)
            self.highlight(sentence_metadata[index_sentence][0], sentence_metadata[index_sentence][1], co_ord_list)
            return detected_sentence, index_sentence 
        except:
            print("Nothing to highlight in the story! Try reading something else!")
            
        
    ###################### HIGHLIGHT A SEQUENCE OF WORDS ##############################
    def highlight(self, first_word_index, last_word_index, co_ord_list):
        print("The highlighting on the page starts\n")
        pyautogui.click(x=co_ord_list[first_word_index][1], y=co_ord_list[first_word_index][2], duration = 0.1)
        pyautogui.keyDown('shift') # press the key
        pyautogui.keyDown('ctrl') 
        pyautogui.dragTo(x=co_ord_list[last_word_index][1]+co_ord_list[last_word_index][3], 
                         y=co_ord_list[last_word_index][2]+co_ord_list[last_word_index][4],  duration = 0.5)
        pyautogui.click()
        pyautogui.keyUp('shift') # release the key
        pyautogui.keyUp('ctrl') 
