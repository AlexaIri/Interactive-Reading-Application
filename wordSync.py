import queue
import cv2
from gpt3Api import MetaverseGenerator

class WordSync():
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
    
    ######################## WORD SYNC ALGORITHM #################################


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
            self.box_words(img, first, last-1, indexes_of_correctly_pronounced_words) 
            


    ######################## GET THE INDEXES OF THE SPOKEN PHRASE AND THE MATCHED SEQUENCE #####################
        
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

        

