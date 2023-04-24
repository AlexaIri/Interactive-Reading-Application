import cv2

######################## PRONUNCIATION CHECKER API #################################

class PronunciationChecker():
    def __init__(self, img, phrase_spoken, input_text_for_sync, co_ord_list):
        self.img = img 
        self.phrase_spoken = phrase_spoken
        self.input_text_for_sync = input_text_for_sync
        self.co_ord_list = co_ord_list

    ######################## ALGORITHM #################################

    def pronunciation_checker(self):

        self.phrase_spoken = self.phrase_spoken['text'].split()
        self.input_text_for_sync = ' '.join(self.input_text_for_sync).lower().split() 
        
        cleaned_list_of_words = [s.replace("?","").replace(",","").replace(".","").replace('"','').replace("!","").replace(":","").replace(";","") for s in self.input_text_for_sync]
        
        print("Indexes associated with each word from screenshot:\n", [(value, count) for count, value in enumerate(cleaned_list_of_words)])
        print("The spoken phrase transcribed by Vosk:", self.phrase_spoken)
        
        matched_indexes_of_words = []
        indexes_of_correctly_pronounced_words = [] 
        for word_index in range(len(self.phrase_spoken)): 
            if self.phrase_spoken[word_index] in cleaned_list_of_words:
                matched_indexes_of_words.append([i for i,val in enumerate(cleaned_list_of_words) if val==self.phrase_spoken[word_index]])
               
                indexes_of_correctly_pronounced_words.append(word_index)
                
        first,last = self.longest_subsequence_of_consecutive_indices(matched_indexes_of_words, indexes_of_correctly_pronounced_words, len(self.phrase_spoken))
        print(first, last)
        if first!= -1:
            print("The portion to be highlighted in the story:" + " ".join(self.input_text_for_sync[first:last]))
            identified_portion_of_text = self.input_text_for_sync[first:last]            
            self.box_words(first, last-1, indexes_of_correctly_pronounced_words, identified_portion_of_text) 
            
    # Text Analysis
    def longest_subsequence_of_consecutive_indices(self,list_of_indexes, corresponding_indexes, jump):
        if len(list_of_indexes) == 0:
            return -1, -1
        if len(list_of_indexes) == 1:
            if len(list_of_indexes[0])==1:      
                return list_of_indexes[0][0] - corresponding_indexes[0],  list_of_indexes[0][0] - corresponding_indexes[0]+jump
        
        lower_boundary, upper_boundary  = -1, -1
        for index in range(len(list_of_indexes)-1):
            if upper_boundary==-1:
                for x in list_of_indexes[index]:
                    if x+1 in list_of_indexes[index+1]:
                        if lower_boundary == -1:
                            lower_boundary = x
                        upper_boundary = x+1
                        break
            else:
                if upper_boundary+1 in list_of_indexes[index+1]:
                    upper_boundary+=1
                else:
                    upper_boundary=-1

        last_index_patial =  jump
        if jump < corresponding_indexes[len(corresponding_indexes)-1]:
            last_index_patial =  corresponding_indexes[len(corresponding_indexes)-1]
            
        return  (lower_boundary - corresponding_indexes[0], last_index_patial + lower_boundary - corresponding_indexes[0]) if upper_boundary-lower_boundary >1 else (-1,-1)
        
     ###################### BORDER WORDS AROUND BOXES BASED ON YOUR PRONOUNTIATION ##############################

    def box_words(self, first_word_index, last_word_index, list_of_correctly_pronounced_words, identified_text): # method accompanying the word_sync() method
        print("Bordering words around boxes starts here:")
        rgb_green, rgb_red, rgb_amber = (0, 154, 0), (0, 0, 204), (0, 191, 255)
        print(self.phrase_spoken)
        for ind in range(last_word_index-first_word_index+1):
            actual_word, spoken_word = identified_text[ind], self.phrase_spoken[ind]
            print(actual_word, spoken_word)
            pronunciation_score = 100*len(set(actual_word).intersection(spoken_word))/len(actual_word)
            if pronunciation_score == 100 and ind in list_of_correctly_pronounced_words:
                image = cv2.rectangle(self.img, (self.co_ord_list[ind+first_word_index][1], self.co_ord_list[ind+first_word_index][2]), 
                                (self.co_ord_list[ind+first_word_index][1]+self.co_ord_list[ind+first_word_index][3], 
                                self.co_ord_list[ind+first_word_index][2]+self.co_ord_list[ind+first_word_index][4]), 
                                color=rgb_green, thickness= 2) # add green box for correct pronounciation 
            elif pronunciation_score > 80:
                image = cv2.rectangle(self.img, (self.co_ord_list[ind+first_word_index][1], self.co_ord_list[ind+first_word_index][2]), 
                                (self.co_ord_list[ind+first_word_index][1]+self.co_ord_list[ind+first_word_index][3], 
                                 self.co_ord_list[ind+first_word_index][2]+self.co_ord_list[ind+first_word_index][4]),
                                color=rgb_amber, thickness=2) # add amber box for partially_corect pronounciation 
            else:
                image = cv2.rectangle(self.img, (self.co_ord_list[ind+first_word_index][1], self.co_ord_list[ind+first_word_index][2]), 
                                (self.co_ord_list[ind+first_word_index][1]+self.co_ord_list[ind+first_word_index][3], 
                                 self.co_ord_list[ind+first_word_index][2]+self.co_ord_list[ind+first_word_index][4]),
                                color=rgb_red, thickness=2) # add red box for wrong pronounciation 
        cv2.imshow('image', image)
        cv2.waitKey(10000) 