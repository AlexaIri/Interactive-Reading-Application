import cv2

class PronunciationChecker():
    def __init__(self, img, phrase_spoken, input_text_for_sync, co_ord_list):
        self.img = img 
        self.phrase_spoken = phrase_spoken
        self.input_text_for_sync = input_text_for_sync
        self.co_ord_list = co_ord_list

    ######################## PRONUNCIATION CHECKER ALGORITHM #################################


    def pronunciation_checker(self):

        self.phrase_spoken = self.phrase_spoken['text'].split()
        self.input_text_for_sync = ' '.join(self.input_text_for_sync).lower().split() 
        
        cleaned_list_of_words = [s.replace("?","").replace(",","").replace(".","").replace('"','').replace("!","").replace(":","").replace(";","") for s in self.input_text_for_sync]
        
        print("FINAL indexes associated with each word from ss:\n", [(value, count) for count, value in enumerate(cleaned_list_of_words)])
        print("Comparing the spoken phrase:", self.phrase_spoken)
        
        matched_indexes_of_words = []
        indexes_of_correctly_pronounced_words = [] 
        for word_index in range(len(self.phrase_spoken)): 
            if self.phrase_spoken[word_index] in cleaned_list_of_words:
                matched_indexes_of_words.append([i for i,val in enumerate(cleaned_list_of_words) if val==self.phrase_spoken[word_index]])
               
                indexes_of_correctly_pronounced_words.append(word_index)
        print("indexes of our text ss: ", matched_indexes_of_words)
        print("indexes of our phrase: ", indexes_of_correctly_pronounced_words)
        first, last = self.get_indexes(matched_indexes_of_words, indexes_of_correctly_pronounced_words, len(self.phrase_spoken))
        print("first and last index:", first, last, '\n')

        if first!= -1:
            print("The portion to be highlighted in the story:" + " ".join(self.input_text_for_sync[first:last]))
            #self.highlight(first, last-1) 
            self.box_words(first, last-1, indexes_of_correctly_pronounced_words) 
            
            
    def get_indexes(self,list_of_indexes, corresponding_indexes, length):
        if len(list_of_indexes) == 0:
            return -1, -1
        if len(list_of_indexes) == 1:
            if len(list_of_indexes[0])==1:                 
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

    def box_words(self, first_word_index, last_word_index, list_of_correctly_pronounced_words): # method accompanying the word_sync() method
        print("THE BOXING STARTS")
        print(list_of_correctly_pronounced_words)
        print(first_word_index, last_word_index)
        rgb_green, rgb_red = (0, 154, 0), (0, 0, 204) 
        for ind in range(last_word_index-first_word_index+1):
            if ind in list_of_correctly_pronounced_words:
                image = cv2.rectangle(self.img, (self.co_ord_list[ind+first_word_index][1], self.co_ord_list[ind+first_word_index][2]), 
                                (self.co_ord_list[ind+first_word_index][1]+self.co_ord_list[ind+first_word_index][3], 
                                 self.co_ord_list[ind+first_word_index][2]+self.co_ord_list[ind+first_word_index][4]), 
                                 color=rgb_green, thickness= 2) # add green box for correct pronounciation 
            else:
                image = cv2.rectangle(self.img, (self.co_ord_list[ind+first_word_index][1], self.co_ord_list[ind+first_word_index][2]), 
                                (self.co_ord_list[ind+first_word_index][1]+self.co_ord_list[ind+first_word_index][3], 
                                 self.co_ord_list[ind+first_word_index][2]+self.co_ord_list[ind+first_word_index][4]),
                                color=rgb_red, thickness=2) # add red box for wrong pronounciation 
        cv2.imshow('image', image)
        cv2.waitKey(10000) # change 0 to 60000 for the window to be deleted after 1 min = 60sec