import queue
import pyautogui
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


    ######################## PHRASE SYNC ALGORITHM #################################


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
        sentences = get_sentences(clean_story_text)
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