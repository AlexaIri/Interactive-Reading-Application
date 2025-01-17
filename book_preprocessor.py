import fitz
from pathlib import Path
import nltk, re
from nltk.collocations import *
from pathlib import Path
import math
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.text import Text
from image_generator import GPT3_DALLE2_Image_Generator
import os
import spacy

class StoryBookPreprocessor():

    def __init__(self, story_book_path, story_book_name):
        self.story_book_path = story_book_path 
        self.story_book_name = story_book_name
        self.json_data_dir = Path.cwd() / "json_image_filestore"        
        self.png_data_dir = Path.cwd() / "png_image_filestore"
        
        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')

    # Clean and pre-process input text
    def clean_text(self, text):
        text = re.sub(r'100 Moral Stories', "", text)
        text = re.sub(r'Stories', "", text)

        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'Free eBooks at Planet eBook.com','', text)
        text = re.sub(r'^(www)?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) # nu merge
        text = re.sub(r'www.islamicoccasions.com', "", text)
        return text

    def delete_from_text(self, list_indexes, tokens):
        for index in sorted(list_indexes, reverse=True):
            del tokens[index]

    def process_text(self, splitted_text_with_punctuation):

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
    

    ######################## CONTEXT ANALYSIS ALGORITHM TO FEED STORY TEXT INTO OPENAI DALLE2-GPT3 #################################

    def get_index(self, dir):
        return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])


    def preprocessMetaversePerBook(self, text_with_punctuation, page_no):
        gpt_max_tokens = 1500
        
        '''
        Preprocess the text for a foundational step in information extraction, namely entity detection.
        To achieve this, we will follow three initial procedures: (1) sentence segmentation, (2) word tokenization, 
        and (3) part-of-speech tagging process.
        ''' 
        if(text_with_punctuation):

            def tokenization(text):
                sentences = nltk.sent_tokenize(text)
                tokens = [nltk.word_tokenize(sentence) for sentence in sentences] 
                return sentences, tokens
            
            def pos_tagging(tokens):
                 return [nltk.pos_tag(token) for token in tokens]

            sentences, tokens = tokenization(text_with_punctuation)

            # The lexical richness metric shows the diversity in the vocabulary
            def lexical_diversity(text): 
                return len(set(text)) / len(text) 

            lexical_diversity = lexical_diversity(text_with_punctuation)
            print(f'{format(lexical_diversity*100,".2f")}% unique words')
  
            text_without_punctuation = ''.join(filter(lambda x: x not in '".,;!?-', text_with_punctuation))
            splitted_text_without_punctuation = [word for word in text_without_punctuation.split() if word.isalpha()]

            # The stop words usually occur frequently at any text level, yet they can negatively impact
            # the context analysis and information extraction since they do not possess a powerful meaning
            def stop_words(words_without_punct, words_with_punct):
                sents_for_collocation_check = []
                stop_words = nltk.corpus.stopwords.words("english")
                without_punct =  [w for w in words_without_punct if w.lower() not in stop_words]
                for sent in words_with_punct:
                    sents_for_collocation_check.append(' '.join([w for w in sent if w.lower() not in stop_words]))
                return without_punct, sents_for_collocation_check
    
            text_without_stop_words, sents_for_collocation_check = stop_words(splitted_text_without_punctuation, tokens)
  
            # print("\n The passage tokens after we eliminated the stop words for a meaningful textual analysis:\n", text_without_stop_words, '\n')
            if(splitted_text_without_punctuation):
                def textual_metrics(words, sentences):
                    average_word_size = sum(map(len, words))/len(words)
                    average_sentence_size = sum(map(len, sentences))/len(sentences)
                    avg_number_words_per_sentence = len(splitted_text_without_punctuation)/len(sentences)
                    word_frequency = Counter(text_without_stop_words) # we will not take into consideration the stopwords since they are the most frequently occurring, yet counting their occurence does not bolster the analysis
                    return average_word_size, average_sentence_size, avg_number_words_per_sentence, word_frequency
  
                average_word_size, average_sentence_size, number_words_per_sentence, word_frequency = textual_metrics(splitted_text_without_punctuation, sentences)
  
                #print("Average Word size: ", average_word_size)
                #print("Average Sentence size: ", average_sentence_size)
                #print("Average Number of Words per Sentence: ", number_words_per_sentence)
                #print("Word Frequency: ", word_frequency, '\n')
                threshold_to_input_to_gpt = math.floor(gpt_max_tokens/number_words_per_sentence)
                print("Maximum number of sentences to input into the GPT3 algorithm: ", threshold_to_input_to_gpt,'\n')

                # Determine the most frequently utilised nouns, i.e., the keywords depicting an image
                frequent_nouns = set()
                pos_tagging_tokens = pos_tagging(tokens)
                for splitted_words in pos_tagging_tokens:
                    for token in splitted_words:
                        if token[1] in ['NN', 'NNS'] and word_frequency[token[0]]>=2:
                            frequent_nouns.add(token[0])

                #print("Most most frequently utilised nouns: ", frequent_nouns, '\n')
  
                '''
                Implement Entity Recognition 
  
                Extract the subjects, objects, and actions from the text based on the word frequency dictionary excluding stop words
  
                Create a subjects and objects dictionary whose keys are the indexes of each sentence, and the values are two lists, first representing the subjects of the sentence and the second representing the objects of the sentence
                '''
  
                subjects_and_objects = defaultdict(list) 

                '''
                based on:  https://subscription.packtpub.com/book/data/9781838987312/2/ch02lvl1sec16/extracting-subjects-and-objects-of-the-sentence
                '''
                def extract_subjects_from_sentences(sentences):
                    for sent_no, sentence in enumerate(sentences):
                        sentence = self.nlp(sentence)
                        subjects = []
                        for token in sentence:
                            if ("subj" in token.dep_):
                                subtree = list(token.subtree)
                                subjects.append(sentence[(subtree[0].i):(subtree[-1].i + 1)]) # there might be multiple subjects in a sentence
                        subjects_and_objects[sent_no].append(subjects)
    
                extract_subjects_from_sentences(sentences)

                def extract_objects_from_sentences(sentences):
                    for sent_no, sentence in enumerate(sentences):
                        sentence = self.nlp(sentence)
                        objects = []
                        for token in sentence:
                            if ("dobj" in token.dep_):
                                subtree = list(token.subtree)
                                objects.append(sentence[subtree[0].i:(subtree[-1].i + 1)]) # there might be multiple objects in a sentence
                        subjects_and_objects[sent_no].append(objects)
    
                extract_objects_from_sentences(sentences)
  
                #print("Subjects and objects per sentence:\n", subjects_and_objects, '\n')
    
                # Extract concordances and collocations using bi-grams for context comprehension 
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
                    concordances.extend(extract_concordance(text_without_stop_words,noun))

                collocations = [' '.join(elem[0]) for elem in collocations]
    
                #print("Collocations:\n", collocations)
                #print("\nConcordance:\n", concordances)

                # Input text generator for the GPT3 Metaverse Algorithm
                step = math.floor(threshold_to_input_to_gpt/(9*int(average_word_size)))
                for ind in range(0, len(sentences), step):
                    
                    start_ind_sentence = ind
                    coll_score, conc_score = 0, 0
                    if(ind+step>len(sentences)):
                        end_ind_sentence = start_ind_sentence + len(sentences) % step
                    else:
                        end_ind_sentence = start_ind_sentence + step - 1
                    passage = ' '.join(sentences[start_ind_sentence:(end_ind_sentence+1)]) 
                    print("The excerpt to be fed into the GPT3 Algorithm is: \n", passage, '\n')
                    print(start_ind_sentence, end_ind_sentence)

                    # Find keywords for image metadata
                    keywords = list(word for word in frequent_nouns if(passage.find(word))!=-1)

                    image_metadata = (start_ind_sentence, end_ind_sentence, keywords, page_no)
                    print("Image metadata: ", image_metadata, '\n')
                  
                    # Generate images on the current page
                    gpt3_dalle2_Image_Generator = GPT3_DALLE2_Image_Generator(passage, image_metadata, self.story_book_name)
                    gpt3_dalle2_Image_Generator.retrieve_image_from_OpenAI()

                    try: 
                        collocated_text = ' '.join(sents_for_collocation_check[start_ind_sentence:end_ind_sentence])
                        for collocation in collocations:
                            if(collocated_text.find(collocation)>-1):
                                coll_score += 1 
                        coll_score *= 100/len(collocations)

                        for concordance in concordances:
                            if(collocated_text.find(concordance)>-1):
                                conc_score += 1 
                        conc_score *= 100/len(concordances)

                        if((coll_score>=30 and conc_score>=30) or (conc_score*100/coll_score<50)):
                            print(f"The collocation strength score ({coll_score}) showcases a meaningful text excerpt. The GPT3 images can be generated successfully!\n")
                 
                        else:
                            print(f"The collocation strength score ({coll_score}) showcases that we should merge two consecutive context extractions or generate a series of images instead of just 1. The GPT3 images can be generated successfully!\n")
                    except:
                        print("Exception thrown when determining collocations and concordances")
                        
    # Process the content of the story book per page and per chapter
    def process_text_load_metaverse(self):
        with fitz.open(self.story_book_path) as doc:
               
            story_text = ''
            for page_no, page in enumerate(doc):
                text = page.get_text()

                try:
                    text = text[text.find('Chapter') + len('Chapter') + 3:] 
                except:
                    print("This story book is not structured on chapters")

                text = self.clean_text(text)
                    
                self.preprocessMetaversePerBook(text, page_no)
                     
                story_text += page.get_text()
                   
            story_text = story_text[story_text.find('Chapter') + len('Chapter') + 2:]
            story_text = self.clean_text(story_text)

            chapter_no = -1 
            while(story_text.find('Chapter')!=-1):
                chapter = story_text[:story_text.find('Chapter')]
                chapter_no += 1
                #print("Start of a new chapter\n\n", chapter, '\n')
                story_text = story_text[story_text.find('Chapter') + len('Chapter') + 2:]
                
                # Generate Metaverse and Proceed with the Context Analyis
                self.preprocessMetaversePerBook(chapter, chapter_no)
                

                # Append the content of the last chapter 
            print("Last chapter\n", story_text)
            self.preprocessMetaversePerBook(story_text, chapter_no+1)
