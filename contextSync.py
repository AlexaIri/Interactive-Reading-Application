import spacy
from itertools import islice
import itertools
import nltk, re, pprint
from nltk.collocations import *
from nltk import FreqDist
from pathlib import Path
import math
from collections import defaultdict, Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.text import Text

nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

about_text = """
One day long ago, some sailors set out to sea in their sailing ship. One of
them brought his pet monkey along for the long journey and the monkey was very playful.
When they were far out at sea, a terrible storm overturned their ship.
Everyone fell into the sea, and the monkey was sure that he would drown.
Suddenly a dolphin appeared and picked him up.
They soon reached the island and the monkey came down from the
dolphin’s back. The dolphin asked the monkey, “Do you know this place?”
The monkey replied, “Yes, I do. In fact, the king of the island is my best
friend. Do you know that I am actually a prince?”
Knowing that no one lived on the island, the dolphin said, “Well, well, so
you are a prince! Now you can be a king!”
The monkey asked, “How can I be a king?”

Once upon a time, there was a king who ruled a prosperous country. One day, he went for a trip to
some distant areas of his country. When he was back to his palace, he complained  
that his feet were very painful, because it was the first time that he went
for such a long trip, and the road that he went through was very rough and stony. 

 He then ordered his people to cover every road of the entire country with leather.
Definitely, this would need thousands of cows’ skin, and would cost a
huge amount of money.
Then one of his wise servants dared himself to tell the king, “Why do
you have to spend that unnecessary amount of money? Why don’t you
just cut a little piece of leather to cover your feet?”

"""


# text =  ['', '', '', '', ' ', '', '', '', '_|', 'Story', '-', 'Notepad', '', 'File', 'Edit', 'Format', 'View', 'Help', '', '', '', 'creation', 'desire',"", '', 'pnm','']
input_text= ['', '', '', '', ' ', '', '', '', '_|', 'Story', '-', 'Notepad', '_', '', 'File', 'Edit', 'Format', 'View', 'Help', '', '', '', 'One', 'day', 'long', 'ago,', 'some', 'sailors', 'set', 'out', 'to', 'sea', 'in', 'their', 'sailing', 'ship.', 'One', 'of', '', 'them', 'brought', 'his', 'pet', 'monkey', 'along', 'for', 'the', 'long', 'journey.', '', '', 'When', 'they', 'were', 'far', 'out', 'at', 'sea,', 'a', 'terrible', 'storm', 'overturned', 'their', 'ship.', '', 'Everyone', 'fell', 'into', 'the', 'sea,', 'and', 'the', 'monkey', 'was', 'sure', 'that', 'he', 'would', 'drown.', '', 'Suddenly', 'a', 'dolphin', 'appeared', 'and', 'picked', 'him', 'up.', '', '', 'They', 'soon', 'reached', 'the', 'island', 'and', 'the', 'monkey', 'came', 'down', 'from', 'the', '', '', 'dolphin’s', 'back.', 'The', 'dolphin', 'asked', 'the', 'monkey,', '“Do', 'you', 'know', 'this', 'place?”', '', 'The', 'monkey', 'replied,', '“Yes,', 'I', 'do.', 'In', 'fact,', 'the', 'king', 'of', 'the', 'island', 'is', 'my', 'best', '', 'friend.', 'Do', 'you', 'know', 'that', 'I', 'am', 'actually', 'a', 'prince?”', '', '', 'Knowing', 'that', 'no', 'one', 'lived', 'on', 'the', 'island,', 'the', 'dolphin', 'said,', '“Well,', 'well,', 'so', '', 'you', 'are', 'a', 'prince!', 'Now', 'you', 'can', 'be', 'a', 'king!”', '', '', 'The', 'monkey', 'asked,', '“How', 'can', 'I', 'be', 'a', 'king?”', '', '', '', 'Once', 'upon', 'a', 'time,', 'there', 'was', 'a', 'king', 'who', 'ruled', 'a', 'prosperous', 'country.', 'One', 'day,', 'he', 'went', 'for', 'a', 'trip', 'to', '', 'some', 'distant', 'areas', 'of', 'his', 'country.', 'When', 'he', 'was', 'back', 'to', 'his', 'palace,', 'he', 'complained', '', '', 'that', 'his', 'feet', 'were', 'very', 'painful,', 'because', 'it', 'was', 'the', 'first', 'time', 'that', 'he', 'went', '', '', 'for', 'such', 'a', 'long', 'trip,', 'and', 'the', 'road', 'that', 'he', 'went', 'through', 'was', 'very', 'rough', 'and', 'stony.', '', '', '', 'He', 'then', 'ordered', 'his', 'people', 'to', 'cover', 'every', 'road', 'of', 'the', 'entire', 'country', 'with', 'leather.', '', 'Definitely,', 'this', 'would', 'need', 'thousands', 'of', 'cows’', 'skin,', 'and', 'would', 'cost', 'a', '', '', 'huge', 'amount', 'of', 'money.', '', '', 'Then', 'one', 'of', 'his', 'wise', 'servants', 'dared', 'himself', 'to', 'tell', 'the', 'king,', '“why', 'do', '', '', 'you', 'have', 'to', 'spend', 'that', 'unnecessary', 'amount', 'of', 'money?', 'Why', 'don’t', 'you', '', '', 'just', 'cut', 'a', 'little', 'piece', 'of', 'leather', 'to', 'cover', 'your', 'feet?”', '', '', '', 'Windows', '(CRLF)', 'Ln', '2,', 'Col', '1', '100%', '', '4:09', 'PM', '', '', '', ' ', '', '', '', '1/14/2023']

def phrase_sync(splitted_text_with_punctuation):
  def eliminate_leading_whitespaces(text):
    ind = 0
    while ind<len(text) and (text[ind].isspace() or text[ind] == ""):
        ind += 1
    return ind
  eliminate_leading_whitespaces(splitted_text_with_punctuation)
  
  splitted_text_with_punctuation = splitted_text_with_punctuation[
 eliminate_leading_whitespaces(splitted_text_with_punctuation):]
  
  def get_whitespace_indexes(iterable, object):
    return (index for index, element in enumerate(iterable) if element == object)
  
  whitespace_indexes = list(get_whitespace_indexes(splitted_text_with_punctuation, ''))
  print(whitespace_indexes)
  
  def index_first_word_of_story(whitespace_indexes): 
    for i in range(len(whitespace_indexes)-2):
        if(whitespace_indexes[i+1]-whitespace_indexes[i] == 1 and whitespace_indexes[i+2]-whitespace_indexes[i+1] == 1):
          if i and whitespace_indexes[i]-whitespace_indexes[i-1]!=1 and i+3<len(whitespace_indexes) and whitespace_indexes[i+3]-whitespace_indexes[i+2]!=1:
            return i

  index_first_word = whitespace_indexes[index_first_word_of_story(whitespace_indexes)]

  
  for i in range(index_first_word_of_story(whitespace_indexes)):
    splitted_text_with_punctuation.pop(whitespace_indexes[i]-1)


  print(splitted_text_with_punctuation)
  how_much_got_removed = index_first_word - eliminate_leading_whitespaces(splitted_text_with_punctuation)
     
      
  print(index_first_word, how_much_got_removed)
  
  splitted_clean_text = splitted_text_with_punctuation[(index_first_word+3):]

  # def get_miscellaneous(text):
  #   miscellaneous_chars = list(itertools.repeat ( 0, len(text)))
    
  #   for i in range(len(text)):
  #     if(text[i]==""):
  #       if i>=1:
  #         miscellaneous_chars[i] = 1+ miscellaneous_chars[i-1]
  #       else:
  #         miscellaneous_chars[i] = 1
  #     else:
  #         miscellaneous_chars[i] = miscellaneous_chars[i-1] 
  #   return miscellaneous_chars
      
  # miscellaneous_chars = get_miscellaneous(splitted_clean_text)
  # print("missc", miscellaneous_chars)
  # return splitted_text_with_punctuation
# splitted_clean_text = phrase_sync(input_text)

  index = 0
  while index<len(splitted_clean_text):
      if splitted_clean_text[index].replace(" ", "") == "": 
          splitted_clean_text.pop(index) # delete the whitespaces
      else:
          index+=1
                      
  print("The indexes of the clean shit:\n", [(value, count) for count, value in enumerate(splitted_clean_text)])
  clean_story_text = ' '.join(splitted_clean_text)

  print("Removed: ", how_much_got_removed)
  
  def get_sentences(text):
    about_text = nlp(text)
    return list(about_text.sents)
  sentences = get_sentences(clean_story_text)
  # sentences = [sentence for sentence in sentences if not sentence.text.isspace()]
  
  print(sentences)
  
  def get_dictio_sentences(phrases):
    sentence_metadata = dict()
    for i in range(len(phrases)):
      number_of_words_in_phrase = len(phrases[i].text.split())
      print(number_of_words_in_phrase)
      if not i:
        sentence_metadata[i] = (i, number_of_words_in_phrase-1)
        prev_end = number_of_words_in_phrase
      else:
        sentence_metadata[i] = (prev_end, prev_end+number_of_words_in_phrase-1)
        prev_end += number_of_words_in_phrase
    return sentence_metadata
  
  sentence_metadata =  get_dictio_sentences(sentences)
  print("sentence_metadata\n", sentence_metadata)
  
  def is_sublist(source, target):
      slen = len(source)
      return any(all(item1 == item2 for (item1, item2) in zip(source, islice(target, i, i+slen))) for i in range(len(target) - slen + 1))
  
  def long_substr_by_word(data):
      subseq = []
      data_seqs = [s.split(' ') for s in data]
      if len(data_seqs) > 1 and len(data_seqs[0]) > 0:
          for i in range(len(data_seqs[0])):
              for j in range(len(data_seqs[0])-i+1):
                  if j > len(subseq) and all(is_sublist(data_seqs[0][i:i+j], x) for x in data_seqs):
                      subseq = data_seqs[0][i:i+j]
      return (' '.join(subseq), len(subseq))
  # imi tb un dictionary {index_phrase:(start_index, end_index)} as if it was split - start index, start_index + len(phrase) ca punctuatia nu se lua nici inainte as one word
  
  # data =[
  #   "One day long ago, some sailors set out to sea on th",
  #   "Windows (CRLF) Ln 2, Col 1 100%  4:09 PM         1/14/2023"
  # ]
  # print("printeaza lungimea maxima comuna",long_substr_by_word(data))
  
  def belong_to_which_sentence(phrase_spoken, sentences):
    max_length_similarity, detected_sentence = 0, ''
    for ind_sentence, sentence in enumerate(sentences):
      data = [phrase_spoken, sentence.text]
      if max_length_similarity<long_substr_by_word(data)[1]:
        max_length_similarity, detected_sentence = long_substr_by_word(data)[1], (sentence, ind_sentence)
    return detected_sentence
  # phrase_spoken = ' '.join(phrase_spoken)
  
  detected_sentence, index_sentence = belong_to_which_sentence(" soon reached the island and the monkey came", sentences)
  for i in range(sentence_metadata[index_sentence][0], sentence_metadata[index_sentence][1]):
    print(splitted_clean_text[i])
  print(detected_sentence, index_sentence)

# phrase_sync(input_text)






def contextSync(text_with_punctuation):
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
    return  pos_tagging

  sentences, tokens = tokenization_and_pos_tagging(text_with_punctuation)
  # The lexical richness metric shows us the percentage of distinct words in  the text
  # print(tokens)
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
  
  print("\nStop words we can eliminate from the textual analysis and information extraction", text_without_stop_words, '\n')

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
  # print("Word Frequency: ", word_frequency)
  threshold_to_input_to_gpt = math.floor(gpt_max_tokens/number_words_per_sentence)
  print("Average number of phrases to input into GPT3: ", threshold_to_input_to_gpt,'\n')

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
  
  def extract_subjects_from_sents(sentences): # redo because it's copied fromhttps://subscription.packtpub.com/book/data/9781838987312/2/ch02lvl1sec16/extracting-subjects-and-objects-of-the-sentence
    for sent_no, sentence in enumerate(sentences):
      sentence = nlp(sentence)
      subjects = []
      for token in sentence:
        if ("subj" in token.dep_):
            subtree = list(token.subtree)
            subjects.append(sentence[(subtree[0].i):(subtree[-1].i + 1)]) # there might be multiple subjects in a phrase
      subjects_and_objects[sent_no].append(subjects)
    
  extract_subjects_from_sents(sentences)

  def extract_objects_from_sents(sentences): # redo it's copied
    for sent_no, sentence in enumerate(sentences):
      sentence = nlp(sentence)
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
     return(text_conc.concordance(filter, lines=5))

  for noun in frequent_nouns:
    collocations.extend(extract_collocations(text_without_stop_words, noun))
    print("herehe", type(noun))
    concordances.extend(extract_concordance(text_with_punctuation,noun))

  collocations = [' '.join(elem[0]) for elem in collocations]
    
  print("Collocations:\n", collocations)
  print("\nConcordance:\n", concordances)

  # Input text generator for the GPT3 Metaverse Algorithm
  step = math.floor(threshold_to_input_to_gpt/(2*average_word_length))
  
  for ind in range(0, len(sentences), step):
    start_ind_phrase = ind
    coll_score = 0 
    if(ind+step>len(sentences)):
      end_ind_phrase = start_ind_phrase + len(sentences) % step
    else:
      end_ind_phrase = start_ind_phrase + step - 1
    print("The excerpt to be fed into the GPT3 Algorithm is: \n", ' '.join(sentences[start_ind_phrase:end_ind_phrase]), '\n')

    collocated_text = ' '.join(sents_for_collocation_check[start_ind_phrase:end_ind_phrase])
    for collocation in collocations:
      if(collocated_text.find(collocation)>-1):
        coll_score += 1 
    coll_score *= 100/len(collocations)

    if(coll_score>=30):
      print(f"The collocation strength score ({coll_score}) showcases a meaningful text excerpt. The GPT3 images can be generated successfully!\n")
      # call GPT3
    else:
      print(f"The collocation strength score ({coll_score}) showcases that we should merge two consecutive context extractions. The GPT3 images then can be generated successfully!\n")
      # call GPT3


contextSync(about_text) 