import sys 
from kita import Kita
from storyBookPreprocessor import StoryBookPreprocessor
import os

MAIN_FOLDER_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__))))
VOSK_PATH = os.path.join(MAIN_FOLDER_PATH, 'models', 'vosk')
STORIES_LIBRARY = os.path.join(MAIN_FOLDER_PATH,  "Story Library")

#lang = sys.argv[1]
#model = sys.argv[2]
#mode = sys.argv[3]
#model_path = sys.argv[4]

# For the karaoke reading powered by the speech engine
speech_engine = Kita("English(US)", "vosk", VOSK_PATH) 
print("The call to the Kita API was successful!")
speech_engine.reading_with_kita()

# For preprocessing the books - retrieve and store the images
for story in os.listdir(STORIES_LIBRARY):
   if os.path.isfile(os.path.join(STORIES_LIBRARY, story)):
       if os.path.exists(os.path.join(MAIN_FOLDER_PATH, 'JSON Images Filestore', story)) == False: # the book doesn't have the images generated yet
        PDF_STORY_PATH = os.path.join(STORIES_LIBRARY, story)
        metaverse = StoryBookPreprocessor(PDF_STORY_PATH, story) 
        metaverse.process_text_load_metaverse() 

