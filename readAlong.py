from kita import Kita
from book_preprocessor import StoryBookPreprocessor
import os

MAIN_FOLDER_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__))))
VOSK_PATH = os.path.join(MAIN_FOLDER_PATH, 'models', 'vosk')
STORIES_LIBRARY = os.path.join(MAIN_FOLDER_PATH,  "Story Library")

# The ReadAlong API (with Karaoke Reading Tracker and Pronunciation  Checker) is powered by KITA
speech_engine = Kita("English(US)", "vosk", VOSK_PATH) 
print("The call to the Kita API was successful!")
speech_engine.reading_with_kita()

# Metaverse Back-end (with Book Preprocessor, Image Generator and Metaverse Generator) is powered by OpenAI's GPT 3 - DALLE 2
for story in os.listdir(STORIES_LIBRARY):
   if os.path.isfile(os.path.join(STORIES_LIBRARY, story)):
       # Render the images for a book which doesn't have them generated yet
       if not os.path.exists(os.path.join(MAIN_FOLDER_PATH, 'JSON Images Filestore', story)): 
        PDF_STORY_PATH = os.path.join(STORIES_LIBRARY, story)
        metaverse = StoryBookPreprocessor(PDF_STORY_PATH, story) 
        metaverse.process_text_load_metaverse() 

