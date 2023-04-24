# Class implemented for testing purposes

from kita import Kita
from storyBookPreprocessor import StoryBookPreprocessor
from metaverse_generator import MetaverseGenerator
import json
import os

# Add absolute paths 
MAIN_FOLDER_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__))))
VOSK_PATH = os.path.join(MAIN_FOLDER_PATH, 'models', 'vosk')
STORIES_LIBRARY = os.path.join(MAIN_FOLDER_PATH,  "Story Library")

# Call Kita
kita = Kita("English(US)", "vosk", VOSK_PATH) 

# Load the configuration settings
with open('configuration.json') as json_file:
    data = json.load(json_file)

selected_story_book = data['story_book']

# Call the ReadAlong Back-end (i.e., Karaoke Reading Tracker and Pronunciation Checker) via the KITA API
kita.reading_with_kita()

 #Preprocess the newly added story books and generate illustrations for each of them 
for story in os.listdir(STORIES_LIBRARY):
   if os.path.isfile(os.path.join(STORIES_LIBRARY, story)):
       if os.path.exists(os.path.join(MAIN_FOLDER_PATH, 'JSON Images Filestore', story)) == False:
        PDF_STORY_PATH = os.path.join(STORIES_LIBRARY, story)
        metaverse = StoryBookPreprocessor(PDF_STORY_PATH, story) 
        metaverse.process_text_load_metaverse() 


# Call the Metaverse Back-end (i.e., Metaverse Generator, Image Generator) to display the metaverse for the selected story book
PDF_STORY_PATH = os.path.join(STORIES_LIBRARY, selected_story_book)
metaverse = MetaverseGenerator(PDF_STORY_PATH, selected_story_book) 
#metaverse.read_story_by_page() 
