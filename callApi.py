# for testing purposes

from kita import Kita
from storyBookPreprocessor import StoryBookPreprocessor
import json
import os

MAIN_FOLDER_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__))))
VOSK_PATH = os.path.join(MAIN_FOLDER_PATH, 'models', 'vosk')
STORIES_LIBRARY = os.path.join(MAIN_FOLDER_PATH,  "Story Library")

# Call Kita
kita = Kita("English(US)", "vosk", VOSK_PATH) 

with open('configuration.json') as json_file:
    data = json.load(json_file)

selected_story_book = data['story_book']
print(selected_story_book)

#for mode, value in data['read_mode'].items():
#    if value == True:
kita.reading_with_kita()

# Preprocess the story books and Load the Metaverse for each of them (call this just once per book to store the image, preprocess the story book etc.)
# Generate the images only once per book, otherwise there will be huge latencies in the algorithm
for story in os.listdir(STORIES_LIBRARY):
   if os.path.isfile(os.path.join(STORIES_LIBRARY, story)):
       if os.path.exists(os.path.join(MAIN_FOLDER_PATH, 'JSON Images Filestore', story)) == False: # the book doesn't have the images generated yet
        PDF_STORY_PATH = os.path.join(STORIES_LIBRARY, story)
        metaverse = StoryBookPreprocessor(PDF_STORY_PATH, story) 
        metaverse.process_text_load_metaverse() 


# Display the metaverse for the selected story book
#PDF_STORY_PATH = os.path.join(STORIES_LIBRARY, selected_story_book)
#metaverse = MetaverseGenerator(PDF_STORY_PATH, selected_story_book) 
#metaverse.read_story_by_page() 
