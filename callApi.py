from readWithMe import ReadWithMe
from metaverseGenerator import MetaverseGenerator
import json
import os

MAIN_FOLDER_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__))))
VOSK_PATH = os.path.join(MAIN_FOLDER_PATH, 'models', 'vosk')
PDF_STORY_PATH = os.path.join(MAIN_FOLDER_PATH, 'Alice_Story.pdf')

# Call the ReadWithMe API
readWithMe = ReadWithMe("English(US)", "vosk", VOSK_PATH) 

# Word Sync
#readWithMe.karaoke_reading_by_words()

# Phrase Sync
#readWithMe.karaoke_reading_by_phrases()

# Context Sync. Preprocess the story books and Generate the Metaverse for each of them 
#readWithMe.karaoke_reading_by_context()

# Preprocess the story books and Load the Metaverse for each of them (call this just once per book to store the image, preprocess the story book etc.)
metaverse = MetaverseGenerator(PDF_STORY_PATH)
#metaverse.process_text_load_metaverse()

with open('configuration.json') as json_file:
    data = json.load(json_file)

for mode, value in data['read_mode'].items():
    if value == True:
        readWithMe.karaoke_sync_reading(mode)
