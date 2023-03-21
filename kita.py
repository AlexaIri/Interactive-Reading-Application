from vosk_model import VoskModel
import json
import os

MAIN_FOLDER_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__))))
VOSK_PATH = os.path.join(MAIN_FOLDER_PATH, 'models', 'vosk')
STORIES_LIBRARY = os.path.join(MAIN_FOLDER_PATH,  "Story Library")

class Kita: 
    def __init__(self, lang, model, model_path):
        self.lang = lang
        self.model = model
        self.voskModel = VoskModel(self.lang, model_path)
        

    def reading_with_kita(self):

        with open('configuration.json') as json_file:
            data = json.load(json_file)

        selected_story_book = data['story_book']

        for mode, value in data['read_mode'].items():
            if value == True:
                self.voskModel.karaoke_reading(mode, selected_story_book)