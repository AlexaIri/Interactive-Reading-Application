from vosk_model import VoskModel

class Kita: 
    def __init__(self, lang, model, model_path):
        self.lang = lang
        self.model = model
        self.finalModel = VoskModel(self.lang, model_path)

  
    #def karaoke_reading_by_words(self):
    #    self.finalModel.karaoke_reading("word_sync")
        
    #def karaoke_reading_by_phrases(self):
    #    self.finalModel.karaoke_reading("phrase_sync")

    #def karaoke_reading_by_context(self):
    #    self.finalModel.karaoke_reading("context_sync")

    def karaoke_sync_reading(self, mode, selected_story_book):
        self.finalModel.karaoke_reading(mode, selected_story_book)