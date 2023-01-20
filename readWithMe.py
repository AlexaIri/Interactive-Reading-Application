from vosk_model import VoskModel

class ReadWithMe:
    def __init__(self, lang, model, model_path):
        self.lang = lang
        self.model = model
        self.finalModel = VoskModel(self.lang, model_path)

  
    def karaoke_reading(self):
        self.finalModel.karaoke_reading()
        
    def word_sync(self):
        self.finalModel.word_sync()