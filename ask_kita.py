from vosk_model import VoskModel

class Ask_KITA:
    def __init__(self, lang, model, model_path):
        self.lang = lang
        self.model = model
        self.finalModel = VoskModel(self.lang, model_path)

    

    

  