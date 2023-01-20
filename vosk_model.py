import queue
import sounddevice as sd
import vosk
vosk.SetLogLevel(-1)
import os
import pytesseract
import spacy

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Asus ZenBook\AppData\Local\Tesseract-OCR\tesseract'

class VoskModel():
    def __init__(self, lang, model_path, mode='transcription', safety_word='stop' ):
        self.model_path = model_path # self._get_model_path()
        self.q = queue.Queue()
        self.previous_line = ""
        self.previous_length = 0
        self.mode = mode
        self.safety_word = safety_word
        self.lang = lang
        self.rec = ""
        self.text_dict = {}
        self.co_ord_list = []
        self.last_index = -1
        self.first_index = -1
        self.match = False
        self.nlp = spacy.load("en_core_web_sm")

    def setUp(self):
        # nltk.download()
        if not os.path.exists(self.model_path):
            print("Please download a model for your language from https://alphacephei.com/vosk/models")
            print(f"and unpack into {self.model_path}.")
        print("model's path:" + self.model_path)
        device_info = sd.query_devices(kind='input')
        samplerate = int(device_info['default_samplerate'])
        location = self.model_path+ "\\"+ self.lang
        
        model = vosk.Model(location)

        rec = vosk.KaldiRecognizer(model, samplerate)
        print('#' * 80)
        print('Press Ctrl+C to stop the recording')
        print('#' * 80)
        return rec,samplerate

    def _get_model_path(self):
        full_path = os.path.realpath(__file__)
        file_dir = os.path.dirname(full_path)
        model_path = os.path.join(file_dir, 'models/vosk') 
        return model_path
    
  