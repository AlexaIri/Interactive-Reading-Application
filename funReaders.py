import sys 
from kita import Kita

lang = sys.argv[1]
model = sys.argv[2]
mode = sys.argv[3]
model_path = sys.argv[4]
font_size = int(sys.argv[5])

speech_engine = Kita(lang, model, model_path)

print("The call to the Kita API was successful!")
if mode == "read":
    speech_engine.karaoke_reading()