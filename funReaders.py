import sys 
from ask_kita import Ask_KITA

lang = sys.argv[1]
model = sys.argv[2]
mode = sys.argv[3]
model_path = sys.argv[4]
font_size = int(sys.argv[5])

speech_engine = Ask_KITA(lang, model, model_path)


