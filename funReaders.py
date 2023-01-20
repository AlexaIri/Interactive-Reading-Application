import sys 
from readWithMe import ReadWithMe

lang = sys.argv[1]
model = sys.argv[2]
mode = sys.argv[3]
model_path = sys.argv[4]
font_size = int(sys.argv[5])

speech_engine = ReadWithMe(lang, model, model_path)

print("The call to the ReadWithMe API was successful!")
if mode == "read":
    speech_engine.karaoke_reading()