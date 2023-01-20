from gpt3Api import MetaverseGenerator
from readWithMe import ReadWithMe

readWithMe = ReadWithMe("English(US)", "vosk", r"C:\Users\Asus ZenBook\Desktop\UCL\Reading_Application\models\vosk") #"C:\Users\DELL\Desktop\Dissertation\models\vosk")
readWithMe.karaoke_reading()

gpt3 = MetaverseGenerator()
#gpt3.view_app()



