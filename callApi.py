from readWithMe import ReadWithMe
from metaverseGenerator import MetaverseGenerator

readWithMe = ReadWithMe("English(US)", "vosk", r"C:\Users\Asus ZenBook\Desktop\UCL\Interactive_Reading_App_with_MotionInput\models\vosk") #"C:\Users\DELL\Desktop\Dissertation\models\vosk")
#readWithMe.karaoke_reading()

# Preprocess the story books and Generate the Metaverse for each of them
metaverse = MetaverseGenerator(r"C:\Users\Asus ZenBook\Desktop\UCL\Alice_removed.pdf")
metaverse.extract_text()
    