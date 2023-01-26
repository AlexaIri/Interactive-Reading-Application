from readWithMe import ReadWithMe
from metaverseGenerator import MetaverseGenerator

readWithMe = ReadWithMe("English(US)", "vosk", r"C:\Users\Asus ZenBook\Desktop\UCL\Interactive_Reading_App_with_MotionInput\models\vosk") #"C:\Users\DELL\Desktop\Dissertation\models\vosk")
#readWithMe.karaoke_reading()

metaverse = MetaverseGenerator(r"C:\Users\Asus ZenBook\Desktop\DISSERTATION\Alice in Wonderland.pdf")
metaverse.extract_text()
