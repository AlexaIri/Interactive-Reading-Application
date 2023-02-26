from readWithMe import ReadWithMe
from metaverseGenerator import MetaverseGenerator

# put relative paths instead 
readWithMe = ReadWithMe("English(US)", "vosk", r"C:\Users\Asus ZenBook\Desktop\UCL\Interactive_Reading_App_with_MotionInput\models\vosk") #"C:\Users\DELL\Desktop\Dissertation\models\vosk")
# class with true/false for each functionality, based on json file values

# Word Sync
readWithMe.karaoke_reading_by_words()

# Phrase Sync
#readWithMe.karaoke_reading_by_phrases()

# Context Sync. Preprocess the story books and Generate the Metaverse for each of them 
#readWithMe.karaoke_reading_by_context()

# Preprocess the story books and Load the Metaverse for each of them (call this just once per book to store the image, preprocess the story book etc.)
metaverse = MetaverseGenerator(r"C:\Users\Asus ZenBook\Desktop\UCL\Alice_removed.pdf")
#metaverse.process_text_load_metaverse()
   