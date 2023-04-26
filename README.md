# Interactive Reading Application with MotionInput

System Manual 
The system manual presents the necessary steps so that the future contributors will be able to run and test the application end-to-end as well as comprehend the internal structure and logic of the system.

Functionalities and dependencies:

1.	The code can be found by accessing the public GitHub repository: 

https://github.com/AlexaIri/Interactive-Reading-Application.git

2.	After cloning the repository (or downloading it and unzipping it), open the Visual Studio 2022 IDE. Using Visual Studio Code for development purposes is not recommended as there can be certain features that are not supported or rendered as expected.

3.	The Interactive Reading Application master folder should have the following internal structure:

 

4.	All the dependencies and libraries needed within the project are enlisted in the requirements.txt file. Install them all at once using pip: -r requirements.txt. If any dependency is installed manually, this should be installed with pip install or python -m pip install followed by the name of the package. A short compilation with the most important ones which  includes the following:

pip install cmake==3.25.2
pip install python==3.25.2
pip install pygments
pip install pyautogui
pip install sounddevice
pip install numpy
pip install opencv-python
pip install pocketsphinx==0.1.15v
pip install SpeechRecognition
pip install pytesseract 
pip install h5py
pip install spacy
pip install Pillow==9.3.0
pip install conda==23.1.0
pip install nltk
pip install PyMuPDF==1.16.14
python -m spacy download en_core_web_sm

As an observation, when using the NLTK library, download and install the resources (e.g., nltk.download(‘punkt’), nltk.download('averaged_perceptron_tagger')) after importing the library.

5.	Additionally, download the following packages from the Internet:
a)	Tesseract-OCR 5.0.0: https://pypi.org/project/pytesseract/
b)	Python 3.9
c)	Vosk 0.3.38: https://alphacephei.com/vosk/install
d)	PyQT5 5.15.6: https://pypi.org/project/PyQt5/
e)	Pyinstaller 4.2: https://pyinstaller.org/en/stable/installation.html

6.	Create an account on OpenAI and obtain the secret key. Input this unique Authorization Token in the auth.py file.

7.	Create a models folder for vosk, with the structure models/vosk/ and populate this location with
 
8.	Locate Vosk in the computer:
A standard location path would be: C:\Users\user\AppData\Local\Programs\Python\Python39\Lib\site-packages\vosk
Due to the large size, the folder was uploaded at the following link:
https://drive.google.com/drive/folders/1SpYwr_uZ8gd0PEgPPyzp1LKheNk-fCdo?usp=sharing
9.	Locate Tesseract OCR in the computer and change the path to the tessdata. 
A standard location path would be: 'C:\Program Files\Tesseract-OCR\tesseract.exe'

Due to the large size, the folder was uploaded at the following link:


10.	The Virtual Portfolio folder needs to be added into the project folder. It is used for rendering the contents of an input story book and for accessing the images to render.Moreover, as seen in the picture below, it contains the Educational Materials Library folder (referred in the code as Story Library), the JSON Images Filestore folder and the PNG Images Filestore folder.
 
Due to being its large size, it was uploaded at the following location: https://drive.google.com/drive/folders/1mvJlOGj_-6DzfLxpuzh6_nwbZo8uaaJ3?usp=share_link
4)	To run the Back-end code – run the class implemented for testing purposes, namely callApi.py:
1.	All the paths are absolute, thus they do not require any chane
2.	This file calls KITA at the line: kita = Kita("English(US)", "vosk", VOSK_PATH), thus change the language parameter to the target language.
3.	Further configuration parameters are read from the configuration.json file:

 

The file contains the three reading modes, namely (1) pronunciation_checker, (2) reading_tracker, (3) metaverse_generator. It contains true-false values for each one of them and the only condition is that only one can be true at any given time.

If the user wants to run the Pronunciation Checker feature, set the value of pronunciation_checker to true, leaving the other two set to false. For this feature, just open any document on the screen (PDF documents by default) and the mode will be triggered automatically.

If the user wants to run the Reading Tracker feature, set the value of reading_tracker to true, leaving the other two set to false. For this feature, just open any document on the screen (PDF documents by default) and the mode will be triggered automatically.

If the user wants to run the Metaverse Generator feature, set the value of metaverse_generator to true, leaving the other two set to false. For this feature, input the title of any document into the story_book parameter of the JSON file after making sure the book with the same name is downloaded in the Story Library folder. 

Also, the callApi.py file also contains a call to the Book Preprocessor class which internally calls the Image Generator class. When a new book is added into the Story Library folder, the API automatically starts the book pre-processing and the rendering of the JSON and PNG, folders respectively.
