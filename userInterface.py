import tkinter as tk
import customtkinter as ctk
from tkinter import *
from readWithMe import ReadWithMe

readWithMe = ReadWithMe("English(US)", "vosk", r"C:\Users\Asus ZenBook\Desktop\UCL\Interactive_Reading_App_with_MotionInput\models\vosk") #"C:\Users\DELL\Desktop\Dissertation\models\vosk")

app = tk.Tk()
app.geometry("512x512")
app.title("Let's Learn Together How to Read!")

#text = Text(app)  
#text.insert(INSERT, "Let's Learn Together How to Read!")  
  
#text.pack()  

canvas = tk.Canvas(app,bg="white", width = 512, height = 600)
#canvas.place(x=10, y = 110)

#magic_button = ctk.CTkButton(master = app, height = 40, width = 120)
#magic_button.configure(text="Apply magic")
#magic_button.place(x=133, y = 60)


#save_button = ctk.CTkButton(master = app, height = 40, width = 120)
#save_button.configure(text="Save image")
#save_button.place(x=200, y = 100)

# Radial Menu
#circle = canvas.create_oval(80,80,420,420,outline="black",fill="#b0d6ea")
#circle2 = canvas.create_oval(180,180,320,320,outline="black",fill="white")

# Word Sync
#readWithMe.karaoke_reading_by_words()
word = ctk.CTkButton(master = app, height = 50, width = 50, bg_color='blue', fg_color="black", command= readWithMe.karaoke_reading_by_words)
word.configure(text="Read By WORD sync")
word.place(x=100, y = 100)

phrase = ctk.CTkButton(master = app, height = 50, width = 50, bg_color='blue', fg_color="black", command= readWithMe.karaoke_reading_by_phrases)
phrase.configure(text="Read By PHRASE sync")
phrase.place(x=90, y = 200)

phrase = ctk.CTkButton(master = app, height = 50, width = 50, bg_color='blue', fg_color="black", command= readWithMe.karaoke_reading_by_context)
phrase.configure(text="Read By CONTEXT sync")
phrase.place(x=90, y = 300)
#wordSync = canvas.create_text(300, 150, text='Read By WORD SYN',tag="wordsync")
#canvas.tag_bind("wordsync", "save_button", lambda e:print ("Hi i am command 1"))

#txt = canvas.create_text(200, 80, text='Command 2',angle=48,tag="command2")
#canvas.tag_bind("command2", "<Button-1>",lambda e:print ("Hi i am command 2"))

canvas.pack()

#root.wm_attributes("-transparentcolor", "white")
#app.overrideredirect(True)

app.mainloop()