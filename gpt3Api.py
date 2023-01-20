from cmd import PROMPT
from fileinput import filename
from http.client import responses
from typing_extensions import Self
from auth import auth_token_gpt3
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import os
import requests
import openai
from pathlib import Path
import json
from base64 import b64decode
from os import listdir
from os.path import isfile, join

class GPT3Api():
    # retrieve the authentication token associated with the OpenAI account of the developer 
    openai.api_key = auth_token_gpt3

    def __init__(self):
        #self.image_metadata = image_metadata
        self.app = tk.Tk()
        self.app.geometry("532x632")
        self.app.title("dalle")
        #ctk.set_appearence_mode("dark")

        self.main_image = tk.Canvas(self.app, width = 512, height = 512)
        self.main_image.place(x=10, y = 110)

        self.prompt_input = ctk.CTkEntry(master = self.app, height = 40, width = 512 )
        self.prompt_input.place(x=10, y=10)

        self.data_dir = Path.cwd() / "json_image_filestore"
        print(self.data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.image_dir = Path.cwd() / "png_image_filestore" 
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def encode_images_to_json(self, prompt, response):
        file_name = self.data_dir / f"{prompt[:15]}-{response['created']}.json"
        with open(file_name, mode="w", encoding="utf-8") as file:
            json.dump(response, file)
        return response, file_name #, image_unique_id

    def decode_image_from_json(self, json_filename):
        #last_generated_file = [f for f in listdir(data_dir) if isfile(join(data_dir, f))][-1]
        #print(last_generated_file)
        #for file in onlyfiles:
        json_file = self.data_dir / json_filename
        with open(json_file, mode="r", encoding="utf-8") as file:
            response = json.load(file)


        for index, image_dict in enumerate(response["data"]):

            image_data = b64decode(image_dict["b64_json"])
            image_file = self.image_dir / f"{json_file.stem}-{index}.png"

            with open(image_file, mode="wb") as png:
                png.write(image_data)


    def retrieve_image_from_gpt3OpenAI(self):
        global tk_img
        global img
        text_prompt = self.prompt_input.get()
        response = openai.Image.create(prompt = text_prompt,
                                       n = 2, 
                                       size ="512x512",
                                       response_format="b64_json",
                                       )
        print(type(response))
        # append to response additional info as json: self.image_metadata 
        #image_url = response["data"][1]["url"]
        #img = Image.open(requests.get(image_url, stream = True).raw)
        #tk_img = ImageTk.PhotoImage(img)
        #main_image.create_image(0, 0, anchor = tk.NW, image = tk_img)
        #image_unique_id = f"{prompt[:5]}-{response['created']}"
        # ADD IMAGE_METADATA IN THAT JSON: the whole passage text from which it was generated, 
        #the start and end indexes of it, the indexes of the phrases it contains 
        _, file_name = self.encode_images_to_json(text_prompt, response)
        self.decode_image_from_json(file_name)

    def save_image_as_pngs(self):
    
        prompt = self.prompt_input.get().replace(" ", "_")
    
        #image_path = "C:\Users\DELL\Desktop\Thesis\GPT3Alg\GPT3_Foundational_Code\GPT3_Foundational_Code\images_filestore"

        #os.mkdir(image_path)

        img.save(f"images_filestore/{prompt}.png")
        #img.save("C:\Users\DELL\Desktop\Thesis\GPT3Alg\GPT3_Foundational_Code\GPT3_Foundational_Code\images_filestore\{}.png".format(prompt), 'JPEG')

    def view_app(self):
        magic_button = ctk.CTkButton(master = self.app, height = 40, width = 120, command = self.retrieve_image_from_gpt3OpenAI)
        magic_button.configure(text="Apply magic")
        magic_button.place(x=133, y = 60)


        save_button = ctk.CTkButton(master = self.app, height = 40, width = 120, command = self.save_image_as_pngs)
        save_button.configure(text="Save image")
        save_button.place(x=266, y = 60)

        self.app.mainloop()


