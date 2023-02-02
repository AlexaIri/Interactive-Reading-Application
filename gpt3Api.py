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


class ImageGenerator():
    # retrieve the authentication token associated with the OpenAI account of the developer 
    openai.api_key = auth_token_gpt3

    def __init__(self, text_prompt, image_metadata):
        self.start_ind_phrase, self.end_ind_phrase, self.keywords, self.chapter = image_metadata
        self.app = tk.Tk()
        self.app.geometry("532x632")

        self.text_prompt = text_prompt

        self.main_image = tk.Canvas(self.app, width = 512, height = 512)
        self.main_image.place(x=10, y = 110)

        self.prompt_input = ctk.CTkEntry(master = self.app, height = 40, width = 512 )
        self.prompt_input.place(x=10, y=10)

        self.json_data_dir = Path.cwd() / "json_image_filestore"
        self.json_data_dir.mkdir(exist_ok=True)
        
        self.png_data_dir = Path.cwd() / "png_image_filestore" 
        self.png_data_dir.mkdir(parents=True, exist_ok=True)
        

    def get_index(self, dir):
        return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])

    def encode_images_to_json(self, response):

        # Declare the json file path name
        file_name = self.json_data_dir / f"{self.get_index(self.json_data_dir)}-json.json"
        
        # Populate the json with the serialised imahge
        with open(file_name, mode="w", encoding="utf-8") as file:
            json.dump(response, file)

        return file_name 

    def decode_image_from_json(self, json_file):
        subfolder = self.png_data_dir / f"{json_file.stem}-subfolder"
        subfolder.mkdir(parents=True, exist_ok=True)

        with open(json_file, mode="r", encoding="utf-8") as file:
            response = json.load(file)
            
        for index, image in enumerate(response["data"]): 

            image_data = b64decode(image["b64_json"])
            image_file = subfolder / f"Image-{index}-{response['created']}.png"

            with open(image_file, mode="wb") as png:
                png.write(image_data)

    def update_jsons(self, json_file):
        with open(json_file, "r",  encoding="utf-8") as file_name:
            data = json.load(file_name)

        data['start_ind_phrase']= self.start_ind_phrase
        data['end_ind_phrase'] = self.end_ind_phrase
        data['keywords']= self.keywords
        data['chapter'] = self.chapter

        with open(json_file, "w") as file_name:
            json.dump(data, file_name)

    def retrieve_image_from_gpt3OpenAI(self):
        global tk_img
        global img
        #text_prompt = self.prompt_input.get()
        try:
            response = openai.Image.create(prompt = self.text_prompt,
                                           n = 6, 
                                           size ="512x512",
                                           response_format="b64_json",
                                           )

            #image_url = response["data"][1]["url"]
            #img = Image.open(requests.get(image_url, stream = True).raw)
            #tk_img = ImageTk.PhotoImage(img)
            #main_image.create_image(0, 0, anchor = tk.NW, image = tk_img)

            # Append image metadata to response
            response['start_ind_phrase']= self.start_ind_phrase
            response['end_ind_phrase'] = self.end_ind_phrase
            response['keywords']= self.keywords
            response['chapter'] = self.chapter
            
            file = self.encode_images_to_json(response)
            self.decode_image_from_json(file)
            
        except:
            print("Exception thrown due to text prompt violating the OpenAI safety guidelines")


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
