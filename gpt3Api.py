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

    def __init__(self, text_prompt, image_metadata, story_book_name):
        self.start_ind_phrase, self.end_ind_phrase, self.keywords, self.page = image_metadata
        self.story_book_name = story_book_name
        self.text_prompt = text_prompt

        #self.json_data_dir = Path.cwd() / "json_image_filestore"
        #self.json_data_dir.mkdir(exist_ok=True)
        
        #self.png_data_dir = Path.cwd() / "png_image_filestore" 
        #self.png_data_dir.mkdir(parents=True, exist_ok=True)

        self.json_dir = Path.cwd() / "JSON Images Filestore" 
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.json_data_dir = self.json_dir / self.story_book_name
        self.json_data_dir.mkdir(exist_ok=True)

        self.png_dir = Path.cwd() / "PNG Images Filestore" 
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.png_data_dir = self.png_dir / self.story_book_name
        self.png_data_dir.mkdir(exist_ok=True)
        

    def get_index(self, dir):
        return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])

    def encode_images_to_json(self, response):

        # Declare the json file path name
        file_name = self.json_data_dir / f"{self.get_index(self.json_data_dir)}-json.json"
        # Populate the json with the serialised image
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
        data['page'] = self.page

        with open(json_file, "w") as file_name:
            json.dump(data, file_name)

    def retrieve_image_from_gpt3OpenAI(self):
        prompt = '"{}"'.format(self.text_prompt) + " digital art for children"
        try:
            response = openai.Image.create(prompt = prompt,
                                           n = 6, 
                                           size ="512x512",
                                           response_format="b64_json",
                                           )
            
            # Append image metadata to response
            response['start_ind_phrase']= self.start_ind_phrase
            response['end_ind_phrase'] = self.end_ind_phrase
            response['keywords']= self.keywords
            response['page'] = self.page

            
            file = self.encode_images_to_json(response)
            self.decode_image_from_json(file)
            
        except:
            print("Exception thrown due to non-compliance with the OpenAI safety guidelines")
