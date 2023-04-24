from auth import auth_token_gpt3
import os
import openai
from pathlib import Path
import json
import sys
from base64 import b64decode

class GPT3_DALLE2_Image_Generator():
    # Retrieve the authentication token associated with the OpenAI account of the developer 
    openai.api_key = auth_token_gpt3

    def __init__(self, text_prompt, image_metadata, story_book_name):
        self.start_ind_phrase, self.end_ind_phrase, self.keywords, self.page = image_metadata
        self.story_book_name = story_book_name
        self.text_prompt = text_prompt

        # Create the JSON Images Filestore folder in the Virtual Portfolio
        self.json_dir = Path.cwd() / "JSON Images Filestore" 
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.json_data_dir = self.json_dir / self.story_book_name
        self.json_data_dir.mkdir(exist_ok=True)

        # Create the PGN Images Filestore folder in the Virtual Portfolio
        self.png_dir = Path.cwd() / "PNG Images Filestore" 
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.png_data_dir = self.png_dir / self.story_book_name
        self.png_data_dir.mkdir(exist_ok=True)
        

    def get_index(self, dir):
        return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
    
    def encode_images_to_json(self, response):
        # Declare the JSON file path name
        file_name = self.json_data_dir / f"{self.get_index(self.json_data_dir)}-json.json"

        # Populate the JSON with the encoded image
        with open(file_name, mode="wb", encoding=sys.stdout.encoding) as file:
            json.dump(response, file)

        return file_name 

    def decode_image_from_json(self, json_file):
        subfolder = self.png_data_dir / f"{json_file.stem}-subfolder"
        subfolder.mkdir(parents=True, exist_ok=True)

        # Load the JSON file into response
        with open(json_file, mode="rb", encoding=sys.stdout.encoding) as file:
            response = json.load(file)

        # Deserialise each of the six images into PNGs
        for index, image in enumerate(response["data"]): 

            image_data = b64decode(image["b64_json"])
            image_file = subfolder / f"Image-{index}-{response['created']}.png"

            with open(image_file, mode="wb", encoding=sys.stdout.encoding) as png:
                png.write(image_data)


    def update_jsons(self, json_file):
        with open(json_file, "rb", encoding=sys.stdout.encoding) as file_name:
            data = json.load(file_name)

        data['start_ind_phrase']= self.start_ind_phrase
        data['end_ind_phrase'] = self.end_ind_phrase
        data['keywords']= self.keywords
        data['page'] = self.page

        with open(json_file, "wb") as file_name:
            json.dump(data, file_name)


    def retrieve_image_from_OpenAI(self):
        # Ensure the images are appropriate for kids and have a uniform style
        prompt = '"{}"'.format(self.text_prompt) + " digital art for children"
        try:
            # Call the GPT3-DALLE2 OpenAI Image Generator API and render 6 image variations
            response = openai.Image.create(prompt = prompt,
                                           n = 6, 
                                           size ="512x512",
                                           response_format="b64_json",
                                           )
            
            # Append image metadata to the response
            response['start_ind_phrase']= self.start_ind_phrase
            response['end_ind_phrase'] = self.end_ind_phrase
            response['keywords']= self.keywords
            response['page'] = self.page

            # Serialise the images into JSONs and deserialise them to PDFs
            file = self.encode_images_to_json(response)
            self.decode_image_from_json(file)
            
        except:
            print("Exception thrown due to non-compliance with the OpenAI safety guidelines")
