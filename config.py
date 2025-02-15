import os
from dotenv import load_dotenv

load_dotenv()

def load_config():
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("Please set your OPENAI_API_KEY in the .env file")
