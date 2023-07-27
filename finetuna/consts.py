from dotenv import load_dotenv
import os

load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

assert OPENAI_API_KEY is not None, "OpenAI API key not found. Please set it in .env."

DATA_PATH = "data"
 
DATASETS_PATH = DATA_PATH + "/datasets"
DATA_GENERATORS_PATH = DATA_PATH + "/data_generators"

EVAL_PATH = DATA_PATH + "/eval"

for path in [
    DATA_PATH,
    DATASETS_PATH,
    DATA_GENERATORS_PATH,
    EVAL_PATH
]:
    if not os.path.exists(path):
        os.makedirs(path)
