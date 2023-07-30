import openai
from dataclasses import dataclass

from finetuna.consts import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

# if you want to test things while being sure that no money is spent:
BLOCK_API_CALLS = False
# (also see DummyCompleter below)

class Completer:
    def __init__(
        self,
        model : str,
        messages=[], # starting messages
        args={}
    ):
        """
        Pass additional args into `args`;
        see full OpenAI API reference for valid args here:
        https://platform.openai.com/docs/api-reference/chat/create
        """
        
        if len(messages) > 0:
            assert "content" in messages[0].keys() and "role" in messages[0].keys(), "Each message must have a 'content' and 'role' key ( + optionally others, see OpenAI API reference)"
        
        self.model = model
        self.args = args
        self.messages = messages
        
        self.used_tokens = 0
    
    def completion(self, prompt : str):
        if BLOCK_API_CALLS:
            raise Exception("Calls to OpenAPI are blocked by BLOCK_API_CALLS = True in completers.py.")
        
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages = self.messages + [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            **self.args
        )
        self.used_tokens += completion.usage["total_tokens"]
        return completion
    
    def response(self, prompt : str):
        return self.completion(prompt).choices[0]["message"]["content"]
    
    def __call__(self, prompt : str):
        return self.response(prompt)


@dataclass
class DummmyCompletion:
    choices: list
    usage: dict
    

class DummyCompleter(Completer):
    def __init__(
        self, 
        response_fn = lambda prompt, **kwargs : "I AM VERI SMRT AI, BOW TO ME"
    ):
        super().__init__("dummy")
        self.response_fn = response_fn
    
    def completion(self, prompt : str):
        return DummmyCompletion(
            choices=[
                {"message": {"content": self.response_fn(prompt)}}
            ],
            usage={"total_tokens": 1}
        )


gpt3turbo = Completer(
    "gpt-3.5-turbo"
) 

gpt3turbo_line = Completer(
    "gpt-3.5-turbo",
    args={
        "stop": "\n",
        "max_tokens": 100
    }
)

