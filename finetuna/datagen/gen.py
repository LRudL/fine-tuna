import dill
from jinja2 import Environment, StrictUndefined, Undefined
import json
import openai
import random
from typing import Callable, Any, Union

from finetuna.utils import files_wo_suffix
from finetuna.consts import OPENAI_API_KEY, DATASETS_PATH, DATA_GENERATORS_PATH

openai.api_key = OPENAI_API_KEY

LatentState = Any
# the model behind data point generation is:
# 1. a latent state (e.g. which type of prompt to generate, any information that both the prompt and completion need to share)
# 2. a prompt generator function that takes in a latent state and returns a prompt, where "prompt" means the same thing as it does in OpenAI finetuning data format
# 3. a completion generator function that takes in the prompt and the latent state

class DataGenerator:
    def __init__(
        self,
        prompt_gen_fn : Callable[[LatentState], str],
        completion_gen_fn : Callable[[str, LatentState], str],
        latent_state_gen_fn : Callable[[], LatentState] = lambda: None,
        name : str = "unnamed_dataset"
    ):
        self.prompt_gen_fn = prompt_gen_fn
        self.completion_gen_fn = completion_gen_fn
        self.latent_state_gen_fn = latent_state_gen_fn
        self.dataset = []
        self.latent_states = []
        self.name = name
     
    @staticmethod
    def name_exists(name : str) -> bool:
        if name in files_wo_suffix(DATASETS_PATH):
            return True
        return False
    
    @staticmethod
    def create_or_load(
        name : str,
        prompt_gen_fn : Callable[[LatentState], str],
        completion_gen_fn : Callable[[str, LatentState], str],
        latent_state_gen_fn : Callable[[], LatentState] = lambda: None
    ) -> 'DataGenerator':
        if DataGenerator.name_exists(name):
            return DataGenerator.load(name)
        return DataGenerator(
            prompt_gen_fn,
            completion_gen_fn,
            latent_state_gen_fn,
            name
        )

    @staticmethod
    def get_dataset_path(name) -> str:
        return f"{DATASETS_PATH}/{name}.jsonl"
    def dataset_path(self) -> str:
        return DataGenerator.get_dataset_path(self.name)
    
    @staticmethod
    def get_data_generator_path(name) -> str:
        return f"{DATA_GENERATORS_PATH}/{name}.pkl"
    def data_generator_path(self) -> str:
        return DataGenerator.get_data_generator_path(self.name)
    
    @staticmethod
    def load(name : str) -> 'DataGenerator':
        with open(DataGenerator.get_data_generator_path(name), "rb") as f:
            data_generator = dill.load(f)
        with open(DataGenerator.get_dataset_path(name), "r") as f:
            dataset = []
            for line in f:
                dataset.append(json.loads(line))
            data_generator.dataset = dataset
        return data_generator
    
    def save(self, warn_if_exists=False):
        if warn_if_exists:
            assert not DataGenerator.name_exists(self.name), f"Dataset {self.name} already exists. Please choose a different name, or set warn_if_exists=True."
        with open(self.data_generator_path(), "wb") as f:
            print(self)
            dill.dump(self, f)
        self.save_data_to_jsonl(self.dataset_path())
        print(f"Wrote dataset {self.name} to {self.dataset_path()}. You can load it with DataGenerator.load('{self.name}').")
    
    def generate(self, n=10):
        """
        Generate n samples of data.
        """
        for _ in range(n):
            latent_state = self.latent_state_gen_fn()
            self.latent_states.append(latent_state)
            prompt = self.prompt_gen_fn(latent_state)
            completion = self.completion_gen_fn(prompt, latent_state)
            self.dataset.append({
                "prompt": prompt,
                "completion": completion
            })
        return self.dataset[-n:]
    
    def save_data_to_jsonl(self, filepath):
        """
        Save the generated dataset to a JSONL file.
        NOTE: in general, you should use the .save() method instead of this method.
        """
        with open(filepath, "w") as f:
            for item in self.dataset:
                f.write(json.dumps(item) + "\n")
    
    def print_sample(self, n=10):
        """
        Pretty-print a random size-n sample of the generated dataset.
        """
        n = min(n, len(self.dataset))
        bold = "\033[1m"
        reset = "\033[0m"
        boundary = "--------------------"
        print(boundary)
        indices = random.sample(range(len(self.dataset)), n)
        for i in indices:
            prompt_i = self.dataset[i]["prompt"]
            completion_i = self.dataset[i]["completion"]
            print(bold + prompt_i + reset + completion_i)
        print(boundary)

def template_filler_fn(
    template : str,
    use_strict_undefined = True,
    **variable_lists
) -> Union[Callable[[dict], str], Callable[[], str]]:
    """
    Returns a function that fills in a Jinja template string (i.e. {{variable}} is a variable) with random values from the given variable lists.
    
    The function returned by this function can take in a dictionary of special values to overwrite the random values. (e.g. when passing in LatentState with the above class)
    
    Example:
    >>> template = "My name is {{name}} and I am {{age}} years old."
    >>> name_list = ["Alice", "Bob", "Charlie"]
    >>> age_list = [18, 19, 20]
    >>> f = template_filler_fn(template, name=name_list, age=age_list)
    >>> f() -> "My name is Bob and I am 20 years old."
    """
    def random_template(special_vars = {}) -> str:
        # Create a Jinja2 environment with undefined variable handling.
        # Use Undefined to suppress errors for missing variables in lists.
        # (StrictUndefined will raise an error if a variable is missing in a list, otherwise it will just be blank.)
        env = Environment(
            undefined=StrictUndefined if use_strict_undefined else Undefined,
            autoescape=True)
        jinja_template = env.from_string(template)

        # Create a dictionary to store the random variable replacements.
        random_values = {
            var_name: random.choice(var_values) for var_name,
            var_values in variable_lists.items()
        }
        
        # overwrite with any special values:
        random_values.update(special_vars)

        # Render the template with the random values.
        return jinja_template.render(**random_values)
    return random_template

def prompt_gpt_for_completion_fn(
    prompt_template : str,
    gpt_model = "gpt-3.5-turbo",
    gpt_extra_messages = [],
    # e.g. if you want to set a role, pass in [{"role": "system", "content": "..."}]
    completion_args = {"stop": ["\n"], "max_tokens": 200}
):
    def get_completion(
        prompt_in_dataset : str,
        special_vars = {}
    ) -> str:
        """
        special_vars is for the latent state.
        prompt_in_dataset is called that to distinguish it from the prompt template used to construct the completion for the dataset
        """
        env = Environment(
            undefined=Undefined,
            autoescape=True)
        jinja_template = env.from_string(prompt_template)
         
        variables = {
            "__PROMPT__": prompt_in_dataset
        }
        
        # overwrite with any special values:
        variables.update(special_vars)
        
        gpt_prompt = jinja_template.render(**variables)

        # now we get the completion:
        completion = openai.ChatCompletion.create(
            model=gpt_model,
            messages = gpt_extra_messages + [
                {
                    "role": "user",
                    "content": gpt_prompt
                }
            ],
            **completion_args
        )
        
        completion_txt : str = completion.choices[0]["message"]["content"]

        return completion_txt
    return get_completion
        