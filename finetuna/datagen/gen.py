import dill
from jinja2 import Environment, StrictUndefined, Undefined
import openai
import os
import random
from typing import Callable, Any, Union

from finetuna.utils import files_wo_suffix, load_from_jsonl, write_to_jsonl
from finetuna.consts import OPENAI_API_KEY, DATASETS_PATH, DATA_GENERATORS_PATH
from finetuna.completers import Completer

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
    def get_dataset_path(name, dir=None) -> str:
        path = f"{DATASETS_PATH}/{name}.jsonl"
        if dir is not None:
            path = f"{dir}/{path}"
        return path
    def dataset_path(self, dir=None) -> str:
        return DataGenerator.get_dataset_path(self.name, dir=dir)
    
    @staticmethod
    def get_data_generator_path(name, dir=None) -> str:
        path = f"{DATA_GENERATORS_PATH}/{name}.pkl"
        if dir is not None:
            path = f"{dir}/{path}"
        return path
    def data_generator_path(self, dir=None) -> str:
        return DataGenerator.get_data_generator_path(self.name, dir=dir)
    
    @staticmethod
    def load(name : str, dir=None) -> 'DataGenerator':
        with open(DataGenerator.get_data_generator_path(name, dir=dir), "rb") as f:
            data_generator = dill.load(f)
        dataset = load_from_jsonl(DataGenerator.get_dataset_path(name, dir=dir))
        data_generator.dataset = dataset
        return data_generator
    
    def save(self, warn_if_exists=False, custom_dir=None):
        if warn_if_exists:
            assert not DataGenerator.name_exists(self.name), f"Dataset {self.name} already exists. Please choose a different name, or set warn_if_exists=True."
        # consts.py already makes these for the default case,
        # but not if a custom_dir is provided (useful for testing)
        os.makedirs(
            os.path.dirname(
                self.data_generator_path(custom_dir), 
            ), exist_ok=True
        )
        os.makedirs(
            os.path.dirname(
                self.dataset_path(custom_dir)
            ), exist_ok=True
        )
        with open(self.data_generator_path(custom_dir), "wb") as f:
            print(self)
            dill.dump(self, f)
        self.save_data_to_jsonl(self.dataset_path(custom_dir))
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
        write_to_jsonl(self.dataset, filepath)
    
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

def raise_not_implemented_error(s):
    raise NotImplementedError(s)

class DataHolder(DataGenerator):
    """
    A DataGenerator that does not assume ability to
    generate additional datapoints (but can support it).
    Useful for loading existing datsets that weren't generated
    by a DataGenerator.
    """
    def __init__(
        self, 
        jsonl_path_or_dataset,
        prompt_gen_fn : Callable[[LatentState], str] = lambda ls : raise_not_implemented_error("DataHolder was not given a prompt_gen_fn"),
        completion_gen_fn : Callable[[str, LatentState], str] = lambda prompt, ls : raise_not_implemented_error("DataHolder was not given a completion_gen_fn"),
        latent_state_gen_fn : Callable[[], LatentState] = lambda : raise_not_implemented_error("DataHolder was not given a latent_state_gen_fn"),
        name : str = "unnamed_dataset"
    ):
        super().__init__(
            prompt_gen_fn,
            completion_gen_fn,
            latent_state_gen_fn,
            name=name
        )

        if isinstance(jsonl_path_or_dataset, str):
            self.dataset = load_from_jsonl(jsonl_path_or_dataset)
        else:
            assert isinstance(jsonl_path_or_dataset, list), "jsonl_path_or_dataset must be a path to a JSONL file or a list of data points"
            assert jsonl_path_or_dataset[0].keys() == {"prompt", "completion"}, "jsonl_path_or_dataset should consist of {'prompt': ..., 'completion': ...} dicts"
            self.dataset = jsonl_path_or_dataset
        self.name = "unnamed_dataset"

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

def completion_maker_fn(
    prompt_template : str,
    completer : Completer
):
    def get_completion(
        prompt_in_dataset : str,
        special_vars = {}
    ) -> str:
        """
        special_vars is for the latent state.
        `prompt_in_dataset` is called that to distinguish it from the prompt template used to construct the completion for the dataset
        """
        env = Environment(
            undefined=Undefined,
            autoescape=True)
        jinja_template = env.from_string(prompt_template)
         
        variables = {
            "prompt": prompt_in_dataset
        }

        if "prompt" in special_vars.keys():
            raise Exception("You cannot use 'prompt' as a latent state property name, because it is already used to store the prompt.")
        
        # overwrite with any special values:
        variables.update(special_vars)
        
        gpt_prompt = jinja_template.render(**variables)

        # now we get the completion:
        completion_text : str = completer(gpt_prompt)
        
        return completion_text
    return get_completion
        