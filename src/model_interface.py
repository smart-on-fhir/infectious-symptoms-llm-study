import os

import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


###############################################################################
#
# Abstract Llm Interface Class
#
###############################################################################
class LlmInterface:
    def get_model_info(self):
        pass

    def saturate_prompt(
        self,
        instruction: str,
        context: str,
        system: str = None,
        prompt_format: str = None,
    ):
        pass

    def fetch_llm_response(
        self,
        instruction: str,
        context: str,
        prompt_format: str = None,
        system: str = None,
    ):
        pass


###############################################################################
#
# Gpt3Interface using the AzureOpenAI client
#
###############################################################################
class Gpt3Interface(LlmInterface):
    def __init__(self, url=None):
        self.client = AzureOpenAI(
            azure_endpoint=url or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
        )
        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        GPT_DEFAULT_PROMPT_FORMAT = (
            "### Instructions ###\n%(instruction)s ### Text ###\n%(context)s"
        )
        self.default_prompt_format = GPT_DEFAULT_PROMPT_FORMAT

    def get_model_info(self):
        return self.client.models.retrieve(self.model)

    # Fills the model's prompt-format with instructions, context, and system information
    def saturate_prompt(
        self,
        instruction: str,
        context: str,
        system: str = None,
        prompt_format: str = None,
    ):
        prompt_format = prompt_format or self.default_prompt_format
        full_prompt = prompt_format % {
            "instruction": instruction,
            "context": context,
            # System might be None
            "system": system,
        }
        return full_prompt

    def fetch_llm_response(
        self,
        instruction: str,
        context: str,
        prompt_format: str = None,
        system: str = None,
    ):
        payload = self.saturate_prompt(
            prompt_format=prompt_format,
            instruction=instruction,
            context=context,
            system=system,
        )

        response = self.client.chat.completions.create(
            model=self.model,  # model = "deployment_name".
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": payload},
            ],
        )
        return response.choices[0].message.content


###############################################################################
#
# TgiClient
# A client for making requests to the Text Generation Inference API
# See here for more: https://huggingface.co/docs/text-generation-inference/index
#
################################################################
class TgiClient:
    def __init__(self, url):
        self.url = url

    # Makes API call and parses response
    def fetch_llm_response(self, payload):
        response = requests.post(
            self.url,
            json={
                "inputs": payload,
                "options": {
                    "wait_for_model": True,
                },
                "parameters": {
                    "max_new_tokens": 1000,
                },
            },
        )
        response.raise_for_status()

        answer = response.json()[0]["generated_text"]

        # In case the answer includes the prompt
        if answer.startswith(payload):
            answer = answer[len(payload) :].strip()

        return answer


###############################################################################
#
# Generic TGI LLM class
# Uses the TGI client to fulfil the generic LLM interface
#
################################################################
class TgiInterface(LlmInterface):
    def __init__(self, url, default_prompt_format):
        self.url = url
        self.client = TgiClient(url)
        self.default_prompt_format = default_prompt_format

    def get_model_info(self):
        return requests.get(self.url + "info").json()

    # Fills the model's prompt-format with instructions, context, and system information
    def saturate_prompt(
        self,
        instruction: str,
        context: str,
        system: str = None,
        prompt_format: str = None,
    ):
        prompt_format = prompt_format or self.default_prompt_format
        full_prompt = prompt_format % {
            "instruction": instruction,
            "context": context,
            # System might be None
            "system": system,
        }
        return full_prompt

    def fetch_llm_response(
        self,
        instruction: str,
        context: str,
        prompt_format: str = None,
        system: str = None,
    ):
        payload = self.saturate_prompt(
            prompt_format=prompt_format,
            instruction=instruction,
            context=context,
            system=system,
        )
        return self.client.fetch_llm_response(payload)


###############################################################################
#
# LLAMA2
#
################################################################
#
# This is the formatting that Llama2's chat model is trained on.
# https://huggingface.co/blog/llama2#how-to-prompt-llama-2
LLAMA2_DEFAULT_PROMPT_FORMAT = """<s>[INST] <<SYS>>
%(instruction)s
<</SYS>>

%(context)s [/INST]"""


class LLAMA2Interface(TgiInterface):
    def __init__(self, url):
        TgiInterface.__init__(self, url, LLAMA2_DEFAULT_PROMPT_FORMAT)


###############################################################################
#
# Mixtral
#
################################################################
#
# This is the formatting that mixtral's instruction model is trained on.
# https://www.promptingguide.ai/models/mixtral#prompt-engineering-guide-for-mixtral-8x7b
MIXTRAL_DEFAULT_PROMPT_FORMAT = "[INST]%(instruction)s \n%(context)s [/INST]"


class MixtralInterface(TgiInterface):
    def __init__(self, url):
        TgiInterface.__init__(self, url, MIXTRAL_DEFAULT_PROMPT_FORMAT)
