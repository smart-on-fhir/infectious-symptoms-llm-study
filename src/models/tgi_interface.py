import requests
from src.models.llm_interface import LlmInterface
from src.models.tgi_client import TgiClient


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
        response = self.client.fetch_llm_response(payload)
        return response
