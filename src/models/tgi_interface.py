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
    def __init__(self, url, prompt_format):
        self.url = url
        self.client = TgiClient(url)
        self.prompt_format = prompt_format

    def get_model_info(self):
        return requests.get(self.url + "info").json()

    def fetch_llm_response(
        self,
        instruction: str,
        context: str,
    ):
        payload = self.saturate_prompt(
            instruction=instruction,
            context=context,
        )
        response = self.client.fetch_llm_response(payload)
        return response
