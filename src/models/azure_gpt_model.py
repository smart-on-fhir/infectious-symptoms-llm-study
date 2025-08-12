import os
import re
from dotenv import load_dotenv
from openai import AzureOpenAI
from src.models.llm_interface import LlmInterface

load_dotenv()


###############################################################################
#
# Gpt3Interface using the AzureOpenAI client
#
###############################################################################
class AzureGptModel(LlmInterface):
    def __init__(self, url=None, api_key=None, api_version=None, model_type=None, REMOVE_WS_FLAG: bool = False):
        self.client = AzureOpenAI(
            azure_endpoint=url or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        self.model_type = model_type or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.prompt_format = (
            "### Instructions ###\n%(instruction)s ### Text ###\n%(context)s"
        )
        self.REMOVE_WS_FLAG = REMOVE_WS_FLAG

    def get_model_info(self):
        raise NotImplementedError


    def remove_trailing_whitespace(self, text):
        return re.sub(r"\s+$", "", text, flags=re.MULTILINE)


    def fetch_llm_response(
        self,
        instruction: str,
        context: str,
    ):
        payload = self.saturate_prompt(
            instruction=instruction,
            context=context,
        )
        if self.REMOVE_WS_FLAG: 
            payload = self.remove_trailing_whitespace(payload)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": payload},
        ]
        response = self.client.chat.completions.create(
            model=self.model_type,  # model = "deployment_name".
            messages=messages,
        )
        # Based on https://platform.openai.com/docs/guides/text-generation/chat-completions-response-format
        return {
            "text": response.choices[0].message.content,
            "stats": {"total_tokens": response.usage.total_tokens},
        }
