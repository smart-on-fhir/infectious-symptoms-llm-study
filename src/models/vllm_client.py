import os
from typing import Self
from openai import OpenAI, ChatCompletion
from src.models.llm_interface import LlmInterface

###############################################################################
#
# LLAMA3 using huggingface_hub
#
################################################################

class VllmClient(LlmInterface):
    def __init__(self, url: str | None, deployment: str | None):
        self.url = url or os.getenv("VLLM_ENDPOINT")
        self.model_type = deployment or os.getenv("VLLM_DEPLOYMENT")
        self._api_key = os.getenv("VLLM_API_KEY") or "EMPTY"
        self.client = OpenAI(
            base_url=self.url,
            api_key=self._api_key,
        )
        print(self.client)

    def _process_response(self: Self, response: ChatCompletion):
        """
        Process an LLM ChatCompletion to store relevant logging/usage information
        and access the relevant Pydantic data in its response
        """
        # Based on https://platform.openai.com/docs/guides/text-generation/chat-completions-response-format
        usage = response.usage
        total_tokens = usage.total_tokens
        response_message = response.choices[0].message
        text = response_message.content

        return {
            "text": text,
            "stats": {
                "total_tokens": total_tokens
            }
        }

    def get_model_info(self):
        return self.client.get_endpoint_info()
    
    def fetch_llm_response(self, instruction: str, context: str):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": context}
        ]
        
        response = self.client.chat.completions.create(               
            model=self.model_type,
            messages=messages,
        )
        return self._process_response(response)
