import os
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
    def __init__(self, url=None, api_key=None, model_type=None):
        self.client = AzureOpenAI(
            azure_endpoint=url or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
        )
        self.model_type = model_type or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.prompt_format = (
            "### Instructions ###\n%(instruction)s ### Text ###\n%(context)s"
        )

    def get_model_info(self):
        raise NotImplementedError
        # return self.client.models.retrieve(self.model_type)

    def fetch_llm_response(
        self,
        instruction: str,
        context: str,
    ):
        payload = self.saturate_prompt(
            instruction=instruction,
            context=context,
        )
        response = self.client.chat.completions.create(
            model=self.model_type,  # model = "deployment_name".
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": payload},
            ],
        )
        # Based on https://platform.openai.com/docs/guides/text-generation/chat-completions-response-format
        return {
            "text": response.choices[0].message.content,
            "stats": {"total_tokens": response.usage.total_tokens},
        }
