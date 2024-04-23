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
    def __init__(self, url=None, model=None):
        self.client = AzureOpenAI(
            azure_endpoint=url or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
        )
        self.model = model or os.getenv("AZURE_OPENAI_DEPLOYMENT")
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
        # Based on https://platform.openai.com/docs/guides/text-generation/chat-completions-response-format
        return {
            "text": response.choices[0].message.content,
            "stats": {"total_tokens": response.usage.total_tokens},
        }
