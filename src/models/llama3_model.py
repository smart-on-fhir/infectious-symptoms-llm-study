from huggingface_hub import InferenceClient
from src.models.llm_interface import LlmInterface

###############################################################################
#
# LLAMA3 using huggingface_hub
#
################################################################

class LLAMA3Model(LlmInterface):
    def __init__(self, url):
        self.url = url
        self.client = InferenceClient(base_url=url)

    def get_model_info(self):
        return self.client.get_endpoint_info()
    
    def fetch_llm_response(self, instruction: str, context: str):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": context}
        ]
        
        response = self.client.chat_completion(messages)
        return  {
            "text": response["choices"][0]["message"]["content"], 
            "stats": {
                "total_tokens": response["usage"]["total_tokens"]
            }
        } 
