import requests

###############################################################################
#
# TGIClient 
# A client for making requests to the Text Generation Inference API 
# See here for more: https://huggingface.co/docs/text-generation-inference/index
#
################################################################
class TGIClient: 
    def __init__(self, url):
        self.url = url

    # Makes API call and parses response
    def fetch_llm_response(self, payload): 
        response = requests.post(self.url, json={
            "inputs": payload,
            "options": {
                "wait_for_model": True,
            },
            "parameters": {
                "max_new_tokens": 1000,
            },
        })
        response.raise_for_status()

        answer = response.json()[0]["generated_text"]
        
        # In case the answer includes the prompt
        if answer.startswith(payload):
            answer = answer[len(payload):].strip()

        return answer

class CommonLLMInterface(): 
    def __init__(self, url, default_prompt_format):
        self.url = url
        self.tgiClient = TGIClient(url)
        self.default_prompt_format = default_prompt_format

    # Fills the model's prompt-format with instructions, context, and system information
    def saturate_prompt(self, instruction: str, context: str, system:str = None,  prompt_format: str = None):
        prompt_format = prompt_format or self.default_prompt_format
        full_prompt = prompt_format % { 
            "instruction": instruction, 
            "context": context,
            # System might be None
            "system": system
        }
        return full_prompt

    def fetch_llm_response(self, instruction: str, context:str, prompt_format: str = None, system:str = None):
        payload = self.saturate_prompt(
            prompt_format=prompt_format, 
            instruction=instruction, 
            context=context, 
            system=system
        )
        return self.tgiClient.fetch_llm_response(payload)



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
class LLAMA2Interface(CommonLLMInterface): 
    def __init__(self, url):
        CommonLLMInterface.__init__(self, url, LLAMA2_DEFAULT_PROMPT_FORMAT)

###############################################################################
#
# Mixtral 
#
################################################################
# 
# This is the formatting that Llama2's chat model is trained on.
# https://huggingface.co/blog/llama2#how-to-prompt-llama-2
MIXTRAL_DEFAULT_PROMPT_FORMAT = "[INST]%(instruction)s \n%(context)s [/INST]"
class MixtralInterface(CommonLLMInterface): 
    def __init__(self, url):
        CommonLLMInterface.__init__(self, url, MIXTRAL_DEFAULT_PROMPT_FORMAT)

