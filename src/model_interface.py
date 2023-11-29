import requests

# For sans-notebook context running on the same network
HOST="http://10.38.4.72"
PORT='8888'
# For notebook version working via ssh forwarding
# HOST="http://localhost"
# PORT='8086'

###############################################################################
#
# TGIClient 
# A client for making requests to the TGI API 
#
################################################################
class TGIClient: 
    def __init__(self, url):
        self.url = url

    # Makes API call and parses response
    def call(self, payload): 
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
        
        # The answer includes the prompt, to make it easier to feed previous
        # history back to llama2 so it learns from a conversation. But we are
        # designing here for a single request, not a conversation.
        if answer.startswith(payload):
            answer = answer[len(payload):].strip()

        return answer

###############################################################################
#
# LLAMA2 
# An interface for LLAMA2 models
#
################################################################
# 
# This is the formatting that Llama2's chat model is trained on.
# https://huggingface.co/blog/llama2#how-to-prompt-llama-2
DEFAULT_PROMPT_FORMAT = """<s>[INST] <<SYS>>
%(instruction)s
<</SYS>>

%(context)s [/INST]"""
class LLAMA2Interface(): 
    def __init__(self, url):
        self.url = url or f"{HOST}:{PORT}/"
        self.tgiClient = TGIClient(url)
        # This is the formatting that Llama2's chat model is trained on.
        # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        self.default_prompt_format = DEFAULT_PROMPT_FORMAT

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

    def call(self, instruction: str, context:str, prompt_format: str = None, system:str = None):
        payload = self.saturate_prompt(
            prompt_format=prompt_format, 
            instruction=instruction, 
            context=context, 
            system=system
        )
        return self.tgiClient.call(payload)

