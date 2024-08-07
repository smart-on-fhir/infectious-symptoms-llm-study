import requests


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
    def fetch_llm_response(self, payload, request_parameters={}):
        # Common default parameters used by all models
        default_parameters = {
            "details": True,
            "max_new_tokens": 1000,
        }
        response = requests.post(
            self.url,
            json={
                "inputs": payload,
                "parameters": {
                    **default_parameters,
                    **request_parameters,
                },
            },
        )
        response.raise_for_status()

        # Based on https://huggingface.github.io/text-generation-inference/#/
        json = response.json()[0]
        text = json["generated_text"]
        total_tokens = (
            len(json["details"]["prefill"]) + json["details"]["generated_tokens"]
        )

        # In case the answer includes the prompt
        if text.startswith(payload):
            text = text[len(payload) :].strip()

        return {"text": text, "stats": {"total_tokens": total_tokens}}
