from src.models.tgi_interface import TgiInterface

###############################################################################
#
# LLAMA3
#
################################################################
#
# This is the formatting that Llama2's chat model is trained on.
# Based on: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
LLAMA3_DEFAULT_PROMPT_FORMAT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

%(instruction)s<|eot_id|><|start_header_id|>user<|end_header_id|>

%(context)s <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


class LLAMA3Model(TgiInterface):
    def __init__(self, url):
        TgiInterface.__init__(self, url, LLAMA3_DEFAULT_PROMPT_FORMAT)
