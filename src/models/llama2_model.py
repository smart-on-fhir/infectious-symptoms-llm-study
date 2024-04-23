from src.models.tgi_interface import TgiInterface

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


class LLAMA2Model(TgiInterface):
    def __init__(self, url):
        TgiInterface.__init__(self, url, LLAMA2_DEFAULT_PROMPT_FORMAT)