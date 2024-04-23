from src.models.tgi_interface import TgiInterface

###############################################################################
#
# Mixtral
#
################################################################
#
# This is the formatting that mixtral's instruction model is trained on.
# https://www.promptingguide.ai/models/mixtral#prompt-engineering-guide-for-mixtral-8x7b
MIXTRAL_DEFAULT_PROMPT_FORMAT = "[INST]%(instruction)s \n%(context)s [/INST]"

class MixtralModel(TgiInterface):
    def __init__(self, url):
        TgiInterface.__init__(self, url, MIXTRAL_DEFAULT_PROMPT_FORMAT)

