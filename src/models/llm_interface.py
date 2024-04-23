###############################################################################
#
# Abstract Llm Interface Class
#
###############################################################################
class LlmInterface:
    def get_model_info(self):
        pass

    def saturate_prompt(
        self,
        instruction: str,
        context: str,
        system: str = None,
        prompt_format: str = None,
    ):
        pass

    def fetch_llm_response(
        self,
        instruction: str,
        context: str,
        prompt_format: str = None,
        system: str = None,
    ):
        pass
