###############################################################################
#
# Abstract Llm Interface Class
#
###############################################################################
class LlmInterface:
    prompt_format: str

    def get_model_info(self):
        pass

    # Fills the model's prompt-format with instructions & context
    def saturate_prompt(
        self,
        instruction: str,
        context: str,
    ):
        return self.prompt_format % {
            "instruction": instruction,
            "context": context,
        }

    def fetch_llm_response(
        self,
        instruction: str,
        context: str,
    ):
        pass
