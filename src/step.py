def default_preprocess(s):
    return s


class Step:
    def __init__(
        self,
        instruction,
        responses,
        step_type: str = "default",
    ):
        self.instruction = instruction
        # Should be covered in the run methods; "default" by default
        self.step_type = step_type
        # Array of responses is passed by reference for future use/storage
        self.responses = responses

    def to_json(self):
        return {
            "instruction": self.instruction,
            "step_type": self.step_type,
        }

    def run(self, model, context):
        if self.step_type == "default":
            normalized_context = context
        elif self.step_type == "previous":
            # Previous input type means we should look at the last response
            normalized_context = self.responses[-1]
        elif self.step_type == "aggregator":
            # Aggregator input type means we should aggregate all responses so far into our context
            normalized_context = "\n".join(self.responses)
        else:
            raise ValueError("Unrecognized input type " + self.step_type)
        return model.fetch_llm_response(
            instruction=self.instruction,
            context=normalized_context,
        )
