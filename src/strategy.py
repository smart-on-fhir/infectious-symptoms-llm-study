from .step import Step


class Strategy:
    def __init__(self, steps, model):
        self.model = model
        self.responses = []
        self.steps = [
            Step(**step_config, model=model, responses=self.responses)
            for step_config in steps
        ]
        self.total_tokens = 0

    def toJSON(self):
        return [step.toJSON() for step in self.steps]

    def run(self, context):
        self.total_tokens = 0 
        self.responses.clear()
        for step in self.steps:
            response = step.run(context)
            # TODO: fix the confusion b/t responses and responses' text
            self.responses.append(response["text"])
            self.total_tokens += response["stats"]["total_tokens"]
        # Only the final response is returned
        return {"text": self.responses[-1], "total_tokens": self.total_tokens}
