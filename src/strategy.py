from src.step import Step


class Strategy:
    def __init__(self, steps):
        self.responses = []
        self.steps = [
            Step(**step_config, responses=self.responses) for step_config in steps
        ]
        self.total_tokens = 0

    def to_json(self):
        return [step.to_json() for step in self.steps]

    def run(self, model, context):
        self.total_tokens = 0
        self.responses.clear()
        for step in self.steps:
            response = step.run(model, context)
            self.responses.append(response["text"])
            self.total_tokens += response["stats"]["total_tokens"]
        # Only the final response is returned
        print(self.responses)
        return {"text": self.responses[-1], "total_tokens": self.total_tokens}
