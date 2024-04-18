from .step import Step


class Strategy:
    def __init__(self, steps, model):
        self.model = model
        self.responses = []
        self.steps = [
            Step(**step_config, model=model, responses=self.responses)
            for step_config in steps
        ]

    def toJSON(self):
        return [step.toJSON() for step in self.steps]

    def run(self, context):
        self.responses.clear()
        for step in self.steps:
            response = step.run(context)
            self.responses.append(response)
        # Only the final response is returned
        return self.responses[-1]
