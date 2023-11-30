# Default pre-process is the identity fn
default_preprocess = lambda s : s
class Step:
    def __init__(self, instruction, model, responses, step_type : str = "default", preprocess = default_preprocess, prompt_format : str = None):
        # Always strip the resulting string
        self.preprocess = lambda s: preprocess(s).strip()
        # Process the instruction 
        self.instruction = preprocess(instruction)
        # Should be covered in the run methods; "default" by default
        self.step_type = step_type
        # Could be empty
        self.prompt_format = prompt_format
        # Must be specified
        self.model = model
        # Array of responses is passed by reference for future use
        self.responses = responses
    
    def toJSON(self):
        return {
            "instruction": self.instruction, 
            "step_type": self.step_type,
            "prompt_format": self.prompt_format or "No custom prompt format",
            "preprocess": "default" if self.preprocess is default_preprocess else "custom"
        }

    def run(self, default_context):
        if self.step_type == "default": 
            cleaned_context = self.preprocess(default_context)
        elif self.step_type == 'previous':
            # Previous input type means we should look at the last response
            cleaned_context = self.preprocess(self.responses[-1])
        elif self.step_type == 'aggregator':
            # Aggregator input type means we should aggregate all responses so far into our context
            cleaned_context = self.preprocess('\n'.join(self.responses))
        else:
            raise ValueError('Unrecognized input type ' + self.step_type)
        return self.model.call(prompt_format=self.prompt_format, instruction=self.instruction, context=cleaned_context)
