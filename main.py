import re
from src.Strategy import Strategy
from src.ModelInterface import LLAMA2Interface
from src.processor import process_small_batch, process_dir
from src.instructions import simple_prompt

# For sans-notebook context running on the same network
# NOTE: Make sure you've updated IPs as needed
HOST="http://10.38.4.72"
PORT='8888'
# For notebook version working via ssh forwarding
# HOST="http://localhost"
# PORT='8086'
URL=f"{HOST}:{PORT}/"

# Use LLAMA2 as our interface
llama2 = LLAMA2Interface(URL)

# Find all instances of multiple white space repeated
pattern = r"(\s){2,}"
# Replace it with one instance via capture group
replacement = r"\1"
def whitespace_normalize(s: str): 
    return re.sub(pattern, replacement, s).strip()

simplify_instruction = """
You are an expert editor reviewing a clinical note summary. 
Previous reviewer may have included irrelevant, negative symptoms in their summarization. 

Remove mentions of negative symptoms from this summary (e.g. "No X, No recent Y, No recent changes in Z").
ONLY reply with the summary. 
Do NOT explain your answers.
"""

simpleDoublePassStrategy = Strategy([
    {
        "instruction": simple_prompt(),
        "preprocess": whitespace_normalize,
    },
    {
        "step_type": "previous",
        "instruction": simplify_instruction,
        "preprocess": whitespace_normalize,
    }
], model=llama2)

simpleSinglePassStrategy = Strategy([
    {
        "instruction": simple_prompt(),
        "preprocess": whitespace_normalize,
    }
], model=llama2)

experiment = { 
  "simpleDoublePassStrategy": simpleDoublePassStrategy,
  "simpleSinglePassStrategy": simpleSinglePassStrategy,
}

if __name__ == "__main__":
  # process_small_batch(experiment=experiment)
  process_dir(experiment=experiment)