import re
import os
from dotenv import load_dotenv
from src.strategy import Strategy
from src.model_interface import LLAMA2Interface
from src.processor import process_small_batch, process_dir
from src.instructions import simple_instruction

###############################################################################
#
# Use LLAMA2 as our interface
# 
load_dotenv()
URL=f"{os.environ['HOST']}:{os.environ['PORT']}/"
llama2 = LLAMA2Interface(URL)


###############################################################################
#
# Custom preprocessor: whitespace normalization
# 
# Find all instances of multiple white space repeated
pattern = r"(\s){2,}"
# Replace it with one instance via capture group
replacement = r"\1"
def whitespace_normalize(s: str): 
    return re.sub(pattern, replacement, s).strip()


###############################################################################
#
# Simplification second-pass
# 
simplify_instruction = """
You are an expert editor reviewing a clinical note summary. 
Previous reviewer may have included irrelevant, negative symptoms in their summarization. 

Remove mentions of negative symptoms from this summary (e.g. "No X, No recent Y, No recent changes in Z").
ONLY reply with the summary. 
Do NOT explain your answers.
"""

###############################################################################
#
# Defining strategies with the components above!
# 
simpleDoublePassStrategy = Strategy([
    {
        "instruction": simple_instruction(),
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
        "instruction": simple_instruction(),
        "preprocess": whitespace_normalize,
    }
], model=llama2)


###############################################################################
#
# Building experiment with strategies 
# 
experiment = { 
  "simpleDoublePassStrategy": simpleDoublePassStrategy,
  "simpleSinglePassStrategy": simpleSinglePassStrategy,
}

if __name__ == "__main__":
  # process_small_batch(experiment=experiment)
  process_dir(experiment=experiment)