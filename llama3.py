import os
from dotenv import load_dotenv

from src.symptom_study_strategies import build_strategies
from src.models import LLAMA3Model
from src.processor import NoteProcessor

load_dotenv(".env.llama3")

###############################################################################
#
# Build model and note processor
#
URL = os.getenv("VLLM_ENDPOINT")
DEPLOYMENT = os.getenv("VLLM_DEPLOYMENT")
model = LLAMA3Model(url=URL, deployment=DEPLOYMENT)
note_processor = NoteProcessor(model, "./note_config/llama3.json", sleep=False)


###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies()
tuning_exp = {
    # JSON
    "prompt-llama3-IdentityJSON": all_strategies["identityJSON"],
    "prompt-llama3-RulesJSON": all_strategies["rulesJSON"],
    "prompt-llama3-IncludeJSON": all_strategies["includeJSON"],
    "prompt-llama3-ExcludeJSON": all_strategies["excludeJSON"],
    "prompt-llama3-VerboseJSON": all_strategies["verboseJSON"],

}

# analysis_exp = {
#     "symptomstudy-llama3-IdentityJSON": all_strategies["identityJSON"],
# }


if __name__ == "__main__":
    note_processor.run_prompt_tuning(
        experiment=tuning_exp, experiment_name="llama3-tuning"
    )
    # note_processor.run_analysis(
    #     experiment=analysis_exp, experiment_name="llama3-analysis"
    # )
