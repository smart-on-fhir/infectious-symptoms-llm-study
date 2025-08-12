import os
from dotenv import load_dotenv

from src.symptom_study_strategies import build_strategies
from src.models import GptOssModel
from src.processor import NoteProcessor

load_dotenv(".env.gptoss")

###############################################################################
#
# Build model and note processor
#
URL = os.getenv("VLLM_ENDPOINT")
DEPLOYMENT = os.getenv("VLLM_DEPLOYMENT")
model = GptOssModel(url=URL, deployment=DEPLOYMENT)
note_processor = NoteProcessor(model, "./note_config/gptoss.json", sleep=False)


###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies()
tuning_exp = {
    # JSON
    "prompt-gptoss-IdentityJSON": all_strategies["identityJSON"],
    "prompt-gptoss-RulesJSON": all_strategies["rulesJSON"],
    "prompt-gptoss-IncludeJSON": all_strategies["includeJSON"],
    "prompt-gptoss-ExcludeJSON": all_strategies["excludeJSON"],
    "prompt-gptoss-VerboseJSON": all_strategies["verboseJSON"],

}

# analysis_exp = {
#     "symptomstudy-gptoss-IdentityJSON": all_strategies["identityJSON"],
# }


if __name__ == "__main__":
    note_processor.run_prompt_tuning(
        experiment=tuning_exp, experiment_name="gptoss-tuning"
    )
    # note_processor.run_analysis(
    #     experiment=analysis_exp, experiment_name="gptoss-analysis"
    # )
