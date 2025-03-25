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
URL = os.getenv("TGI_URL")
model = LLAMA3Model(url=URL)
note_processor = NoteProcessor(model, "./note_config/open_llm.json", sleep=False)


###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies()
tuning_exp = {
    "prompt-llama3-Identity": all_strategies["identity"],
    "prompt-llama3-Rules": all_strategies["rules"],
    "prompt-llama3-Include": all_strategies["include"],
    "prompt-llama3-Exclude": all_strategies["exclude"],
    "prompt-llama3-Verbose": all_strategies["verbose"],
    # DoublePass
    "prompt-llama3-IdentityDoublePass": all_strategies["identityDoublePass"],
    "prompt-llama3-RulesDoublePass": all_strategies["rulesDoublePass"],
    "prompt-llama3-IncludeDoublePass": all_strategies["includeDoublePass"],
    "prompt-llama3-ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "prompt-llama3-VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "prompt-llama3-IdentityJSON": all_strategies["identityJSON"],
    "prompt-llama3-RulesJSON": all_strategies["rulesJSON"],
    "prompt-llama3-IncludeJSON": all_strategies["includeJSON"],
    "prompt-llama3-ExcludeJSON": all_strategies["excludeJSON"],
    "prompt-llama3-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "prompt-llama3-IdentityJSONDoublePass": all_strategies["identityJSONDoublePass"],
    "prompt-llama3-RulesJSONDoublePass": all_strategies["rulesJSONDoublePass"],
    "prompt-llama3-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "prompt-llama3-ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "prompt-llama3-VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

# analysis_exp = {
#     "symptomstudy-llama3-IdentityJSON": all_strategies["IdentityJSON"],
# }


if __name__ == "__main__":
    note_processor.run_prompt_tuning(
        experiment=tuning_exp, experiment_name="llama3-tuning"
    )
    # note_processor.run_analysis(
    #     experiment=analysis_exp, experiment_name="llama3-analysis"
    # )
