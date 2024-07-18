import os
from dotenv import load_dotenv

from src.symptom_study_strategies import build_strategies
from src.models import LLAMA2Model
from src.processor import NoteProcessor

load_dotenv(".env.llama2")

###############################################################################
#
# Build model and note processor
#
URL = os.getenv("TGI_URL")
model = LLAMA2Model(url=URL)
note_processor = NoteProcessor(model, "./note_config/open_llm.json", sleep=False)


###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies()
tuning_exp = {
    "prompt-llama2-Identity": all_strategies["identity"],
    "prompt-llama2-Rules": all_strategies["rules"],
    "prompt-llama2-Include": all_strategies["include"],
    "prompt-llama2-Exclude": all_strategies["exclude"],
    "prompt-llama2-Verbose": all_strategies["verbose"],
    # DoublePass
    "prompt-llama2-IdentityDoublePass": all_strategies["identityDoublePass"],
    "prompt-llama2-RulesDoublePass": all_strategies["rulesDoublePass"],
    "prompt-llama2-IncludeDoublePass": all_strategies["includeDoublePass"],
    "prompt-llama2-ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "prompt-llama2-VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "prompt-llama2-IdentityJSON": all_strategies["identityJSON"],
    "prompt-llama2-RulesJSON": all_strategies["rulesJSON"],
    "prompt-llama2-IncludeJSON": all_strategies["includeJSON"],
    "prompt-llama2-ExcludeJSON": all_strategies["excludeJSON"],
    "prompt-llama2-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "prompt-llama2-IdentityJSONDoublePass": all_strategies["identityJSONDoublePass"],
    "prompt-llama2-RulesJSONDoublePass": all_strategies["rulesJSONDoublePass"],
    "prompt-llama2-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "prompt-llama2-ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "prompt-llama2-VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

analysis_exp = {
    "symptomstudy-llama2-IdentityJSON": all_strategies["IdentityJSON"],
}


if __name__ == "__main__":
    # note_processor.run_prompt_tuning(
    #     experiment=tuning_exp, experiment_name="llama2-tuning"
    # )
    note_processor.run_analysis(
        experiment=analysis_exp, experiment_name="llama2-analysis"
    )
