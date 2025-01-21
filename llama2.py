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
    "development-llama2-Identity": all_strategies["identity"],
    "development-llama2-Rules": all_strategies["rules"],
    "development-llama2-Include": all_strategies["include"],
    "development-llama2-Exclude": all_strategies["exclude"],
    "development-llama2-Verbose": all_strategies["verbose"],
    # DoublePass
    "development-llama2-IdentityDoublePass": all_strategies["identityDoublePass"],
    "development-llama2-RulesDoublePass": all_strategies["rulesDoublePass"],
    "development-llama2-IncludeDoublePass": all_strategies["includeDoublePass"],
    "development-llama2-ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "development-llama2-VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "development-llama2-IdentityJSON": all_strategies["identityJSON"],
    "development-llama2-RulesJSON": all_strategies["rulesJSON"],
    "development-llama2-IncludeJSON": all_strategies["includeJSON"],
    "development-llama2-ExcludeJSON": all_strategies["excludeJSON"],
    "development-llama2-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "development-llama2-IdentityJSONDoublePass": all_strategies["identityJSONDoublePass"],
    "development-llama2-RulesJSONDoublePass": all_strategies["rulesJSONDoublePass"],
    "development-llama2-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "development-llama2-ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "development-llama2-VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

analysis_exp = {
    "test-llama2-IdentityJSON": all_strategies["IdentityJSON"],
}


if __name__ == "__main__":
    # note_processor.run_prompt_tuning(
    #     experiment=tuning_exp, experiment_name="llama2-tuning"
    # )
    note_processor.run_analysis(
        experiment=analysis_exp, experiment_name="llama2-analysis"
    )
