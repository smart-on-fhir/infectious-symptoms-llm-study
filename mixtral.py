import os
from dotenv import load_dotenv

from src.symptom_study_strategies import build_strategies
from src.models import MixtralModel
from src.processor import NoteProcessor

load_dotenv(".env.mixtral")

###############################################################################
#
# Build model and note processor
#
URL = os.getenv("TGI_URL")
model = MixtralModel(url=URL)
note_processor = NoteProcessor(model, "./note_config/open_llm.json", sleep=False)

###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies()
tuning_exp = {
    "development-mixtral-Identity": all_strategies["identity"],
    "development-mixtral-Rules": all_strategies["rules"],
    "development-mixtral-Include": all_strategies["include"],
    "development-mixtral-Exclude": all_strategies["exclude"],
    "development-mixtral-Verbose": all_strategies["verbose"],
    # DoublePass
    "development-mixtral-IdentityDoublePass": all_strategies["identityDoublePass"],
    "development-mixtral-RulesDoublePass": all_strategies["rulesDoublePass"],
    "development-mixtral-IncludeDoublePass": all_strategies["includeDoublePass"],
    "development-mixtral-ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "development-mixtral-VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "development-mixtral-IdentityJSON": all_strategies["identityJSON"],
    "development-mixtral-RulesJSON": all_strategies["rulesJSON"],
    "development-mixtral-IncludeJSON": all_strategies["includeJSON"],
    "development-mixtral-ExcludeJSON": all_strategies["excludeJSON"],
    "development-mixtral-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "development-mixtral-IdentityJSONDoublePass": all_strategies["identityJSONDoublePass"],
    "development-mixtral-RulesJSONDoublePass": all_strategies["rulesJSONDoublePass"],
    "development-mixtral-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "development-mixtral-ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "development-mixtral-VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

analysis_exp = {
    "test-mixtral-ExcludeJSON": all_strategies["excludeJSON"],
}

if __name__ == "__main__":
    # note_processor.run_prompt_tuning(
    #     experiment=tuning_exp, experiment_name="mixtral-tuning"
    # )
    note_processor.run_analysis(
        experiment=analysis_exp, experiment_name="mixtral-analysis"
    )
