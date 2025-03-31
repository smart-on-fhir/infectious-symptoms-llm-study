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
    # Simplification
    "development-mixtral-IdentitySimplification": all_strategies["identitySimplification"],
    "development-mixtral-RulesSimplification": all_strategies["rulesSimplification"],
    "development-mixtral-IncludeSimplification": all_strategies["includeSimplification"],
    "development-mixtral-ExcludeSimplification": all_strategies["excludeSimplification"],
    "development-mixtral-VerboseSimplification": all_strategies["verboseSimplification"],
    # JSON
    "development-mixtral-IdentityJSON": all_strategies["identityJSON"],
    "development-mixtral-RulesJSON": all_strategies["rulesJSON"],
    "development-mixtral-IncludeJSON": all_strategies["includeJSON"],
    "development-mixtral-ExcludeJSON": all_strategies["excludeJSON"],
    "development-mixtral-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "development-mixtral-IdentityJSONValidation": all_strategies["identityJSONValidation"],
    "development-mixtral-RulesJSONValidation": all_strategies["rulesJSONValidation"],
    "development-mixtral-IncludeJSONValidation": all_strategies["includeJSONValidation"],
    "development-mixtral-ExcludeJSONValidation": all_strategies["excludeJSONValidation"],
    "development-mixtral-VerboseJSONValidation": all_strategies["verboseJSONValidation"],
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
