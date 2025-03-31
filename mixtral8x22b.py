import os
from dotenv import load_dotenv

from src.symptom_study_strategies import build_strategies
from src.models import MixtralModel
from src.processor import NoteProcessor

load_dotenv(".env.mixtral8x22b")

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
    "development-mixtral8x22b-Identity": all_strategies["identity"],
    "development-mixtral8x22b-Rules": all_strategies["rules"],
    "development-mixtral8x22b-Include": all_strategies["include"],
    "development-mixtral8x22b-Exclude": all_strategies["exclude"],
    "development-mixtral8x22b-Verbose": all_strategies["verbose"],
    # Simplification
    "development-mixtral8x22b-IdentitySimplification": all_strategies["identitySimplification"],
    "development-mixtral8x22b-RulesSimplification": all_strategies["rulesSimplification"],
    "development-mixtral8x22b-IncludeSimplification": all_strategies["includeSimplification"],
    "development-mixtral8x22b-ExcludeSimplification": all_strategies["excludeSimplification"],
    "development-mixtral8x22b-VerboseSimplification": all_strategies["verboseSimplification"],
    # JSON
    "development-mixtral8x22b-IdentityJSON": all_strategies["identityJSON"],
    "development-mixtral8x22b-RulesJSON": all_strategies["rulesJSON"],
    "development-mixtral8x22b-IncludeJSON": all_strategies["includeJSON"],
    "development-mixtral8x22b-ExcludeJSON": all_strategies["excludeJSON"],
    "development-mixtral8x22b-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Validation
    "development-mixtral8x22b-IdentityJSONValidation": all_strategies[
        "identityJSONValidation"
    ],
    "development-mixtral8x22b-RulesJSONValidation": all_strategies["rulesJSONValidation"],
    "development-mixtral8x22b-IncludeJSONValidation": all_strategies[
        "includeJSONValidation"
    ],
    "development-mixtral8x22b-ExcludeJSONValidation": all_strategies[
        "excludeJSONValidation"
    ],
    "development-mixtral8x22b-VerboseJSONValidation": all_strategies[
        "verboseJSONValidation"
    ],
}

analysis_exp = {
    "test-mixtral8x22b-RulesJSON": all_strategies["rulesJSON"],
}


if __name__ == "__main__":
    # note_processor.run_prompt_tuning(
    #     experiment=tuning_exp, experiment_name="mixtral8x22b-tuning"
    # )
    note_processor.run_analysis(
        experiment=analysis_exp, experiment_name="mixtral8x22b-analysis"
    )
