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
    # Simplification
    "development-llama2-IdentitySimplification": all_strategies["identitySimplification"],
    "development-llama2-RulesSimplification": all_strategies["rulesSimplification"],
    "development-llama2-IncludeSimplification": all_strategies["includeSimplification"],
    "development-llama2-ExcludeSimplification": all_strategies["excludeSimplification"],
    "development-llama2-VerboseSimplification": all_strategies["verboseSimplification"],
    # JSON
    "development-llama2-IdentityJSON": all_strategies["identityJSON"],
    "development-llama2-RulesJSON": all_strategies["rulesJSON"],
    "development-llama2-IncludeJSON": all_strategies["includeJSON"],
    "development-llama2-ExcludeJSON": all_strategies["excludeJSON"],
    "development-llama2-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Validation
    "development-llama2-IdentityJSONValidation": all_strategies["identityJSONValidation"],
    "development-llama2-RulesJSONValidation": all_strategies["rulesJSONValidation"],
    "development-llama2-IncludeJSONValidation": all_strategies["includeJSONValidation"],
    "development-llama2-ExcludeJSONValidation": all_strategies["excludeJSONValidation"],
    "development-llama2-VerboseJSONValidation": all_strategies["verboseJSONValidation"],
}

analysis_exp = {
    "test-llama2-IdentityJSON": all_strategies["identityJSON"],
}


if __name__ == "__main__":
    # note_processor.run_prompt_tuning(
    #     experiment=tuning_exp, experiment_name="llama2-tuning"
    # )
    note_processor.run_analysis(
        experiment=analysis_exp, experiment_name="llama2-analysis"
    )
