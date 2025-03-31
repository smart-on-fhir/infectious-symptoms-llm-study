import os
from dotenv import load_dotenv

from src.symptom_study_strategies import build_strategies
from src.models import AzureGptModel
from src.processor import NoteProcessor

load_dotenv(".env.gpt4")

###############################################################################
#
# Build model and note processor
#
url = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
model_type = os.getenv("AZURE_OPENAI_DEPLOYMENT")
model = AzureGptModel(
    url=url, api_key=api_key, api_version=api_version, model_type=model_type
)
note_processor = NoteProcessor(model, "./note_config/gpt_api.json", sleepRate=5)

###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies()

tuning_exp = {
    "development-gpt4Turbo-Identity": all_strategies["identity"],
    "development-gpt4Turbo-Rules": all_strategies["rules"],
    "development-gpt4Turbo-Include": all_strategies["include"],
    "development-gpt4Turbo-Exclude": all_strategies["exclude"],
    "development-gpt4Turbo-Verbose": all_strategies["verbose"],
    # Simplification
    "development-gpt4Turbo-IdentitySimplification": all_strategies["identitySimplification"],
    "development-gpt4Turbo-RulesSimplification": all_strategies["rulesSimplification"],
    "development-gpt4Turbo-IncludeSimplification": all_strategies["includeSimplification"],
    "development-gpt4Turbo-ExcludeSimplification": all_strategies["excludeSimplification"],
    "development-gpt4Turbo-VerboseSimplification": all_strategies["verboseSimplification"],
    # JSON
    "development-gpt4Turbo-IdentityJSON": all_strategies["identityJSON"],
    "development-gpt4Turbo-RulesJSON": all_strategies["rulesJSON"],
    "development-gpt4Turbo-IncludeJSON": all_strategies["includeJSON"],
    "development-gpt4Turbo-ExcludeJSON": all_strategies["excludeJSON"],
    "development-gpt4Turbo-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Validation
    "development-gpt4Turbo-IdentityJSONValidation": all_strategies["identityJSONValidation"],
    "development-gpt4Turbo-RulesJSONValidation": all_strategies["rulesJSONValidation"],
    "development-gpt4Turbo-IncludeJSONValidation": all_strategies["includeJSONValidation"],
    "development-gpt4Turbo-ExcludeJSONValidation": all_strategies["excludeJSONValidation"],
    "development-gpt4Turbo-VerboseJSONValidation": all_strategies["verboseJSONValidation"],
}

analysis_exp = {
    "test-gpt4Turbo-IncludeJSON": all_strategies["includeJSON"],
}

if __name__ == "__main__":
    # note_processor.run_prompt_tuning(
    #     experiment=tuning_exp, experiment_name="gpt4-tuning"
    # )
    note_processor.run_analysis(
        experiment=analysis_exp, experiment_name="gpt4-analysis"
    )
