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
    # DoublePass
    "development-gpt4Turbo-IdentityDoublePass": all_strategies["identityDoublePass"],
    "development-gpt4Turbo-RulesDoublePass": all_strategies["rulesDoublePass"],
    "development-gpt4Turbo-IncludeDoublePass": all_strategies["includeDoublePass"],
    "development-gpt4Turbo-ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "development-gpt4Turbo-VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "development-gpt4Turbo-IdentityJSON": all_strategies["identityJSON"],
    "development-gpt4Turbo-RulesJSON": all_strategies["rulesJSON"],
    "development-gpt4Turbo-IncludeJSON": all_strategies["includeJSON"],
    "development-gpt4Turbo-ExcludeJSON": all_strategies["excludeJSON"],
    "development-gpt4Turbo-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "development-gpt4Turbo-IdentityJSONDoublePass": all_strategies["identityJSONDoublePass"],
    "development-gpt4Turbo-RulesJSONDoublePass": all_strategies["rulesJSONDoublePass"],
    "development-gpt4Turbo-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "development-gpt4Turbo-ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "development-gpt4Turbo-VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

analysis_exp = {
    "test-gpt4Turbo-IncludeJSON": all_strategies["IncludeJSON"],
}

if __name__ == "__main__":
    # note_processor.run_prompt_tuning(
    #     experiment=tuning_exp, experiment_name="gpt4-tuning"
    # )
    note_processor.run_analysis(
        experiment=analysis_exp, experiment_name="gpt4-analysis"
    )
