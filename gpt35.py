import os
from dotenv import load_dotenv

from src.symptom_study_strategies import build_strategies
from src.models import AzureGptModel
from src.processor import NoteProcessor

load_dotenv(".env.gpt35")

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
note_processor = NoteProcessor(model, "./note_config/gpt_api.json", sleepRate=2)

###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies()

tuning_exp = {
    "development-gpt35Turbo-Identity": all_strategies["identity"],
    "development-gpt35Turbo-Rules": all_strategies["rules"],
    "development-gpt35Turbo-Include": all_strategies["include"],
    "development-gpt35Turbo-Exclude": all_strategies["exclude"],
    "development-gpt35Turbo-Verbose": all_strategies["verbose"],
    # DoublePass
    "development-gpt35Turbo-IdentityDoublePass": all_strategies["identityDoublePass"],
    "development-gpt35Turbo-RulesDoublePass": all_strategies["rulesDoublePass"],
    "development-gpt35Turbo-IncludeDoublePass": all_strategies["includeDoublePass"],
    "development-gpt35Turbo-ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "development-gpt35Turbo-VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "development-gpt35Turbo-IdentityJSON": all_strategies["identityJSON"],
    "development-gpt35Turbo-RulesJSON": all_strategies["rulesJSON"],
    "development-gpt35Turbo-IncludeJSON": all_strategies["includeJSON"],
    "development-gpt35Turbo-ExcludeJSON": all_strategies["excludeJSON"],
    "development-gpt35Turbo-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "development-gpt35Turbo-IdentityJSONDoublePass": all_strategies[
        "identityJSONDoublePass"
    ],
    "development-gpt35Turbo-RulesJSONDoublePass": all_strategies["rulesJSONDoublePass"],
    "development-gpt35Turbo-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "development-gpt35Turbo-ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "development-gpt35Turbo-VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

analysis_exp = {
    "test-gpt35Turbo-IdentityJSON": all_strategies["IdentityJSON"],
}

if __name__ == "__main__":
    # note_processor.run_prompt_tuning(
    #     experiment=tuning_exp, experiment_name="gpt3-tuning"
    # )
    note_processor.run_analysis(
        experiment=analysis_exp, experiment_name="gpt3-analysis"
    )
