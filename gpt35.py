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
model = AzureGptModel(url=url, api_key=api_key, api_version=api_version model_type=model_type)
note_processor = NoteProcessor(model, "./note_config/gpt_api.json", sleepRate=2)

###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies()

tuning_exp = {
    "prompt-gpt35Turbo-Identity": all_strategies["identity"],
    "prompt-gpt35Turbo-Rules": all_strategies["rules"],
    "prompt-gpt35Turbo-Include": all_strategies["include"],
    "prompt-gpt35Turbo-Exclude": all_strategies["exclude"],
    "prompt-gpt35Turbo-Verbose": all_strategies["verbose"],
    # DoublePass
    "prompt-gpt35Turbo-IdentityDoublePass": all_strategies["identityDoublePass"],
    "prompt-gpt35Turbo-RulesDoublePass": all_strategies["rulesDoublePass"],
    "prompt-gpt35Turbo-IncludeDoublePass": all_strategies["includeDoublePass"],
    "prompt-gpt35Turbo-ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "prompt-gpt35Turbo-VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "prompt-gpt35Turbo-IdentityJSON": all_strategies["identityJSON"],
    "prompt-gpt35Turbo-RulesJSON": all_strategies["rulesJSON"],
    "prompt-gpt35Turbo-IncludeJSON": all_strategies["includeJSON"],
    "prompt-gpt35Turbo-ExcludeJSON": all_strategies["excludeJSON"],
    "prompt-gpt35Turbo-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "prompt-gpt35Turbo-IdentityJSONDoublePass": all_strategies[
        "identityJSONDoublePass"
    ],
    "prompt-gpt35Turbo-RulesJSONDoublePass": all_strategies["rulesJSONDoublePass"],
    "prompt-gpt35Turbo-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "prompt-gpt35Turbo-ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "prompt-gpt35Turbo-VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

analysis_exp = {
    "symptomstudy-gpt35Turbo-RulesJSON": all_strategies["rulesJSON"],
}

if __name__ == "__main__":
    note_processor.run_prompt_tuning(
        experiment=tuning_exp, experiment_name="gpt3-tuning"
    )
    note_processor.run_analysis(
        experiment=analysis_exp, experiment_name="gpt3-analysis"
    )
