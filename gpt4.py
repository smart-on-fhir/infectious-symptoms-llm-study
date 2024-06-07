import os 
from functools import reduce
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
model_type = os.getenv("AZURE_OPENAI_DEPLOYMENT")
model = AzureGptModel(url=url, api_key=api_key, model_type=model_type)
note_processor = NoteProcessor(model, './note_config/gpt_api.json', sleepRate=5)

###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies()

tuning_exp = {
    "prompt-gpt4Turbo-Identity": all_strategies["identity"],
    "prompt-gpt4Turbo-Rules": all_strategies["rules"],
    "prompt-gpt4Turbo-Include": all_strategies["include"],
    "prompt-gpt4Turbo-Exclude": all_strategies["exclude"],
    "prompt-gpt4Turbo-Verbose": all_strategies["verbose"],
    # DoublePass
    "prompt-gpt4Turbo-IdentityDoublePass": all_strategies["identityDoublePass"],
    "prompt-gpt4Turbo-RulesDoublePass": all_strategies["rulesDoublePass"],
    "prompt-gpt4Turbo-IncludeDoublePass": all_strategies["includeDoublePass"],
    "prompt-gpt4Turbo-ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "prompt-gpt4Turbo-VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "prompt-gpt4Turbo-IdentityJSON": all_strategies["identityJSON"],
    "prompt-gpt4Turbo-RulesJSON": all_strategies["rulesJSON"],
    "prompt-gpt4Turbo-IncludeJSON": all_strategies["includeJSON"],
    "prompt-gpt4Turbo-ExcludeJSON": all_strategies["excludeJSON"],
    "prompt-gpt4Turbo-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "prompt-gpt4Turbo-IdentityJSONDoublePass": all_strategies["identityJSONDoublePass"],
    "prompt-gpt4Turbo-RulesJSONDoublePass": all_strategies["rulesJSONDoublePass"],
    "prompt-gpt4Turbo-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "prompt-gpt4Turbo-ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "prompt-gpt4Turbo-VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

analysis_exp = { 
    "symptomstudy-gpt4Turbo-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
}

def list_experiment_strategies(exp):
    return reduce(lambda k1, k2: k1 + ", " + k2, exp.keys())

if __name__ == "__main__":
    list_experiment_strategies(tuning_exp)
    note_processor.run_prompt_tuning(experiment=tuning_exp)
    # note_processor.run_analysis(experiment=analysis_exp, experiment_name="gpt4-analysis")
