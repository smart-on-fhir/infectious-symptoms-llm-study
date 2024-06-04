import os
from functools import reduce
from dotenv import load_dotenv

from src.symptom_study_strategies import build_strategies
from src.models import LLAMA3Model
from src.processor import NoteProcessor
load_dotenv()

###############################################################################
#
# Build model and note processor 
#
URL = os.getenv("TGI_URL")
model = LLAMA3Model(url=URL)


note_processor = NoteProcessor(model, './note_config/llama3.json', sleep=False)


###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies()
tuning_exp = {
    "prompt-llama3-Simple": all_strategies["simple"],
    "prompt-llama3-Identity": all_strategies["identity"],
    "prompt-llama3-Include": all_strategies["include"],
    "prompt-llama3-Exclude": all_strategies["exclude"],
    "prompt-llama3-Verbose": all_strategies["verbose"],
    # DoublePass
    "prompt-llama3-SimpleDoublePass": all_strategies["simpleDoublePass"],
    "prompt-llama3-IdentityDoublePass": all_strategies["identityDoublePass"],
    "prompt-llama3-IncludeDoublePass": all_strategies["includeDoublePass"],
    "prompt-llama3-ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "prompt-llama3-VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "prompt-llama3-SimpleJSON": all_strategies["simpleJSON"],
    "prompt-llama3-IdentityJSON": all_strategies["identityJSON"],
    "prompt-llama3-IncludeJSON": all_strategies["includeJSON"],
    "prompt-llama3-ExcludeJSON": all_strategies["excludeJSON"],
    "prompt-llama3-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "prompt-llama3-SimpleJSONDoublePass": all_strategies["simpleJSONDoublePass"],
    "prompt-llama3-IdentityJSONDoublePass": all_strategies["identityJSONDoublePass"],
    "prompt-llama3-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "prompt-llama3-ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "prompt-llama3-VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

analysis_exp = { 
    "symptomstudy-llama3-SimpleJSON": all_strategies["simpleJSON"],
}


def list_experiment_strategies(exp):
    return reduce(lambda k1, k2: k1 + ", " + k2, exp.keys())


if __name__ == "__main__":
    list_experiment_strategies(analysis_exp)
    note_processor.run_prompt_tuning(experiment=tuning_exp, experiment_name="llama3-tuning")
    note_processor.run_analysis(experiment=analysis_exp, experiment_name="llama3-analysis")
