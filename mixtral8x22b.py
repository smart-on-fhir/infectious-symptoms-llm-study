import os
from functools import reduce
from dotenv import load_dotenv

from src.symptom_study_strategies import build_strategies
from src.models import MixtralModel
from src.processor import NoteProcessor
load_dotenv()

###############################################################################
#
# Build model and note processor 
#
URL = os.getenv("TGI_URL")
model = MixtralModel(url=URL)
note_processor = NoteProcessor(model, './note_config/open_llm.json', sleep=False)

###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies()
tuning_exp = {
    "prompt-mixtral8x22b-Simple": all_strategies["simple"],
    "prompt-mixtral8x22b-Identity": all_strategies["identity"],
    "prompt-mixtral8x22b-Include": all_strategies["include"],
    "prompt-mixtral8x22b-Exclude": all_strategies["exclude"],
    "prompt-mixtral8x22b-Verbose": all_strategies["verbose"],
    # DoublePass
    "prompt-mixtral8x22b-SimpleDoublePass": all_strategies["simpleDoublePass"],
    "prompt-mixtral8x22b-IdentityDoublePass": all_strategies["identityDoublePass"],
    "prompt-mixtral8x22b-IncludeDoublePass": all_strategies["includeDoublePass"],
    "prompt-mixtral8x22b-ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "prompt-mixtral8x22b-VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "prompt-mixtral8x22b-SimpleJSON": all_strategies["simpleJSON"],
    "prompt-mixtral8x22b-IdentityJSON": all_strategies["identityJSON"],
    "prompt-mixtral8x22b-IncludeJSON": all_strategies["includeJSON"],
    "prompt-mixtral8x22b-ExcludeJSON": all_strategies["excludeJSON"],
    "prompt-mixtral8x22b-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "prompt-mixtral8x22b-SimpleJSONDoublePass": all_strategies["simpleJSONDoublePass"],
    "prompt-mixtral8x22b-IdentityJSONDoublePass": all_strategies["identityJSONDoublePass"],
    "prompt-mixtral8x22b-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "prompt-mixtral8x22b-ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "prompt-mixtral8x22b-VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

# analysis_exp = { 
#     "symptomstudy-mixtral8x22b-SimpleJSON": all_strategies["simpleJSON"],
# }


def list_experiment_strategies(exp):
    return reduce(lambda k1, k2: k1 + ", " + k2, exp.keys())


if __name__ == "__main__":
    # list_experiment_strategies(analysis_exp)
    list_experiment_strategies(tuning_exp)
    note_processor.run_prompt_tuning(experiment=tuning_exp, experiment_name="mixtral8x22b-tuning")
    # note_processor.run_analysis(experiment=analysis_exp, experiment_name="mixtral8x22b-analysis")
