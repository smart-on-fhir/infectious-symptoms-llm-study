import os
from functools import reduce
from dotenv import load_dotenv

from src.covid_study_strategies import build_strategies
from src.models import MixtralModel
from src.processor import NoteProcessor
load_dotenv()

###############################################################################
#
# Build model and note processor 
#
noteConfig = {
    "DIR_TUNING": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes-tuning",
    "DIR_OUTPUT_TUNING": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/output-prompt-tuning-timing",
    "DIR_INPUT": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes-original-study/i2b2",
    "DIR_OUTPUT": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/output-open-llms",
}
note_processor = NoteProcessor(noteConfig, sleep=False)
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
model = MixtralModel(url=f'{HOST}:{PORT}')

###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies(model)
tuning_exp = {
    "prompt-mixtral-Simple": all_strategies["simple"],
    "prompt-mixtral-Identity": all_strategies["identity"],
    "prompt-mixtral-Include": all_strategies["include"],
    "prompt-mixtral-Exclude": all_strategies["exclude"],
    "prompt-mixtral-Verbose": all_strategies["verbose"],
    # DoublePass
    "prompt-mixtral-SimpleDoublePass": all_strategies["simpleDoublePass"],
    "prompt-mixtral-IdentityDoublePass": all_strategies["identityDoublePass"],
    "prompt-mixtral-IncludeDoublePass": all_strategies["includeDoublePass"],
    "prompt-mixtral-ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "prompt-mixtral-VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "prompt-mixtral-SimpleJSON": all_strategies["simpleJSON"],
    "prompt-mixtral-IdentityJSON": all_strategies["identityJSON"],
    "prompt-mixtral-IncludeJSON": all_strategies["includeJSON"],
    "prompt-mixtral-ExcludeJSON": all_strategies["excludeJSON"],
    "prompt-mixtral-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "prompt-mixtral-SimpleJSONDoublePass": all_strategies["simpleJSONDoublePass"],
    "prompt-mixtral-IdentityJSONDoublePass": all_strategies["identityJSONDoublePass"],
    "prompt-mixtral-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "prompt-mixtral-ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "prompt-mixtral-VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

analysis_exp = { 
    "covidstudy-mixtral-SimpleJSON": all_strategies["simpleJSON"],
}


def list_experiment_strategies(exp):
    return reduce(lambda k1, k2: k1 + ", " + k2, exp.keys())


if __name__ == "__main__":
    list_experiment_strategies(analysis_exp)
    note_processor.run_prompt_tuning(experiment=tuning_exp, experiment_name="mixtral-tuning")
    note_processor.run_analysis(experiment=analysis_exp, experiment_name="mixtral-analysis")
