import os
from functools import reduce
from dotenv import load_dotenv

from src.covid_study_strategies import build_strategies
from src.models import LLAMA2Model
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
model = LLAMA2Model(url=f'{HOST}:{PORT}')

###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies(model)
tuning_exp = {
    "prompt-llama2-Simple": all_strategies["simple"],
    "prompt-llama2-Identity": all_strategies["identity"],
    "prompt-llama2-Include": all_strategies["include"],
    "prompt-llama2-Exclude": all_strategies["exclude"],
    "prompt-llama2-Verbose": all_strategies["verbose"],
    # DoublePass
    "prompt-llama2-SimpleDoublePass": all_strategies["simpleDoublePass"],
    "prompt-llama2-IdentityDoublePass": all_strategies["identityDoublePass"],
    "prompt-llama2-IncludeDoublePass": all_strategies["includeDoublePass"],
    "prompt-llama2-ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "prompt-llama2-VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "prompt-llama2-SimpleJSON": all_strategies["simpleJSON"],
    "prompt-llama2-IdentityJSON": all_strategies["identityJSON"],
    "prompt-llama2-IncludeJSON": all_strategies["includeJSON"],
    "prompt-llama2-ExcludeJSON": all_strategies["excludeJSON"],
    "prompt-llama2-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "prompt-llama2-SimpleJSONDoublePass": all_strategies["simpleJSONDoublePass"],
    "prompt-llama2-IdentityJSONDoublePass": all_strategies["identityJSONDoublePass"],
    "prompt-llama2-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "prompt-llama2-ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "prompt-llama2-VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

analysis_exp = { 
    "covidstudy-llama2-SimpleJSON": all_strategies["simpleJSON"],
}


def list_experiment_strategies(exp):
    return reduce(lambda k1, k2: k1 + ", " + k2, exp.keys())


if __name__ == "__main__":
    list_experiment_strategies(analysis_exp)
    note_processor.run_prompt_tuning(experiment=tuning_exp, experiment_name="llama2-tuning")
    note_processor.run_analysis(experiment=analysis_exp, experiment_name="llama2-analysis")
