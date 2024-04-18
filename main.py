from functools import reduce

from src.covid_study_strategies import build_strategies
from src.model_interface import LLAMA2Interface

## Custom methods
from src.processor import NoteProcessor

###############################################################################
#
# Use LLAMA2 as our interface
#
# Make sure you check the info printed later to be sure you are on the right port
HOST = "http://localhost"
PORT = "8086"
URL = f"{HOST}:{PORT}/"

noteConfig = {
    "DIR_OUTPUT_TUNING": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/output-prompt-tuning",
    "DIR_OUTPUT": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/output-pipeline-analysis-i2b2",
    "DIR_INPUT": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes-original-study/i2b2",
}
note_processor = NoteProcessor(noteConfig)
model = LLAMA2Interface(URL)


def list_experiment_strategies(exp):
    return reduce(lambda k1, k2: k1 + ", " + k2, exp.keys())


###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies(model)
exp = {
    "prompt-llama2Simple": all_strategies["simple"],
    "prompt-llama2Identity": all_strategies["identity"],
    "prompt-llama2Include": all_strategies["include"],
    "prompt-llama2Exclude": all_strategies["exclude"],
    "prompt-llama2Verbose": all_strategies["verbose"],
    # DoublePass
    "prompt-llama2SimpleDoublePass": all_strategies["simpleDoublePass"],
    "prompt-llama2IdentityDoublePass": all_strategies["identityDoublePass"],
    "prompt-llama2IncludeDoublePass": all_strategies["includeDoublePass"],
    "prompt-llama2ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "prompt-llama2VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "prompt-llama2SimpleJSON": all_strategies["simpleJSON"],
    "prompt-llama2IdentityJSON": all_strategies["identityJSON"],
    "prompt-llama2IncludeJSON": all_strategies["includeJSON"],
    "prompt-llama2ExcludeJSON": all_strategies["excludeJSON"],
    "prompt-llama2VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "prompt-llama2SimpleJSONDoublePass": all_strategies["simpleJSONDoublePass"],
    "prompt-llama2IdentityJSONDoublePass": all_strategies["identityJSONDoublePass"],
    "prompt-llama2IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "prompt-llama2ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "prompt-llama2VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

list_experiment_strategies(exp)

if __name__ == "__main__":
    for k, v in exp.items():
        print("----------------------------")
        print(k)
        for s in v.steps:
            print("# step")
            print(s.instruction)
