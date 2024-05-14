from functools import reduce

from src.covid_study_strategies import build_strategies
from src.models import AzureGptModel
from src.processor import NoteProcessor

###############################################################################
#
# Build model and note processor 
#
noteConfig = {
    # "DIR_TUNING": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes-gpt-exp/tuning",
    "DIR_TUNING": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes-tuning",
    "DIR_OUTPUT_TUNING": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/output-gpt-tuning",
    # "DIR_INPUT": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes-gpt-exp/original",
    "DIR_INPUT": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes-original-study/i2b2",
    "DIR_OUTPUT": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/output-gpt",
}
note_processor = NoteProcessor(noteConfig, sleepRate=2)
model = AzureGptModel()

###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies(model)
tuning_exp = {
    "prompt-gpt35Turbo-Simple": all_strategies["simple"],
    "prompt-gpt35Turbo-Identity": all_strategies["identity"],
    "prompt-gpt35Turbo-Include": all_strategies["include"],
    "prompt-gpt35Turbo-Exclude": all_strategies["exclude"],
    "prompt-gpt35Turbo-Verbose": all_strategies["verbose"],
    # DoublePass
    "prompt-gpt35Turbo-SimpleDoublePass": all_strategies["simpleDoublePass"],
    "prompt-gpt35Turbo-IdentityDoublePass": all_strategies["identityDoublePass"],
    "prompt-gpt35Turbo-IncludeDoublePass": all_strategies["includeDoublePass"],
    "prompt-gpt35Turbo-ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "prompt-gpt35Turbo-VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "prompt-gpt35Turbo-SimpleJSON": all_strategies["simpleJSON"],
    "prompt-gpt35Turbo-IdentityJSON": all_strategies["identityJSON"],
    "prompt-gpt35Turbo-IncludeJSON": all_strategies["includeJSON"],
    "prompt-gpt35Turbo-ExcludeJSON": all_strategies["excludeJSON"],
    "prompt-gpt35Turbo-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "prompt-gpt35Turbo-SimpleJSONDoublePass": all_strategies["simpleJSONDoublePass"],
    "prompt-gpt35Turbo-IdentityJSONDoublePass": all_strategies["identityJSONDoublePass"],
    "prompt-gpt35Turbo-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "prompt-gpt35Turbo-ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "prompt-gpt35Turbo-VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

analysis_exp = { 
    "covidstudy-gpt35Turbo-SimpleJSON": all_strategies["simpleJSON"],
}


def list_experiment_strategies(exp):
    return reduce(lambda k1, k2: k1 + ", " + k2, exp.keys())


if __name__ == "__main__":
    list_experiment_strategies(analysis_exp)
    note_processor.run_prompt_tuning(experiment=tuning_exp)
    note_processor.run_analysis(experiment=analysis_exp)
