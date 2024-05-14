from functools import reduce

from src.covid_study_strategies import build_strategies
from src.models import AzureGptModel
from src.processor import NoteProcessor

###############################################################################
#
# Build model and note processor 
#
# No tuning needed
noteConfig = {
    # "DIR_TUNING": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes-gpt-exp/tuning",
    "DIR_TUNING": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes-tuning",
    "DIR_OUTPUT_TUNING": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/output-gpt4Turbo-tuning",
    # "DIR_INPUT": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes-gpt-exp/original",
    "DIR_INPUT": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes-original-study/i2b2",
    "DIR_OUTPUT": "/lab-share/CHIP-Mandl-e2/Public/covid-llm/output-gpt",
}

note_processor = NoteProcessor(noteConfig, sleepRate=5)

# Use the gpt4Turbo deployment - named very simply
model = AzureGptModel(model="gpt-4")

###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies(model)

tuning_exp = {
    "prompt-gpt4Turbo-Simple": all_strategies["simple"],
    "prompt-gpt4Turbo-Identity": all_strategies["identity"],
    "prompt-gpt4Turbo-Include": all_strategies["include"],
    "prompt-gpt4Turbo-Exclude": all_strategies["exclude"],
    "prompt-gpt4Turbo-Verbose": all_strategies["verbose"],
    # DoublePass
    "prompt-gpt4Turbo-SimpleDoublePass": all_strategies["simpleDoublePass"],
    "prompt-gpt4Turbo-IdentityDoublePass": all_strategies["identityDoublePass"],
    "prompt-gpt4Turbo-IncludeDoublePass": all_strategies["includeDoublePass"],
    "prompt-gpt4Turbo-ExcludeDoublePass": all_strategies["excludeDoublePass"],
    "prompt-gpt4Turbo-VerboseDoublePass": all_strategies["verboseDoublePass"],
    # JSON
    "prompt-gpt4Turbo-SimpleJSON": all_strategies["simpleJSON"],
    "prompt-gpt4Turbo-IdentityJSON": all_strategies["identityJSON"],
    "prompt-gpt4Turbo-IncludeJSON": all_strategies["includeJSON"],
    "prompt-gpt4Turbo-ExcludeJSON": all_strategies["excludeJSON"],
    "prompt-gpt4Turbo-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Double Pass
    "prompt-gpt4Turbo-SimpleJSONDoublePass": all_strategies["simpleJSONDoublePass"],
    "prompt-gpt4Turbo-IdentityJSONDoublePass": all_strategies["identityJSONDoublePass"],
    "prompt-gpt4Turbo-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
    "prompt-gpt4Turbo-ExcludeJSONDoublePass": all_strategies["excludeJSONDoublePass"],
    "prompt-gpt4Turbo-VerboseJSONDoublePass": all_strategies["verboseJSONDoublePass"],
}

analysis_exp = { 
    "covidstudy-gpt4Turbo-IncludeJSONDoublePass": all_strategies["includeJSONDoublePass"],
}

if __name__ == "__main__":
    # print(model.get_model_info())
    # print(model.client)
    print('starting gpt4Turbo experiments')
    # note_processor.run_prompt_tuning(experiment=tuning_exp)
    note_processor.run_analysis(experiment=analysis_exp, experiment_name="gpt4-analysis-includeJSONDoublePass")
