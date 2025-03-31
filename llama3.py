import os
from dotenv import load_dotenv

from src.symptom_study_strategies import build_strategies
from src.models import LLAMA3Model
from src.processor import NoteProcessor

load_dotenv(".env.llama3")

###############################################################################
#
# Build model and note processor
#
URL = os.getenv("TGI_URL")
model = LLAMA3Model(url=URL)
note_processor = NoteProcessor(model, "./note_config/open_llm.json", sleep=False)


###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies()
tuning_exp = {
    "prompt-llama3-Identity": all_strategies["identity"],
    "prompt-llama3-Rules": all_strategies["rules"],
    "prompt-llama3-Include": all_strategies["include"],
    "prompt-llama3-Exclude": all_strategies["exclude"],
    "prompt-llama3-Verbose": all_strategies["verbose"],
    # Simplification
    "prompt-llama3-IdentitySimplification": all_strategies["identitySimplification"],
    "prompt-llama3-RulesSimplification": all_strategies["rulesSimplification"],
    "prompt-llama3-IncludeSimplification": all_strategies["includeSimplification"],
    "prompt-llama3-ExcludeSimplification": all_strategies["excludeSimplification"],
    "prompt-llama3-VerboseSimplification": all_strategies["verboseSimplification"],
    # JSON
    "prompt-llama3-IdentityJSON": all_strategies["identityJSON"],
    "prompt-llama3-RulesJSON": all_strategies["rulesJSON"],
    "prompt-llama3-IncludeJSON": all_strategies["includeJSON"],
    "prompt-llama3-ExcludeJSON": all_strategies["excludeJSON"],
    "prompt-llama3-VerboseJSON": all_strategies["verboseJSON"],
    # JSON Validation
    "prompt-llama3-IdentityJSONValidation": all_strategies["identityJSONValidation"],
    "prompt-llama3-RulesJSONValidation": all_strategies["rulesJSONValidation"],
    "prompt-llama3-IncludeJSONValidation": all_strategies["includeJSONValidation"],
    "prompt-llama3-ExcludeJSONValidation": all_strategies["excludeJSONValidation"],
    "prompt-llama3-VerboseJSONValidation": all_strategies["verboseJSONValidation"],
}

# analysis_exp = {
#     "symptomstudy-llama3-IdentityJSON": all_strategies["identityJSON"],
# }


if __name__ == "__main__":
    note_processor.run_prompt_tuning(
        experiment=tuning_exp, experiment_name="llama3-tuning"
    )
    # note_processor.run_analysis(
    #     experiment=analysis_exp, experiment_name="llama3-analysis"
    # )
