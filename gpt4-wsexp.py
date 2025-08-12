import os
from dotenv import load_dotenv

from src.symptom_study_strategies import build_strategies
from src.models import AzureGptModel
from src.processor import NoteProcessor

load_dotenv(".env.gpt4_ws")

###############################################################################
#
# Build model and note processor
#
url = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
model_type = os.getenv("AZURE_OPENAI_DEPLOYMENT")
model_ws = AzureGptModel(
    url=url, 
    api_key=api_key, 
    api_version=api_version, 
    model_type=model_type,
    REMOVE_WS_FLAG=False
)
model_no_ws = AzureGptModel(
    url=url, 
    api_key=api_key, 
    api_version=api_version, 
    model_type=model_type,
    REMOVE_WS_FLAG=True
)
note_processor = NoteProcessor(model_ws, "./note_config/gpt_ws_test.json", sleepRate=2)
note_processor_no_ws = NoteProcessor(model_no_ws, "./note_config/gpt_ws_test.json", sleepRate=2)

###############################################################################
#
# Building experiment with strategies
#
all_strategies = build_strategies()

test_exp_ws = {
    "test-gpt4o-IncludeJSON-ws": all_strategies["includeJSON"],
}
test_exp_no_ws = {
    "test-gpt4o-IncludeJSON-NO-ws": all_strategies["includeJSON"],
}

if __name__ == "__main__":
    note_processor.run_analysis(
        experiment=test_exp_ws, experiment_name="gpt4-ws-analysis", limit=200
    )
    note_processor_no_ws.run_analysis(
        experiment=test_exp_no_ws, experiment_name="gpt4-no-ws-analysis", limit=200
    )
