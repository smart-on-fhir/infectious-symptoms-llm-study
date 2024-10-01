# Detecting Infectious Respiratory Disease Symptoms using LLMs

Results and code used in CHIP's study on detecting infectious respiratory disease symptoms using large language models (LLMs). Code includes the experiments used in our study, as well as boilerplate classes/helper methods for running additional prompt experiments against LLAMA2/Mixtral/GPT models served via HuggingFace's Text Generation Inference (TGI) API or Azure's Open AI API to identify successful prompting strategies.

## Results: Prompt, Test, and Validation Results

In this repo's `results` folder you can find three sub-directories containing the tabular results of our experiments: 
- `results/prompt`: Results of experiments identifying, for each LLM model, the optimal prompting scenario combining prompting instructions and output parsing pipelines. Four models are explored, and each model explores 20 different prompting scenarios. Models' symptom output are evaluated against a prompt-specific dataset. Files are named using the following pattern: 
  - `accuracy-prompt-{model}-{promptingScenario}`
- `results/test`: Results of experiments identifying the optimal model for this respiratory disease symptom identification task, comparing the performance of model's using each model's optimal prompting scenario. Four models are explored. Models' symptom output are evaluated against a test-specific dataset and are compared against two annotators. Files are named using the following pattern: 
  - `accuracy-{annotator}-test-{model}-{optimalScenarioForThisModel}`
- `results/validation`: Results of experiments validating the performance of our optimal model against a novel data provider for this respiratory disease symptom identification task. Files are named using the following pattern: 
  - `TBD`

## Replicating: Generating LLM Symptom Responses

To generate some experimental output using an Azure-hosted GPT4 instance as an example:

1. Create an `.env` file based on our `.env.template` file. For self-hosted TGI models, update TGI_URL to point at your TGI server. For Azure hosted models, update the endpoint URL, API key, API version, and deployment to match the GPT model you're prompting. 
  - You should create an .env file specific to each experiment/model you want to run. For example, the `gpt4.py` script looks for a `.env.gpt4` env file.
2. Using at least python v3.10, ensure that packages in `requirements.txt` are installed 
  - Consider creating a venv with `python -m venv venv`, activating that venv with `source venv/bin/activate` and installing dependencies with `python -m pip install -r requirements.txt`.
3. Once dependencies are installed, define a note-config file that you will use for your experiment, building off of `note_config/_example.json`. This defines where the model input (read: clinical notes for prompt engineering and model testing) live, where the outputs should live, and if only a subset of notes should be examined in the tuning phase. 
  - **Notes must adhere to a simple naming convention `<NOTE_ID>.txt`**. This is important for downstream tasks, which rely on `NOTE_ID` to be parsed from the file names.
  - Note: You can define a `default.json` note_config to set some default values for all 5 required fields. Other note_configs can then build off that, changing only what they need.
4. Run an experiment! We have scripts for each of the experiments we conducted in the project's home directory. If you had an instance of gpt4 running and wanted to replicate our results try: 
```shell
python gpt4.py
```

This will run a single experiment using a single prompting strategy – for GPT4, that's the optimal-identified Include instruction and the JSON pipeline. For each `<NOTE_ID>.txt` file, a corresponding `<NOTE_ID>.txt.<STRATEGY>`file will be generated in your note-config specified output directory, where `NOTE_ID` is the note's ID and `STRATEGY` corresponds to the prompting strategy's name. If you're following along exactly, you should see several files named `<NOTE_ID>.txt.symptomstudy-gpt4Turbo-IncludeJSON`.

## Replicating: Getting Chart Review Results from LLM Symptom Responses

After generating symptom predictions, you can translate those files into a format that the [SMART on FHIR Chart Review](https://docs.smarthealthit.org/cumulus/chart-review/) module can consume, enabling comparisons against ground-truth labelled symptoms data exported from LabelStudio. Note that this feature expects you to upload notes to Label Studio using Cumulus ETL's `upload-notes` command. That way the document IDs get stored correctly as Label Studio metadata. 

Producing Chart Review results is as simple as: 
1. Export your annotated notes from Label Studio as a `./data/labelstudio-export.json` file.
  - If you haven't uploaded your notes to Label Studio using the Cumulus ETL's `upload-notes` command, see our [Troubleshooting section](#troubleshooting) for advice on linking the export file's tasks to the LLM-generated output.
2. Generate a Chart Review-consumable external label file using the `jsonToLabels.py` script. 
  - Assuming your LLM responses lived in `./data/responses` and you wanted your exported label file to live in `./data`, you would run the following: `python jsonToLabels.py -e symptomstudy-gpt4Turbo-IncludeJSON -s data/responses -d data`
3. Create a `./data/config.yaml` file for Chart Review: Follow the instructions found [here](https://docs.smarthealthit.org/cumulus/chart-review/config.html). Make sure that your ground truth annotator ID matches the ID used in your labelstudio export file. To reference the above external label file as an annotator, add the following: 
```
  symptomstudy-gpt4Turbo-IncludeJSON:
    filename: ./symptomstudy-gpt4Turbo-IncludeJSON.csv
```
  Make sure the relative path above is accurate relative to where this config lives.
4. Run Chart Review: From within the directory containing your `config.yaml` and `labelstudio-export.json`, run the following with GROUND_TRUTH_ANNOTATOR replaced with whatever you specified in `config.yaml` file:
```
chart-review accuracy --save <GROUND_TRUTH_ANNOTATOR> symptomstudy-gpt4Turbo-IncludeJSON
```


## Replicating: Troubleshooting

> What if I didn't use the Cumulus ETL's `upload-notes` command when setting up my Label Studio environment with clinical notes? 

This step is necessary because it connects the ID's used in Label Studio to ID's we're expecting to be included in the clinical note files.A manual workaround for this is to add (either with a script or manually) a `data` property to every Label Studio exported task object. This object should have a property docref_mappings that connects the Label Studio ids to the note ids, like so: 
```json
    // ... Label Studio task data
    "data": {
      "docref_mappings": {
        "<LABEL_STUDIO_ID>": "<NOTE_ID>",
      }
    }
    // ... more Label Studio task data
```

> What if I want to fully replicate your results? 

We encourage you to! That said, our pipeline for processing text data was more convoluted. We are happy to provide example code on an as-requested basis, but to get you going we can sketch out our high level process: 
1. Generate responses from LLMs using the commented out "prompt engineering" experiments in our scripts 
2. Set up an instance of the [Cumulus covid symptom study](https://docs.smarthealthit.org/cumulus/etl/studies/covid-symptom.html) for processing the notes into document references with the CTAKEs identified symptoms.
3. Create a script that translates the CTAKEs document references into a label format akin to the `jsonToLabels.py` script. 
4. Use this label file in your chart-review analysis, as done above. 

# Replicating: Project Structure and Terminology

Our goal of experimenting with an LLM is to create an effective way of interacting with a model for
a given task – we can think of this as creating an effective prompting Strategy. To make this process
easier in the future, we define certain classes and methods that plug and play together to make for a
flexible, Strategy-experimentation playground. In our example experiment – see `main.py` – you'll see how we use the following:

- `strategy.py`: Defines Strategy, the logic of a given prompting-strategy. Each strategy is made of one or
  many different Steps. The result of a Strategy is always the output of the last Step in its sequence.
  The input to a strategy in the context of our experiments is always the clinical content the LLM is reasoning over,
  but how Steps use this input may vary.

- `step.py`: Defines Step, the atomic unit of a Strategy. Steps are run by a Strategy, and each Step corresponds to
  a call to a ModelInterface. There are three different types of Steps, each presenting different text
  as context for the LLM to reason over: `default` which reasons over the original clinical context
  (e.g. a note) provided by the Strategy, `previous` which reason over the output of the previous Step
  (e.g. to ask the LLM to simplify the result of a previous query), and `aggregator` which reason over
  the collated-output of all previous Steps (e.g. to ask many targeted, individual questions about a
  clinical note and then to simplify the combination of all those responses into a single result).
  Caution: There are "ungrammatical" ways of combining steps and currently there is no validation to
  prevent nonsensical combinations; it is the responsibility of a Strategy author to ensure that the
  combination of Steps makes sense.

- `instructions.py`: All of the core instructions to LLMs that we've tested so far. Think of this as a
  library of instruction to pull from within your Steps and Strategies.

- `symptom_study_strategies.py`: We combine the instructions in `instructions.py` and various different Steps to create 20 unique Strategies we explore in our study.

- `processor.py`: A NoteProcessor class, which iterates over a set of notes and runs a given experiment against a given mode. The two methods of interest are `note_processor.run_prompt_tuning` and `note_processor.run_analysis`, for running prompt_tuning and analysis experiments respectively.

- `models/*`: A home for all model wrappers and API clients used in interacting with hosted LLMs. Models currently supported include
  - Several TGI Models: Mixtral8x22B, Mixtral8x7B (Mixtral), Llama2
  - Azure-hosted Models: So far this has been tested with gpt3.5-turbo and gpt4-turbo