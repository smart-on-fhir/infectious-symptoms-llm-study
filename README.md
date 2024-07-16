# Detecting Infectious Respiratory Disease Symptoms using LLMs

Code used in CHIP's study on detecting infectious respiratory disease symptoms using large language models (LLMs). Code includes the experiments used in our study, as well as boilerplate classes/helper methods for running additional prompt experiments against LLAMA2/Mixtral/GPT models served via HuggingFace's Text Generation Inference (TGI) API or Azure's Open AI API to identify successful prompting strategies.

## Getting Started

To generate some experimental output using an Azure-hosted GPT4 instance as an example:

1. Create an `.env` file based on our `.env.template` file. For self-hosted TGI models, update TGI_URL to point at your TGI server. For Azure hosted models, update the endpoint URL, API key, API version, and deployment to match the GPT model you're prompting. 
  - You should create an .env file specific to each experiment/model you want to run. For example, the `gpt4.py` script looks for a `.env.gpt4` env file. Feel free to change this as needed, but this is the default behavior.
2. Using at least python v3.9.16, ensure that packages in `requirements.txt` are installed 
  - Consider creating a venv with `python -m venv venv`, activating that venv with `source venv/bin/activate` and installing dependencies with `python -m pip install -r requirements.txt`.
3. Once dependencies are installed, define a note-config file that you will use for your experiment, building off of `note_config/_example.json`. This will define where the clinical notes for tuning and your final analysis live, where the output for both of these steps should be written, and if only a subset of notes should be examined in the tuning phase. 
  - Note: You can define a `default.json` note_config to set some default values for all 5 required fields. Other note_configs can then build off that, changing only what they need.
4. Run an experiment! We have scripts for each of the experiments we conducted in the project's home directory. If you had an instance of gpt4 running and wanted to replicate our results try: 
```shell
python gpt4.py
```


## Terminology

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