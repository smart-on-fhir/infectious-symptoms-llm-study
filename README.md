# llm-covid-prompts

Boilerplate classes and helper methods for running prompt experiments against LLAMA2 models served via 
HuggingFace's Text Generation Inference (TGI) API to identify successful prompting strategy.

Our goal of experimenting with an LLM is to create an effective way of interacting with a model for 
a given task – we can think of this as creating an effective prompting Strategy. To make this process 
easier in the future, we define certain classes and methods that plug and play together to make for a 
flexible, Strategy-experimentation playground. In our example experiment – see `main.py` – you'll see how we use the following:

- `LLAMA2Interface`: An interface for making LLAMA2 inferences given a prompt_format, an instruction, 
  some context over which the LLM should reason, and an optional system message (this is unused by 
  LLAMA2's default prompt, but it is common in other prompts and potentially useful)

- `Strategy`: The logic of a given prompting-strategy. Each strategy is made up a list of one or 
  many different Steps. The result of a Strategy is always the output of the last Step in its sequence. 
  The input to a strategy is always the clinical content the LLM is reasoning over, but how Steps use this input may vary.

- `Step`: The atomic unit of a Strategy. Steps are run by a Strategy, and each Step corresponds to 
  a call to a ModelInterface. There are three different types of Steps, each presenting different text 
  as context for the LLM to reason over: `default` which reasons over the original clinical context 
  (e.g. a note) provided by the Strategy, `previous` which reason over the output of the previous Step 
  (e.g. to ask the LLM to simplify the result of a previous query), and `aggregator` which reason over 
  the collated-output of all previous Steps (e.g. to ask many targeted, individual questions about a 
  clinical note and then to simplify the combination of all those responses into a single result). 
  Caution: There are "ungrammatical" ways of combining steps and currently there is no validation to 
  prevent nonsensical combinations; it is the responsibility of a Strategy author to ensure that the 
  combination of Steps makes sense.

- `instructions.py`: All of the instructions to LLMs that we've tested so far. Think of this as a 
  library of instruction to pull from within your Steps and Strategies.

- `processor.py`: Techniques for processing an experiment – i.e. dictionary of strategies – over a 
  large quantity of notes. Two common methods you'll use are `process_small_batch`, for running an 
  experiment on a small batch of notes that you can review outside of the typical output-to-compare 
  pipeline, and `process_dir`, for running an experiment on an entire directory of notes and generating 
  results to process in the E2 output directory.
