import os
import requests
import json
import random
################################################################
# E2 directories

DIR_MTSMAPLES = '/lab-share/CHIP-Mandl-e2/Public/autollm/ctakes-examples/ctakes_examples/resources/curated'
DIR_COVID = '/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes'
DIR_OUTPUT = '/lab-share/CHIP-Mandl-e2/Public/covid-llm/output'

def get_mtsample(name: str) -> str:
    path = f"{DIR_MTSMAPLES}/{name}"
    with open(path, "r", encoding="utf8") as f:
        return f.read().strip()


def get_covid_note(name: str) -> str:
    path = f"{DIR_COVID}/{name}"
    with open(path, "r", encoding="utf8") as f:
        return f.read().strip()


################################################################

# This is Meta's official prompt they use for their public Chat tool
# https://huggingface.co/blog/llama2#how-to-prompt-llama-2
DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant.

Always answer as helpfully as possible, while being safe.

Your answers should not include any harmful, unethical,
racist, sexist, toxic, dangerous, or illegal content.

Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent,
explain why instead of answering something not correct.

If you don't know the answer to a question, please don't share false information.
"""

# This is the formatting that Llama2's chat model is trained on.
# https://huggingface.co/blog/llama2#how-to-prompt-llama-2
PROMPT_FORMAT = """<s>[INST] <<SYS>>
%(system)s
<</SYS>>

%(user)s [/INST]"""


# Convenience method to prompt llama2
def prompt(note: str, system: str = None) -> str:
    system = system or DEFAULT_SYSTEM_PROMPT
    full_prompt = PROMPT_FORMAT % {"system": system.strip(), "user": note.strip()}
    response = requests.post("http://localhost:8086/", json={
        "inputs": full_prompt,
        "options": {
            "wait_for_model": True,
        },
        "parameters": {
            "max_new_tokens": 1000,
        },
    })
    response.raise_for_status()

    answer = response.json()[0]["generated_text"]

    # The answer includes the prompt, to make it easier to feed previous
    # history back to llama2 so it learns from a conversation. But we are
    # designing here for a single request, not a conversation.
    if answer.startswith(full_prompt):
        answer = answer[len(full_prompt):].strip()
    return answer


def process_dir(source_dir: str, choice_prompt: str, output_ext) -> None:
    """
    Process all NOTES in $target_dir
    :param source_dir: dir input
    :param choice_prompt: selected prompt for LLM
    :param output_ext: extension to save file as
    """
    for fname in os.listdir(source_dir):
        note = get_covid_note(fname)
        target = f'{DIR_OUTPUT}/{fname}.{output_ext}'

        print('######################################################')
        if not os.path.exists(target):
            print(f"{target} processing....")
            response = prompt(note, choice_prompt)
            with open(target, 'w') as fp:
                fp.write(response)
        else:
            print(f"{target} SKIP (file exists)")

def process_dir_multi_prompts(source_dir: str, multiple_prompts: dict, skip_list=None) -> None:
    """
    Process all NOTES in $target_dir
    :param source_dir: dir input
    :param choice_prompt: selected prompt for LLM
    :param output_ext: extension to save file as
    :param skip_list: notes to skip because known to fail. So far only 1 is failing (largest ED note in BCH)
    """
    for output_ext, choice_prompt in multiple_prompts.items():
        print('########################')
        print('#' + output_ext + '\n')
        print(choice_prompt)

    with open(f'{DIR_OUTPUT}/multiple_prompts.json', 'w') as fp:
        json.dump(multiple_prompts, fp)

    for fname in os.listdir(source_dir):
        note = get_covid_note(fname)
        for output_ext, choice_prompt in multiple_prompts.items():
            target = f'{DIR_OUTPUT}/{fname}.{output_ext}'

            print('######################################################')
            if not os.path.exists(target):

                if skip_list and fname in skip_list:
                    print(f"{target} is believed to cause issues (SKIP)")
                else:
                    print(f"{target} processing....")
                    response = prompt(note, choice_prompt) + '\n'
                    with open(target, 'w') as fp:
                        fp.write(response)
            else:
                print(f"{target} SKIP (file exists)")
