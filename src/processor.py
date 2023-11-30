import os
import json
import random
import datetime
################################################################
# E2 directories

DIR_CURATED     = '/lab-share/CHIP-Mandl-e2/Public/autollm/ctakes-examples/ctakes_examples/resources/curated'
DIR_COVID       = '/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes'
DIR_SMALL_BATCH = '/lab-share/CHIP-Mandl-e2/Public/covid-llm/output-small-batches'
DIR_OUTPUT      = '/lab-share/CHIP-Mandl-e2/Public/covid-llm/output'
TARGET_NOTES    = '/lab-share/CHIP-Mandl-e2/Public/covid-llm/target-notes.json'

################################################################
# Helpers for creating folders and getting notes
# 
def get_target_notes(): 
    """
    Returns a list of target notes from a well-known spot on disc
    """
    with open(TARGET_NOTES, "r", encoding="utf8") as target_notes: 
        return json.load(target_notes)
def createDatedFolder(path: str): 
    """
    Creates a folder for the current day at a given path.
    Useful for running small-batch experiment over multiple days against the same notes.
    """
    mydir = os.path.join(path or os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d'))
    os.makedirs(mydir, exist_ok=True)
    return mydir

def get_curated_note(name: str) -> str:
    """
    This grabs a note from our corpus of curated (fake) notes (no PHI).
    Visit the following github directory to see the list of names you can use:
    https://github.com/Machine-Learning-for-Medical-Language/ctakes-examples/tree/main/ctakes_examples/resources/curated
    """
    path = f"{DIR_CURATED}/{name}"
    with open(path, "r", encoding="utf8") as f:
        return f.read().strip()
    
def get_covid_note(name: str = None) -> str:
    """
    Retrieves a given covid note or selects one randomly; 
    Returns the note and the note's name (helpful when one is selected randomly)
    """
    note_dir = DIR_COVID
    if not name: 
        # If no name is provided, select a note at random
        files = os.listdir(note_dir)
        files = [file for file in files if file.replace('.txt','') in get_target_notes()]
        file = files[random.randrange(len(files))]
        print("# Using file: ")
        print(file)
        with open(os.path.join(note_dir, file), "r", encoding="utf8") as f:
            return f.read().strip(), file
    path = f"{note_dir}/{name}"
    with open(path, "r", encoding="utf8") as f:
        return f.read().strip(), name


################################################################
# Note processing methods
# 
def process_dir_single_strategy(source_dir: str, strategy: str, output_ext: str) -> None:
    """
    Process all NOTES in $target_dir for a single strategy
    :param source_dir: dir input
    :param strategy: selected prompt strategy for LLM
    :param output_ext: extension to save file as; typically the string-name of the strategy
    """
    for fname in os.listdir(source_dir):
        note, _ = get_covid_note(fname)
        target = f'{DIR_OUTPUT}/{fname}.{output_ext}'

        print('######################################################')
        if not os.path.exists(target):
            print(f"{target} processing....")
            response = strategy.run(note) + '\n'
            with open(target, 'w') as fp:
                fp.write(response)
        else:
            print(f"{target} SKIP (file exists)")

def process_dir(experiment: dict, source_dir: str = DIR_COVID, note_list = None, skip_list = None) -> None:
    """
    Process all NOTES in $target_dir
    :param experiment: dictionary of prompting strategies to run  
    :param source_dir: dir input; DIR_COVID by default
    :param note_list: notes to process because we have labels for them; TARGET_NOTES by default
    :param skip_list: notes to skip because known to fail. So far only 1 is failing (largest ED note in BCH)
    """
    note_list = note_list or get_target_notes()
    for output_ext, strategy in experiment.items():
        print('########################')
        print('#' + output_ext + '\n')
        print(strategy.toJSON())

    with open(f'{DIR_OUTPUT}/experiment.json', 'w') as fp:
        serialized_experiment = {}
        for key, value in experiment.items(): 
            serialized_experiment[key] = value.toJSON()
        json.dump(serialized_experiment, fp)

    for fname in os.listdir(source_dir):
        note, _ = get_covid_note(fname)
        for output_ext, strategy in experiment.items():
            target = f'{DIR_OUTPUT}/{fname}.{output_ext}'

            print('######################################################')
            if not os.path.exists(target):
                # if we have a skip-list and its in it, ignore
                if skip_list and fname in skip_list:
                    print(f"{target} is believed to cause issues (SKIP)")
                # if we have a note-list, skip notes not in our list 
                elif note_list and (fname.replace('.txt', '') not in note_list):
                    print(f"{fname} is not in the list of notes to inspect (SKIP)")
                    continue
                else:
                    print(f"{target} processing....")
                    response = strategy.run(note) + '\n'
                    with open(target, 'w') as fp:
                        fp.write(response)
            else:
                print(f"{target} SKIP (file exists)")

def process_small_batch(experiment: dict, dir: str = DIR_SMALL_BATCH, n: int = 10):
    """
    Process some randomly selected $n covid-notes,
    :param experiment: dictionary of prompting strategies to run  
    :param dir: output dir for responses; DIR_SMALL_BATCH by default
    :param n: number of notes to process; 10 by default 
    """
    # Create unique dir for this run 
    new_dir = createDatedFolder(dir)
    for _ in range(n): 
        # select a random note
        note, name = get_covid_note()
        for strategy_name, strategy in experiment.items():
            print("###################################")
            print(f"# strategy_name: '{strategy_name}'")
            print("###################################")
            response = strategy.run(note) + "\n"
            # Record result
            with open(os.path.join(new_dir, strategy_name + "-" + name), "w+", encoding="utf8") as fp: 
                fp.write(response)
