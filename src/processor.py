import os
import json
import random
import datetime
################################################################
# E2 directories

DEFAULT_CONFIG = {
    'DIR_TUNING'          : '/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes-tuning',
    'DIR_OUTPUT_TUNING'   : '/lab-share/CHIP-Mandl-e2/Public/covid-llm/output-paper',
    'DIR_INPUT'           : '/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes-cerner',
    'DIR_OUTPUT'          : '/lab-share/CHIP-Mandl-e2/Public/covid-llm/output-paper',
    'PROMPT_TUNING_NOTES' : '/lab-share/CHIP-Mandl-e2/Public/covid-llm/prompt-tuning-notes.json',
}

class NoteProcessor(): 
    def __init__(self, note_config = None):
        self.note_config = DEFAULT_CONFIG if not note_config else {**DEFAULT_CONFIG, **note_config}

    def _get_prompt_tuning_note_ids(self): 
        """
        :returns: list of notes ids used for for prompt-tuning 
        """
        with open(self.note_config["PROMPT_TUNING_NOTES"], "r", encoding="utf8") as target_notes: 
            return json.load(target_notes)

    def _create_dated_folder(self, path: str): 
        """
        Creates a folder for the current day at a given path.
        Useful for running small-batch experiment over multiple days against the same notes.
        :param path: where to create the new directory
        :returns: path to the newly created, dated directory
        """
        mydir = os.path.join(path or os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(mydir, exist_ok=True)
        return mydir
    
    def _get_note(self, note_dir: str, name: str = None, only_these_notes = None):
        """
        Retrieves a given covid note based on name, or selects one randomly;
        will limit selection to only_these_notes if defined,
        :param note_dir: where the notes are found
        :type note_dir: str
        :param name: The file to be retrieved
        :type name: str, optional
        :param: A subset of notes to look at
        :returns: the note and the note's name (helpful when one is selected randomly)
        """
        # If no name is provided, select a note at random
        if not name: 
            files = os.listdir(note_dir)
            # If there is a list of target notes by id, limit our selection to that list
            if only_these_notes: 
                files = [file for file in files if file.replace('.txt','') in only_these_notes]
            file = files[random.randrange(len(files))]
            with open(os.path.join(note_dir, file), "r", encoding="utf8") as f:
                return f.read().strip(), file
        path = f"{note_dir}/{name}"
        with open(path, "r", encoding="utf8") as f:
            return f.read().strip(), name

    def get_sample_input_note(self, name: str = None):
        """
        Retrieves a given covid note or selects one randomly from the input directory
        :param name: The note to get
        :type name: str, optional
        :returns: the note and the note's name (helpful when one is selected randomly)
        """
        return self._get_note(self.note_config["DIR_INPUT"], name)


    def get_sample_tuning_note(self, name: str = None):
        """
        Retrieves a given covid note or selects one randomly from the tuning diredctory; 
        :param name: The note to get
        :type name: str, optional
        :returns: the note and the note's name (helpful when one is selected randomly)
        """
        return self._get_note(self.note_config["DIR_TUNING"], name, self._get_prompt_tuning_note_ids())


    def _process_dir(self, input_dir: str, output_dir : str, experiment: dict, experiment_name: str = 'experiment', note_list = None, skip_list = None) -> None:
        """
        Run an experiment against all notes in input_dir
        :param input_dir: dir of input notes; required
        :param output_dir: dir where responses will be written; required
        :param experiment: dictionary of prompting strategies to run; required
        :param experiment_name: name for the experiment file; 'experiment' by default
        :param note_list: notes to process because we have labels for them; None by default
        :param skip_list: notes to skip because of documented reasons; None by default
        """
        # Record experiment configuration in output
        with open(f'{output_dir}/{experiment_name}.json', 'w') as fp:
            serialized_experiment = {}
            for key, value in experiment.items(): 
                serialized_experiment[key] = value.toJSON()
            json.dump(serialized_experiment, fp)

        # For all files in the source directory
        for index, fname in enumerate(os.listdir(input_dir)):
            note, _ = self._get_note(input_dir, fname)
            print("###################################")
            print(f"# Note {index + 1}: '{fname}'")
            print("###################################")
            for strategy_name, strategy in experiment.items():
                target = f'{output_dir}/{fname}.{strategy_name}'
                # Skip if we already have the file
                if os.path.exists(target):
                    print(f"{target} SKIP (file exists)")
                    next()
                # if we have a skip-list and its in it, ignore
                if skip_list and fname in skip_list:
                    print(f"{target} is believed to cause issues (SKIP)")
                # if we have a note-list, skip notes not in our list 
                # remove any .txt extensions
                elif note_list and (fname.replace('.txt', '') not in note_list):
                    print(f"{fname} is not in the list of notes to inspect (SKIP)")
                    continue
                else:
                    print(f"{target} processing....")
                    response = strategy.run(note) + '\n'
                    with open(target, 'w') as fp:
                        fp.write(response)
                    
        print("###########\n# DONE\n###########\n")

    def run_prompt_tuning(self, experiment: dict, experiment_name: str = 'prompt-tuning-experiment', note_list = None, skip_list = None):  
        """
        Run prompt-tuning data set to determine the most performant prompt
        """
        self._process_dir(
            self.note_config["DIR_TUNING"],
            self.note_config["DIR_OUTPUT_TUNING"],
            experiment,
            experiment_name,
            # Note list for tuning should be the tuning subset by default
            note_list=note_list or self._get_prompt_tuning_note_ids(),
            skip_list=skip_list,
        )

    def run_analysis(self, experiment: dict, experiment_name: str = 'analysis-experiment', note_list = None, skip_list = None):  
        """
        Run prompt-tuning data set to determine the most performant prompt
        """
        # TODO: determine if there's need for a skiplist based on labeled export
        self._process_dir(
            self.note_config["DIR_INPUT"],
            self.note_config["DIR_OUTPUT"],
            experiment,
            experiment_name,
            note_list=note_list,
            skip_list=skip_list,
        )
