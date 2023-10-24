################################################################
# https://replicate.com/blog/how-to-prompt-llama
# 
# Act as if…
# You are…
# Always/Never…
# Speak like…
#
################################################################

DIR_COVID = '/lab-share/CHIP-Mandl-e2/Public/covid-llm/notes'
DIR_MTSMAPLES = '/lab-share/CHIP-Mandl-e2/Public/autollm/ctakes-examples/ctakes_examples/resources/curated'

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
def prompt(prompt: str, system: str = None) -> str:
    system = system or DEFAULT_SYSTEM_PROMPT
    full_prompt = PROMPT_FORMAT % {"system": system.strip(), "user": prompt.strip()}
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

    #print('#################')
    #print(answer)
    #print('#################')          

    return answer


# This grabs a note from our corpus of curated (fake) notes (no PHI).
# Visit the following github directory to see the list of names you can use:
# https://github.com/Machine-Learning-for-Medical-Language/ctakes-examples/tree/main/ctakes_examples/resources/curated
def get_mtsample(name: str) -> str:
    path = f"{DIR_MTSMAPLES}/{name}"
    with open(path, "r", encoding="utf8") as f:
        return f.read().strip()
    
def get_covid_note(name: str) -> str:
    path = f"{DIR_COVID}/{name}"
    with open(path, "r", encoding="utf8") as f:
        return f.read().strip()

################################################################

symptom_list = [
    'Congestion or runny nose',
    'Cough',
    'Diarrhea',
    'Dyspnea',
    'Fatigue',
    'Fever or chills',
    'Headache',
    'Loss of taste or smell',
    'Muscle or body aches',
    'Nausea or vomiting',
    'Sore throat']

crit1 = 'Encounter time. Symptoms must be relevant to the present encounter either as the reason for visit, documented symptom, or observed sign.'
crit2 = 'Medical section heading. Patient symptoms are present and not related to past medical history or a medication prescribed unrelated to the present encounter.'
crit3 = 'Positive symptom mentions must explicitly mention the symptom or synonym.'

crit2_include=['Chief complaint',
               'History of presenting illness',
               'Review of systems',
               'Physical exam',
               'Vital signs',
               'Assessment and plan',
               'hospital course',
               'Assessment and plan',
               'diagnosis']

crit2_exclude=[
    'Past medical history',
    'Social history',
    'History',
    'Medication list',
    'Imaging',
    'Diagnostic Study']

crit3_include={
      'Loss of taste or smell': ['Anosmia', 'loss of taste', 'loss of smell']
    , 'Congestion or runny nose' : ['Rhinorrhea', 'congestion', 'discharge', 'nose is dripping', 'runny nose', 'stuffy nose']
    , 'Cough' : ['Cough', 'Tussive or post-tussive', 'cough is unproductive', 'productive cough', 'dry cough', 'wet cough', 'producing sputum']
    , 'Diarrhea' : ['Diarrhea', 'watery stool']
    , 'Fatigue' : ['Fatigue', 'tired', 'exhausted', 'weary', 'malaise', 'feeling generally unwell']
    , 'Fever or Chills' : ['Fever', 'pyrexia', 'chills', 'temperature greater than or equal 100.4 Fahrenheit or 38 celsius', 'Temperature >= 100.4F', 'Temperature >= 38C']
    , 'Headache' : ['Headache', 'HA', 'migraine', 'cephalgia', 'head pain']
    , 'Muscle or body aches' : ['muscle or body aches', 'muscle aches', 'generalized aches and pains', 'body aches', 'myalgias', 'myoneuralgia', 'soreness', 'generalized aches and pains']
    , 'Nausea or vomiting' : ['Nausea or vomiting', 'Nausea', 'vomiting', 'emesis', 'throwing up', 'queasy', 'regurgitated']
    , 'Shortness of breath or difficulty breathing' : ['Shortness of breath', 'difficulty breathing', 'SOB', 'Dyspnea', 'breathing is short', 'increased breathing', 'labored breathing', 'distressed breathing']
    , 'Sore throat' : ['Sore throat', 'throat pain', 'pharyngeal pain', 'pharyngitis', 'odynophagia'] 
}

crit3_exclude={
      'Loss of taste or smell': ['Injury related to loss of taste or smell']
    , 'Congestion or runny nose' : []
    , 'Cough' : ['Wheezing', 'crackles', 'croup']
    , 'Diarrhea' : ['Loose stool', 'bloody stool']
    , 'Fatigue' : ['Looked ill']
    , 'Fever or Chills' : ['Afebrile', 'felt warm']
    , 'Headache' : ['Headache due to injury']
    , 'Muscle or body aches' : ['Localized pain', 'injury', 'abdominal pain', 'lower back pain']
    , 'Nausea or vomiting' : ['Gastritis', 'gastroparesis']
    , 'Shortness of breath or difficulty breathing' : ['BiPAP', 'CPAP', 'oxygen need']
    , 'Sore throat' : ['Streptococcus', 'dysphagia', 'hoarseness', 'red throat'] 
}
crit3_exclude_list = list() 
for outer in [val for val in crit3_exclude.values()]:
    for inner in outer:
        crit3_exclude_list.append(inner)


criteria_simple = f'Three criteria define True or False for each symptom.\n Criteria 1 is {crit1}\n Criteria 2 is {crit2}\n Criteria 3 is {crit3}\n'

criteria_verbose = f'Three criteria define True or False for each symptom.\n'
criteria_verbose+= f'Criteria 1 is {crit1}\n\n'

criteria_verbose+= f'Criteria 2 is {crit2}\n\n'
criteria_verbose+= f'Include positive symptom mentions from these medical section headings :\n' + ',\n'.join(crit2_include) +'.\n\n'
criteria_verbose+= f'Exclude any symptom mentions from these medical section headings :\n' + ',\n'.join(crit2_exclude) +'.\n\n'

criteria_verbose+= f'Criteria 3 is {crit3}\n\n'
criteria_verbose+= f'Include positive symptom for this JSON of symptom synonyms:\n' + json.dumps(crit3_include) +'.\n\n'
criteria_verbose+= f'Exclude all symptom for this JSON of symptom synonyms:\n' + json.dumps(crit3_exclude_list) +'.\n\n'

print(criteria_verbose)

criteria_exclude = f'Three criteria define True or False for each symptom.\n'
criteria_exclude+= f'Criteria 1 is {crit1}\n\n'

criteria_exclude+= f'Criteria 2 is {crit2}\n\n'
criteria_exclude+= f'Exclude any symptom mentions from these medical section headings :\n' + ',\n'.join(crit2_exclude) +'.\n\n'

criteria_exclude+= f'Criteria 3 is {crit3}\n\n'
criteria_exclude+= f'Exclude all symptom for this JSON of symptom synonyms:\n' + json.dumps(crit3_exclude_list) +'.\n\n'

print(criteria_exclude)

prompt_identity = 'Act as if you are a medical records reviewer for research. You will provide the best estimate without explaining your answers.'
prompt_identity+= ' Always output your answer in JSON where the values are True or False keys are the name of each symptom:\n' + ',\n'.join(symptom_list) + '.\n\n'

prompt_simple = prompt_identity +'\n'+ criteria_simple
prompt_exclude = prompt_identity +'\n'+ criteria_exclude
prompt_verbose = prompt_identity +'\n'+ criteria_verbose

prompt_select = prompt_simple

################################################################

for fname in os.listdir(DIR_COVID): 
    print(f"{fname} processing....")
    note = get_covid_note(fname)
    print('##################')
    print(note)
    print('##################')
    response = prompt(note, prompt_simple)
    with open(f"/home/ch112531/output/{fname}.llm.simple", 'w') as fp: 
        fp.write(response)
