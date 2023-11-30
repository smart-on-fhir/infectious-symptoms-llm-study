###############################################################################
#
# Formatting helper functions
#
def quote(text: str) -> str:
    """
    :return: text quoted like "quoted text"
    """
    return f'"{text}"'

def quote_list(criteria: list) -> list:
    """
    :return: text quoted like ["fever","cough"]
    """
    return [quote(item) for item in criteria]

def quote_join(criteria: list, sep=', ') -> str:
    """
    :return: text quoted like "fever", "cough"
    """
    return sep.join(quote_list(criteria))

def quote_unpack(synonyms: dict) -> list:
    """
    :param synonyms: dictionary of synonyms (example: criteria 3)
    :return: list of synonym terms defined for each key (covid symptom)
    """
    unpacked = list()
    for outer in [val for val in synonyms.values()]:
        for inner in outer:
            unpacked.append(inner)
    return unpacked


###############################################################################
#
# COVID Symptoms
#
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
    'Sore throat'
]

###############################################################################
#
# Criteria (simplest view)
#
crit1 = 'Symptoms must be positively documented and relevant to the presenting illness or reason for visit.'
crit2 = 'Medical section headings must be specific to the present emergency department encounter.'
crit3 = 'Positive symptom mentions must be a definite medical synonym.'

###############################################################################
#
# criteria 1: Present ED Visit only
#

crit1_include = ['positive symptom mentions documented pertaining to the present emergency department encounter']
crit1_exclude = ['symptoms that are negative or denied']

###############################################################################
#
# criteria 2: Medical Section Headings
#
crit2_include = [
    'Chief Complaint',
    'History of Present Illness', 'HPI',
    'Review of Systems',
    'Physical Exam',
    'Vital Signs',
    'Assessment and Plan',
    'Medical Decision Making'
]

crit2_exclude = [
    'Past Medical History', 'PMH'
    'Family History', 'FHX',
    'Social History', 'SHX',
    'Medications',
    'Allergies',
    'Imaging',
    'Diagnostic Study'
]

###############################################################################
#
# criteria 3: Medical Synonyms
#
crit3_include = {
    'Loss of taste or smell': ['anosmia', 'loss of taste', 'loss of smell'],
    'Congestion or runny nose': ['rhinorrhea', 'congestion', 'discharge', 'nose is dripping', 'runny nose', 'stuffy nose'],
    'Cough': ['cough', 'tussive or post-tussive', 'cough is unproductive', 'productive cough', 'dry cough', 'wet cough', 'producing sputum'],
    'Diarrhea': ['diarrhea', 'watery stool'],
    'Fatigue': ['fatigue', 'tired', 'exhausted', 'weary', 'malaise', 'feeling generally unwell'],
    'Fever or Chills': ['fever', 'pyrexia', 'chills', 'temperature greater than or equal 100.4 Fahrenheit or 38 celsius', 'Temperature >= 100.4F', 'Temperature >= 38C'],
    'Headache': ['headache', 'HA', 'migraine', 'cephalgia', 'head pain'],
    'Muscle or body aches': ['muscle or body aches', 'muscle aches', 'generalized aches and pains', 'body aches', 'myalgias', 'myoneuralgia', 'soreness', 'generalized aches and pains'],
    'Nausea or vomiting': ['nausea or vomiting', 'Nausea', 'vomiting', 'emesis', 'throwing up', 'queasy', 'regurgitated'],
    'Shortness of breath or difficulty breathing': ['shortness of breath', 'difficulty breathing', 'SOB', 'Dyspnea', 'breathing is short', 'increased breathing', 'labored breathing', 'distressed breathing'],
    'Sore throat': ['sore throat', 'throat pain', 'pharyngeal pain', 'pharyngitis', 'odynophagia']
}
# With Myalgia changes
crit3_include_myalgia = {
    'Loss of taste or smell': ['anosmia', 'loss of taste', 'loss of smell'],
    'Congestion or runny nose': ['rhinorrhea', 'congestion', 'discharge', 'nose is dripping', 'runny nose', 'stuffy nose'],
    'Cough': ['cough', 'tussive or post-tussive', 'cough is unproductive', 'productive cough', 'dry cough', 'wet cough', 'producing sputum'],
    'Diarrhea': ['diarrhea', 'watery stool'],
    'Fatigue': ['fatigue', 'tired', 'exhausted', 'weary', 'malaise', 'feeling generally unwell'],
    'Fever or Chills': ['fever', 'pyrexia', 'chills', 'temperature greater than or equal 100.4 Fahrenheit or 38 celsius', 'Temperature >= 100.4F', 'Temperature >= 38C'],
    'Headache': ['headache', 'HA', 'migraine', 'cephalgia', 'head pain'],
    'Myalgia': ['muscle or body aches', 'muscle aches', 'generalized aches and pains', 'body aches', 'myalgia', 'myalgias', 'myoneuralgia', 'soreness', 'generalized aches and pains'],
    'Nausea or vomiting': ['nausea or vomiting', 'Nausea', 'vomiting', 'emesis', 'throwing up', 'queasy', 'regurgitated'],
    'Shortness of breath or difficulty breathing': ['shortness of breath', 'difficulty breathing', 'SOB', 'Dyspnea', 'breathing is short', 'increased breathing', 'labored breathing', 'distressed breathing'],
    'Sore throat': ['sore throat', 'throat pain', 'pharyngeal pain', 'pharyngitis', 'odynophagia']
}
crit3_exclude = {
    'Loss of taste or smell': ['injury related to loss of taste or smell'],
    'Congestion or runny nose': [],
    'Cough': ['wheezing', 'crackles', 'croup'],
    'Diarrhea': ['loose stool', 'bloody stool'],
    'Fatigue': ['looked ill'],
    'Fever or Chills': ['afebrile', 'felt warm'],
    'Headache': ['headache due to injury'],
    'Muscle or body aches': ['localized pain', 'injury', 'abdominal pain', 'lower back pain'],
    'Nausea or vomiting': ['gastritis', 'gastroparesis'],
    'Shortness of breath or difficulty breathing': ['BiPAP', 'CPAP', 'oxygen need'],
    'Sore throat': ['streptococcus', 'dysphagia', 'hoarseness', 'red throat']
}

###############################################################################
#
# Chain of thought text 
#   to be combined with identity prompts later
#
CHAIN_OF_THOUGHT_instruction = """
Use Chain-of-Thought methodology to justify summarizations.
For example, point to the verbatim text from the note that you are summarizing.
You must do this to improve the explainability of your responses. 
Again, use chain of thought methodology to justify your summaries.
"""

###############################################################################
#
# Identity
#

identity = [
    'Summarize the emergency department ED note using simple language',
    'Output the positively documented COVID-19 symptoms',
    'Symptoms only need to be positively mentioned once to be included',
    'Do NOT explain your answers'
]
def identity_instruction():
    return '.\n'.join(identity) + '.\n'

identity_no_covid = [
    'Summarize the emergency department ED note using simple language',
    'Output the positively documented symptoms',
    'Symptoms only need to be positively mentioned once to be included',
    'Do NOT explain your answers'
]
def identity_no_covid_instruction():
    return '.\n'.join(identity_no_covid) + '.\n'

identity_no_covid_specific_symptoms = [
    'Summarize the emergency department ED note using simple language',
    'Output only present positive (+) mentions of symptoms relating to: ' + ', '.join(symptom_list),
    'Symptoms only need to be positively mentioned once to be included',
    'Do NOT explain your answers'
]
def identity_no_covid_specific_symptoms_instruction():
    return '.\n'.join(identity_no_covid_specific_symptoms) + '.\n'

def identity_chain_of_thought_instruction(): 
    return '.\n'.join([CHAIN_OF_THOUGHT_instruction, *identity])

def identity_symptom_specific_instruction(symptom: str): 
    return '.\n'.join([
    'Summarize ' + symptom + ' information from this department ED note using simple language',
    'Output only present positive (+) mentions of symptoms relating to ' + symptom,
    'Symptoms only need to be positively mentioned once to be included',
    'Do NOT mention symptoms unrelated to ' + symptom,
    'Do NOT explain your answers'
])

###############################################################################
#
# Simple prompt
#
def simple_instruction():
    simple = identity_instruction()
    simple += f'\nFollow these rules:'
    simple += f'\nRule (1): {crit1} '
    simple += f'\nRule (2): {crit2} '
    simple += f'\nRule (3): {crit3} '
    return simple
def simple_no_covid_instruction():
    simple = identity_no_covid_instruction()
    simple += f'\nFollow these rules:'
    simple += f'\nRule (1): {crit1} '
    simple += f'\nRule (2): {crit2} '
    simple += f'\nRule (3): {crit3} '
    return simple
def simple_no_covid_specific_symptoms_instruction():
    simple = identity_no_covid_specific_symptoms_instruction()
    simple += f'\nFollow these rules:'
    simple += f'\nRule (1): {crit1} '
    simple += f'\nRule (2): {crit2} '
    simple += f'\nRule (3): {crit3} '
    return simple
def simple_chain_of_thought_instruction():
    simple = identity_chain_of_thought_instruction()
    simple += f'\nFollow these rules:'
    simple += f'\nRule (1): {crit1} '
    simple += f'\nRule (2): {crit2} '
    simple += f'\nRule (3): {crit3} '
    return simple
def simple_specific_symptom_instruction(symptom: str):
    simple = identity_symptom_specific_instruction(symptom)
    simple += f'\nFollow these rules:'
    simple += f'\nRule (1): {crit1} '
    simple += f'\nRule (2): {crit2} '
    simple += f'\nRule (3): {crit3} '
    return simple

###############################################################################
#
# Inclusion specifics
#
def include_instruction():
    include = identity_instruction()
    include += f'\nFollow these rules:'
    include += f'\nRule (1): {crit1} '
    include += f'\nRule (2): {crit2} '
    include += f'\n Include positive symptoms from these medical section headings: ' + quote_join(crit2_include) + '.'
    include += f'\nRule (3): {crit3} '
    include += f'\n Include positive mentions of: ' + quote_join(quote_unpack(crit3_include)) + '.'
    return include
def include_no_covid_instruction():
    include = identity_no_covid_instruction()
    include += f'\nFollow these rules:'
    include += f'\nRule (1): {crit1} '
    include += f'\nRule (2): {crit2} '
    include += f'\n Include positive symptoms from these medical section headings: ' + quote_join(crit2_include) + '.'
    include += f'\nRule (3): {crit3} '
    include += f'\n Include positive mentions of: ' + quote_join(quote_unpack(crit3_include)) + '.'
    return include
def include_myalgia_instruction():
    include = identity_instruction()
    include += f'\nFollow these rules:'
    include += f'\nRule (1): {crit1} '
    include += f'\nRule (2): {crit2} '
    include += f'\n Include positive symptoms from these medical section headings: ' + quote_join(crit2_include) + '.'
    include += f'\nRule (3): {crit3} '
    include += f'\n Include positive mentions of: ' + quote_join(quote_unpack(crit3_include_myalgia)) + '.'
    return include
def include_no_covid_myalgia_instruction():
    include = identity_no_covid_instruction()
    include += f'\nFollow these rules:'
    include += f'\nRule (1): {crit1} '
    include += f'\nRule (2): {crit2} '
    include += f'\n Include positive symptoms from these medical section headings: ' + quote_join(crit2_include) + '.'
    include += f'\nRule (3): {crit3} '
    include += f'\n Include positive mentions of: ' + quote_join(quote_unpack(crit3_include_myalgia)) + '.'
    return include

###############################################################################
#
# Exclusion specifics
#
def exclude_instruction():
    exclude = identity_instruction()
    exclude += f'\nFollow these rules:'
    exclude += f'\nRule (1): {crit1} '
    exclude += f'\nRule (2): {crit2} '
    exclude += f'\n Exclude symptoms from these medical section headings: ' + quote_join(crit2_exclude) + '.'
    exclude += f'\nRule (3): {crit3} '
    exclude += f'\n Exclude these symptoms: ' + quote_join(quote_unpack(crit3_exclude)) + '.'
    return exclude
def exclude_no_covid_instruction():
    exclude = identity_no_covid_instruction()
    exclude += f'\nFollow these rules:'
    exclude += f'\nRule (1): {crit1} '
    exclude += f'\nRule (2): {crit2} '
    exclude += f'\n Exclude symptoms from these medical section headings: ' + quote_join(crit2_exclude) + '.'
    exclude += f'\nRule (3): {crit3} '
    exclude += f'\n Exclude these symptoms: ' + quote_join(quote_unpack(crit3_exclude)) + '.'
    return exclude

###############################################################################
#
# Verbose specifics
#
def verbose_instruction():
    verbose = identity_instruction()
    verbose += f'\nFollow these rules:'
    verbose += f'\nRule (1): {crit1} '
    verbose += f'\nRule (2): {crit2} '
    verbose += f'\n Include positive symptoms from these medical section headings: ' + quote_join(crit2_include) + '.'
    verbose += f'\n Exclude all symptoms from these medical section headings: ' + quote_join(crit2_exclude) + '.'
    verbose += f'\nRule (3): {crit3} '
    verbose += f'\n Include positive mentions of these medical terms: ' + quote_join(quote_unpack(crit3_include)) + '.'
    verbose += f'\n Exclude these symptoms: ' + quote_join(quote_unpack(crit3_exclude)) + '.'
    return verbose
def verbose_no_covid_instruction():
    verbose = identity_no_covid_instruction()
    verbose += f'\nFollow these rules:'
    verbose += f'\nRule (1): {crit1} '
    verbose += f'\nRule (2): {crit2} '
    verbose += f'\n Include positive symptoms from these medical section headings: ' + quote_join(crit2_include) + '.'
    verbose += f'\n Exclude all symptoms from these medical section headings: ' + quote_join(crit2_exclude) + '.'
    verbose += f'\nRule (3): {crit3} '
    verbose += f'\n Include positive mentions of these medical terms: ' + quote_join(quote_unpack(crit3_include)) + '.'
    verbose += f'\n Exclude these symptoms: ' + quote_join(quote_unpack(crit3_exclude)) + '.'
    return verbose


# See the prompts we've defined so far
def show_usage():
    print('\n###############################################################')
    print('# identity_instruction() ')
    print(identity_instruction())

    print('\n###############################################################')
    print('# identity_no_covid_instruction() ')
    print(identity_no_covid_instruction())

    print('\n###############################################################')
    print('# identity_no_covid_specific_symptoms_instruction() ')
    print(identity_no_covid_specific_symptoms_instruction())

    print('\n###############################################################')
    print('# identity_chain_of_thought_instruction() ')
    print(identity_chain_of_thought_instruction())

    print('\n###############################################################')
    print('# identity_symptom_specific_instruction(' + symptom_list[0] + ') ')
    print(identity_symptom_specific_instruction(symptom_list[0]))

    print('\n###############################################################')
    print('# simple_instruction() ')
    print(simple_instruction())

    print('\n###############################################################')
    print('# simple_no_covid_instruction() ')
    print(simple_no_covid_instruction())

    print('\n###############################################################')
    print('# simple_no_covid_specific_symptoms_instruction() ')
    print(simple_no_covid_specific_symptoms_instruction())

    print('\n###############################################################')
    print('# simple_chain_of_thought_instruction() ')
    print(simple_chain_of_thought_instruction())

    print('\n###############################################################')
    print('# simple_specific_symptom_instruction(' + symptom_list[0] + ') ')
    print(simple_specific_symptom_instruction(symptom_list[0]))

    print('\n###############################################################')
    print('# include_instruction() ')
    print(include_instruction())

    print('\n###############################################################')
    print('# include_no_covid_instruction() ')
    print(include_no_covid_instruction())

    print('\n###############################################################')
    print('# include_myalgia_instruction() ')
    print(include_myalgia_instruction())

    print('\n###############################################################')
    print('# include_no_covid_myalgia_instruction() ')
    print(include_no_covid_myalgia_instruction())

    print('\n###############################################################')
    print('# exclude_instruction() ')
    print(exclude_instruction())

    print('\n###############################################################')
    print('# exclude_no_covid_instruction() ')
    print(exclude_no_covid_instruction())

    print('\n###############################################################')
    print('# verbose_instruction() ')
    print(verbose_instruction())

    print('\n###############################################################')
    print('# verbose_no_covid_instruction() ')
    print(verbose_no_covid_instruction())


if __name__ == "__main__":
    show_usage()
