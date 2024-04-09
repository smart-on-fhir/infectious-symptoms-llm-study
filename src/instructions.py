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

def join_lines(lines: list) -> str:
    """
    :param lines: a list of lines to concatenate 
    :return: a single string joining all list-items with newlines, including a trailing newline
    """ 
    return '\n'.join(lines) + '\n'

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
    'You are a helpful assistant identifying COVID-19 symptoms from emergency department notes.',
    'Output the positively documented COVID-19 symptoms, looking out specifically for the following: ' + ', '.join(symptom_list) + '.',
    'Symptoms only need to be positively mentioned once to be included.',
    'Do not mention symptoms that are not present in the note.',
]
def identity_instruction():
    return join_lines(identity)

###############################################################################
#
# Simple prompt
#
simple = [
    identity_instruction(),
    f'Follow these rules:',
    f'Rule (1): {crit1}',
    f'Rule (2): {crit2}',
    f'Rule (3): {crit3}',
]
def simple_instruction():
    return join_lines(simple)


###############################################################################
#
# Inclusion specifics
#
include = [
    identity_instruction(),
    f'Follow these rules:',
    f'Rule (1): {crit1}',
    f'Rule (2): {crit2}',
    # Include criteria don't end with periods
    f'Include positive symptoms from these medical section headings: {quote_join(crit2_include)}.',
    f'Rule (3): {crit3}',
    # Include criteria don't end with periods
    f'Include positive mentions of: {quote_join(quote_unpack(crit3_include))}.',
]
def include_instruction():
    return join_lines(include)

###############################################################################
#
# Exclusion specifics
#

exclude = [
    identity_instruction(),
    f'Follow these rules:',
    f'Rule (1): {crit1}',
    f'Rule (2): {crit2}',
    f'Exclude symptoms from these medical section headings: {quote_join(crit2_exclude)}.'
    f'Rule (3): {crit3}',
    f'Exclude these symptoms: {quote_join(quote_unpack(crit3_exclude))}.',
]
def exclude_instruction():
    return join_lines(exclude)

###############################################################################
#
# Verbose specifics
#
verbose = [ 
    identity_instruction(),
    f'Follow these rules:',
    f'Rule (1): {crit1}',
    f'Rule (2): {crit2}',
    f'Include positive symptoms from these medical section headings: {quote_join(crit2_include)}.',
    f'Exclude all symptoms from these medical section headings: {quote_join(crit2_exclude)}.',
    f'Rule (3): {crit3}',
    f'Include positive mentions of these medical terms: {quote_join(quote_unpack(crit3_include))}.',
    f'Exclude these symptoms: {quote_join(quote_unpack(crit3_exclude))}.',
]
def verbose_instruction():
    return join_lines(verbose)


# See the prompts we've defined so far
def show_instructions():
    print('\n###############################################################')
    print('# identity_instruction() ')
    print(identity_instruction())

    print('\n###############################################################')
    print('# simple_instruction() ')
    print(simple_instruction())

    print('\n###############################################################')
    print('# include_instruction() ')
    print(include_instruction())

    print('\n###############################################################')
    print('# exclude_instruction() ')
    print(exclude_instruction())

    print('\n###############################################################')
    print('# verbose_instruction() ')
    print(verbose_instruction())


if __name__ == "__main__":
    show_instructions()
