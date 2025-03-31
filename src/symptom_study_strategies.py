from pathlib import Path 
import pprint
import json
from src.instructions import (
    exclude_instruction,
    identity_instruction,
    include_instruction,
    join_lines,
    rules_instruction,
    symptom_list,
    verbose_instruction,
)
from src.strategy import Strategy


def build_strategies():
    ##################################################
    #
    # Basic Strats
    #
    def newStrat(inst):
        return Strategy([{"instruction": inst}])

    # 5 Strategies
    sampleIdentity = newStrat(identity_instruction())
    sampleRules = newStrat(rules_instruction())
    sampleInclude = newStrat(include_instruction())
    sampleExclude = newStrat(exclude_instruction())
    sampleVerbose = newStrat(verbose_instruction())

    ##################################################
    #
    # Simplification for double-pass experiments
    #
    simplify = join_lines(
        [
            "You are an expert editor reviewing a clinical note summary.",
            "The previous reviewer may have included irrelevant, negative symptoms in their summarization.",
            'Simplify repetitive information found in this summary, and remove mentions of negative symptoms from this summary (e.g. "No X, No recent Y, No recent changes in Z").',
            "ONLY reply with your new summary. Do NOT explain your answers.",
        ]
    )

    def newStratSimplification(inst):
        return Strategy(
            [
                {"instruction": inst},
                {"instruction": simplify, "step_type": "previous"},
            ],
        )

    # 5 Double-Pass Strategies
    sampleRulesSimplification = newStratSimplification(rules_instruction())
    sampleIdentitySimplification = newStratSimplification(identity_instruction())
    sampleIncludeSimplification = newStratSimplification(include_instruction())
    sampleExcludeSimplification = newStratSimplification(exclude_instruction())
    sampleVerboseSimplification = newStratSimplification(verbose_instruction())

    ##################################################
    #
    # JSON Strats
    #
    # Define the JSON schema INST
    jsonSchemaInst = join_lines(
        [
            "Your reply must be parsable as JSON.",
            "Format your response using only the following JSON schema: {"
            + ", ".join(f'"{symptom}": boolean' for symptom in symptom_list)
            + "}. Each JSON key should correspond to a symptom, and each value should be true if that symptom is indicated in the clinical note; false otherwise.",
            "Never explain yourself, and only reply with JSON.",
        ]
    )
    # 5 Baseline Strategies, JSON
    sampleRulesJSON = newStrat(rules_instruction() + jsonSchemaInst)
    sampleIdentityJSON = newStrat(identity_instruction() + jsonSchemaInst)
    sampleIncludeJSON = newStrat(include_instruction() + jsonSchemaInst)
    sampleExcludeJSON = newStrat(exclude_instruction() + jsonSchemaInst)
    sampleVerboseJSON = newStrat(verbose_instruction() + jsonSchemaInst)

    ##################################################
    #
    # JSON Strats, Double Pass
    #
    #
    jsonFixInst = join_lines(
        [
            "If the following text is parseable, valid JSON, do nothing and return the text as is.",
            "Otherwise, remove all non-JSON information. Then format the text so that it's valid, parsable JSON that conforms to the following JSON schema: {"
            + ", ".join(f'"{symptom}": boolean' for symptom in symptom_list)
            + "}",
            "Never explain yourself, and only reply with JSON.",
        ]
    )

    def jsonStratSimplification(inst):
        return Strategy(
            [
                {"instruction": inst + jsonSchemaInst},
                {"instruction": jsonFixInst, "step_type": "previous"},
            ],
        )

    # 5 Baseline Strategies, JSON with Double Pass
    sampleRulesJSONValidation = jsonStratSimplification(rules_instruction())
    sampleIdentityJSONValidation = jsonStratSimplification(identity_instruction())
    sampleIncludeJSONValidation = jsonStratSimplification(include_instruction())
    sampleExcludeJSONValidation = jsonStratSimplification(exclude_instruction())
    sampleVerboseJSONValidation = jsonStratSimplification(verbose_instruction())

    return {
        "rules": sampleRules,
        "identity": sampleIdentity,
        "include": sampleInclude,
        "exclude": sampleExclude,
        "verbose": sampleVerbose,
        "rulesSimplification": sampleRulesSimplification,
        "identitySimplification": sampleIdentitySimplification,
        "includeSimplification": sampleIncludeSimplification,
        "excludeSimplification": sampleExcludeSimplification,
        "verboseSimplification": sampleVerboseSimplification,
        "rulesJSON": sampleRulesJSON,
        "identityJSON": sampleIdentityJSON,
        "includeJSON": sampleIncludeJSON,
        "excludeJSON": sampleExcludeJSON,
        "verboseJSON": sampleVerboseJSON,
        "rulesJSONValidation": sampleRulesJSONValidation,
        "identityJSONValidation": sampleIdentityJSONValidation,
        "includeJSONValidation": sampleIncludeJSONValidation,
        "excludeJSONValidation": sampleExcludeJSONValidation,
        "verboseJSONValidation": sampleVerboseJSONValidation,
    }


def write_strategies():
    """
    Write JSON of strategies to disk in a .json file
    """
    all_strategies = build_strategies()
    pprint.pprint(all_strategies)
    serialized_strats = {}
    for strat_name, strat in all_strategies.items(): 
        serialized_strats[strat_name] = strat.to_json()
    Path("strategies.json").write_text(json.dumps(serialized_strats))
    return all_strategies