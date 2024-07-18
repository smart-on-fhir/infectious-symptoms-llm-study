import json
import os
import re
from pathlib import Path
from argparse import ArgumentParser


# Process all files in {src} of expension {exp}
def process_exp(exp, src, dest):
    llm_responses = os.listdir(src)
    print("Total #files in dir: " + str(len(llm_responses)))
    # only look for files matching this pattern alone ($ avoids substring matches)
    regex = re.escape(exp) + r"$"
    relevant_files = list(
        filter(lambda f_name: re.search(regex, f_name), llm_responses)
    )
    print("# files with exp: " + str(len(relevant_files)))
    # Track the failed json parses
    failures = []
    output = "docref_id,label\n"
    for f_name in relevant_files:
        f_id = f_name.split(".")[0]
        try:
            # Open this file
            file_path = os.path.join(src, f_name)
            with open(file_path, "r") as fp:
                text = fp.read()
                # Try parsing some JSON off the text
                json_excerpt = re.search(r"\{[\w|\W]*\}", text)
                file_json = json.loads(json_excerpt[0])
                # Track if we have a blank JSON object by some chance, or if all values are false
                is_blank = True
                for k, v in file_json.items():
                    if v:
                        is_blank = False
                        # Add a row for each positive symptom mention
                        # NOTE: this adds rows for irrelevant symptoms; no validation
                        output += f"{f_id},{k}\n"
                # Add an empty row if all symptoms are false or if JSON is empty
                if is_blank:
                    output += f"{f_id},\n"

        except Exception as e:
            # Any files we fail to parse JSON for should have a blank row
            print(f"> Could not parse valid json from {f_name}")
            print(str(e))
            output += f"{f_id},\n"
            # Track failures
            failures.append(f_name)

    with open(Path(f"{dest}/{exp}.csv").resolve(), "w") as fp:
        fp.write(output)

    if len(failures) > 0:
        print("# failing documents: ", str(len(failures)) + "\n")
        with open(Path(f"{dest}/{exp}-failures.json"), "w") as fp:
            fp.write(json.dumps(failures))
    return failures


def main(experiments, src, dest):
    all_failures = {}
    experiments = [exp.strip() for exp in experiments]
    for exp in experiments:
        print("===============================")
        print("Processing exp: " + exp)
        print("===============================")
        # Process_exp returns failures
        all_failures[exp] = process_exp(exp, src, dest)
    print()
    print("===============================")
    print("# All Failures: ")
    print("===============================")
    print(json.dumps(all_failures, indent=4))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiments",
        dest="experiments",
        help="Comma-separated list of strategies to generate labels for",
    )
    parser.add_argument(
        "-s",
        "--src",
        dest="src",
        default="./data/responses",
        help="Directory containing LLM responses to process; relative or absolute paths should work",
    )
    parser.add_argument(
        "-d",
        "--dest",
        dest="dest",
        default="./data/",
        help="Where the pipeline instructions file will be written",
    )

    args = parser.parse_args()
    if not args.experiments:
        print("Must provide (1 or many) experiments to process. See below:\n ")
        parser.print_help()
        exit(1)
    experiments = re.split(r"\s*,\s*", args.experiments)

    try:
        src = Path(f"{args.src}").resolve()
    except:
        print(
            f"Could not resolve --src of {args.src}; make sure this directory contains your LLM responses."
        )
        parser.print_help()
        exit(1)

    try:
        dest = Path(f"{args.dest}").resolve()
    except:
        print(f"Could not resolve --dest {args.dest}; make sure this directory exists")
        parser.print_help()
        exit(1)

    main(experiments, src, dest)
