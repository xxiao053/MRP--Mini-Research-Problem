import os
import json
import pandas as pd

RESULTS_DIR = "results"
MODEL_NAME = "gpt-5.1"


########################################
# Helper: normalize GPT answers
########################################

def normalize(ans):
    if not isinstance(ans, str):
        return "unknown"
    ans = ans.lower().strip()
    if ans.startswith("y"):
        return "yes"
    if ans.startswith("n"):
        return "no"
    return "unknown"


########################################
# Load results for gpt-5.1 only
########################################

def load_results(model):
    files = [
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith(model)
    ]

    rows = []
    for fpath in files:
        full = os.path.join(RESULTS_DIR, fpath)
        with open(full, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            item["prompt"] = item["prompt"]
            item["gpt_norm"] = normalize(item["gpt_raw_answer"])
            item["model"] = item["model"]
            rows.append(item)

    return pd.DataFrame(rows)


########################################
# Find two typical cases
########################################

def find_cases(df):

    # Only analyze prompts we care about
    base = df[df["prompt"] == "baseline"]
    mis  = df[df["prompt"] == "misleading1"]
    miti = df[df["prompt"] == "mitigate1"]

    # Merge on filename + object
    merge_base_mis = pd.merge(
        base, mis, 
        on=["filename", "object", "foldername", "flag"],
        suffixes=("_base", "_mis")
    )

    merge_base_miti = pd.merge(
        base, miti,
        on=["filename", "object", "foldername", "flag"],
        suffixes=("_base", "_miti")
    )

    ##############################
    # CASE A: baseline OK → misleading hallucinated
    ##############################

    caseA = merge_base_mis[
        (merge_base_mis["flag"] == 0) & 
        (merge_base_mis["gpt_norm_base"] == "no") &   # baseline correct
        (merge_base_mis["gpt_norm_mis"] == "yes")     # misleading hallucinated
    ]

    ##############################
    # CASE B: baseline hallucinated → mitigation fixed
    ##############################

    caseB = merge_base_miti[
        (merge_base_miti["flag"] == 0) & 
        (merge_base_miti["gpt_norm_base"] == "yes") &   # baseline hallucinated
        (merge_base_miti["gpt_norm_miti"] == "no")       # mitigation corrected
    ]

    return caseA, caseB


########################################
# MAIN
########################################

def main():

    df = load_results(MODEL_NAME)

    caseA, caseB = find_cases(df)

    print("\n==============================")
    print("CASE A: Baseline correct, Misleading hallucinated")
    print("==============================")
    if len(caseA) == 0:
        print("No example found.")
    else:
        print(caseA[[
            "filename", "foldername", "object", 
            "gpt_norm_base", "gpt_norm_mis"
        ]].head(5))

    print("\n==============================")
    print("CASE B: Baseline hallucinated, Mitigation fixed it")
    print("==============================")
    if len(caseB) == 0:
        print("No example found.")
    else:
        print(caseB[[
            "filename", "foldername", "object", 
            "gpt_norm_base", "gpt_norm_miti"
        ]].head(5))

    # Optional save for your report:
    caseA.to_csv("typical_caseA_misleading.csv", index=False)
    caseB.to_csv("typical_caseB_mitigation.csv", index=False)

    print("\nSaved typical cases to CSV.")


if __name__ == "__main__":
    main()