import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob


RESULTS_DIR = "results"
OUTPUT_DIR = "evaluation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Normalize GPT answer to yes/no/unknown

def normalize_answer(ans):
    if not isinstance(ans, str):
        return "unknown"
    ans = ans.strip().lower()
    if ans.startswith("y"):
        return "yes"
    if ans.startswith("n"):
        return "no"
    return "unknown"


# Load all JSON results into a single DataFrame

def load_all_results():
    rows = []
    json_files = glob(os.path.join(RESULTS_DIR, "*.json"))

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            item["filepath"] = os.path.basename(jf)
            item["gpt_raw_answer_norm"] = normalize_answer(item["gpt_raw_answer"])
            rows.append(item)

    df = pd.DataFrame(rows)
    return df


# Compute hallucination metrics

def compute_overall_metrics(df):
    df["is_fp"] = (df["flag"] == 0) & (df["gpt_raw_answer_norm"] == "yes")

    grouped = df.groupby(["model", "prompt"]).agg(
        total=("object", "count"),
        fp=("is_fp", "sum")
    ).reset_index()

    grouped["hallucination_rate"] = grouped["fp"] / grouped["total"]

    return grouped


def compute_object_level(df):
    df["is_fp"] = (df["flag"] == 0) & (df["gpt_raw_answer_norm"] == "yes")

    g = df.groupby(["model", "prompt", "object"]).agg(
        total=("object", "count"),
        fp=("is_fp", "sum")
    ).reset_index()

    g["hallucination_rate"] = g["fp"] / g["total"]
    return g


def compute_folder_level(df):
    df["is_fp"] = (df["flag"] == 0) & (df["gpt_raw_answer_norm"] == "yes")

    g = df.groupby(["model", "prompt", "foldername"]).agg(
        total=("object", "count"),
        fp=("is_fp", "sum")
    ).reset_index()

    g["hallucination_rate"] = g["fp"] / g["total"]
    return g


# Visualization Helpers

def plot_overall_bar(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="prompt",
        y="hallucination_rate",
        hue="model"
    )
    plt.xticks(rotation=45)
    plt.title("Hallucination Rate by Model & Prompt")
    plt.ylabel("Hallucination Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overall_hallucination_rate.png"))
    plt.close()


def plot_object_heatmap(df):
    pivot = df.pivot_table(
        index="object",
        columns="prompt",
        values="hallucination_rate"
    )

    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds")
    plt.title("Object-level Hallucination Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "object_hallucination_heatmap.png"))
    plt.close()


def plot_folder_heatmap(df):
    pivot = df.pivot_table(
        index="foldername",
        columns="prompt",
        values="hallucination_rate"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Folder-level Hallucination Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "folder_hallucination_heatmap.png"))
    plt.close()


# MAIN

def main():
    print("Loading JSON result files...")
    df = load_all_results()
    print(f"Loaded {len(df)} rows.")

    # Compute evaluation
    overall = compute_overall_metrics(df)
    obj_level = compute_object_level(df)
    folder_level = compute_folder_level(df)

    # Save CSVs
    overall.to_csv(os.path.join(OUTPUT_DIR, "overall_metrics.csv"), index=False)
    obj_level.to_csv(os.path.join(OUTPUT_DIR, "object_level_metrics.csv"), index=False)
    folder_level.to_csv(os.path.join(OUTPUT_DIR, "folder_level_metrics.csv"), index=False)

    print("CSV files saved.")

    # Plots
    print("Generating graphs...")
    plot_overall_bar(overall)
    plot_object_heatmap(obj_level)
    plot_folder_heatmap(folder_level)

    print("All evaluation outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
