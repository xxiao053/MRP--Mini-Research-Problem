import os
import csv
import json
import base64
import time
import logging
from openai import OpenAI, RateLimitError


############################################
# 0. CONFIGURATIONS
############################################

# Models you want to support 
# Reference: https://platform.openai.com/docs/models/gpt-5
# Pricing: https://platform.openai.com/docs/pricing 
AVAILABLE_MODELS = [
    "gpt-5.1",     # latest flagship (2025)
    "gpt-5",
    "gpt-4.1",
    "gpt-4o",      # strong multimodal model
]

# User chooses which model to run:
MODEL_TO_USE = "gpt-4o"   # <-- you can modify this

# Folders to run (must match your dataset directory names)
TARGET_FOLDERS = ["person", "car", "dog", "cat", "chair"]   # <-- you can modify this
# Other options: "bottle", "cup", "bicycle", "bird", "laptop"

GROUNDTRUTH_PATH = "GroundTruth.csv"
IMAGE_ROOT = "images"
OUTPUT_DIR = "results"
LOG_FILE = "run_all_prompts.log"


############################################
# 1. LOGGING SETUP
############################################

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


############################################
# 2. PROMPT DEFINITIONS
############################################

def baseline_prompt(obj):
    return f"""
You are an image recognition system. I will give you an image and ask whether a specific object is present.
You must answer strictly with only one word: "yes" or "no". No explanation.

Object to check: "{obj}"

Does this image contain this object?
"""


########### MISLEADING PROMPTS ###########

def misleading1(obj):
    return f"""
You are analyzing this image as part of a visual survey.
In similar images from this dataset, the object "{obj}" often appears,
even if partially hidden or small.

Answer strictly with "yes" or "no":
Is the object "{obj}" present in this image?
"""

def misleading2(obj):
    return f"""
I thought I saw a "{obj}" somewhere in this image, possibly in the background.
Please confirm my observation.

Answer only with "yes" or "no":
Is the object "{obj}" actually present in this image?
"""

def misleading3(obj):
    return f"""
According to visual statistics, images like this often contain a "{obj}", 
even if not the main focus.

Answer strictly with "yes" or "no":
Is a "{obj}" present in this image?
"""

def misleading4(obj):
    return f"""
This image appears to show a scene where objects commonly related to "{obj}" may appear.

Based on this context, answer only “yes” or “no”:
Is a "{obj}" present in the image?
"""


########### MITIGATION PROMPTS ###########

def mitigate1(obj):
    return f"""
You must only answer "yes" if the object is clearly and unambiguously visible in the image.
If the object cannot be visually confirmed, or is uncertain, answer "no".

Object: "{obj}"

Based strictly on visible evidence only, does the image contain this object?
Answer only "yes" or "no".
"""

def mitigate2(obj):
    return f"""
Do not guess.
If you are not completely certain the object is present, answer "no".
Never infer or assume anything beyond what is visually obvious.

Object: "{obj}"

Answer strictly "yes" or "no":
Is this object visible in the image?
"""

def mitigate3(obj):
    return f"""
You are a highly cautious visual verification system designed to avoid hallucination.
You never report an object unless it is clearly identifiable in the pixels.

Object to check: "{obj}"

Answer with "yes" or "no":
Is this object clearly visible?
"""

def mitigate4(obj):
    return f"""
Follow this strict rule:

1. Internally analyze the image and form a detailed understanding of the scene.
2. Internally check if the object "{obj}" is visually obvious.
3. If obvious → final answer "yes".
4. If not obvious → final answer "no".

Do all analysis internally.
For the final output, answer only with a single word: "yes" or "no".
"""


############################################
# 3. PROMPT BANK (user can choose subset)
############################################

PROMPT_TEMPLATES = {
    "baseline": baseline_prompt,

    "misleading1": misleading1,
    "misleading2": misleading2,
    "misleading3": misleading3,
    "misleading4": misleading4,

    "mitigate1": mitigate1,
    "mitigate2": mitigate2,
    "mitigate3": mitigate3,
    "mitigate4": mitigate4,
}

# User chooses which prompt modes to run
PROMPTS_TO_RUN = [
    "baseline",
    "misleading1",
    # "misleading2",
    # "misleading3",
    # "misleading4",
    "mitigate1",
    # "mitigate2",
    # "mitigate3",
    # "mitigate4",
]


############################################
# 4. GPT Vision Call
############################################

API_KEY = os.getenv("OPENAI_API_KEY")   # No key in code, please set your API KEY as environment variable first 

if API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set in your environment variables.")

client = OpenAI(api_key=API_KEY)

def call_with_retries(func, max_retries=8):
    """
    Wrap any OpenAI API call with automatic retries using exponential backoff.
    - If API provides a 'try again in X ms', we honor it.
    - Otherwise we use exponential backoff: 1s, 2s, 4s, 8s, ...
    """

    for retry in range(max_retries):
        try:
            return func()   # attempt the call

        except RateLimitError as e:
            # Try-extract "Retry-After" from error (very reliable)
            msg = str(e)
            logging.warning(f"Rate limit hit: {msg}")

            # Case 1: API tells us wait time (e.g., "try again in 558ms")
            if "try again in" in msg and "ms" in msg:
                try:
                    ms = int(msg.split("try again in")[1].split("ms")[0].strip())
                    wait = ms / 1000
                    logging.warning(f"Waiting {wait:.3f} seconds (from API suggestion)")
                    time.sleep(wait)
                    continue
                except:
                    pass  # fallback to exponential

            # Case 2: fallback → exponential backoff
            wait = min(2 ** retry, 30)  # cap at 30 seconds
            logging.warning(f"Retrying in {wait} seconds (exponential backoff)")
            time.sleep(wait)

        except Exception as e:
            logging.error(f"Non-rate-limit error during API call: {e}")
            raise e

    # If all retries exhausted:
    raise RuntimeError("Max retries reached for API call")

def is_gpt5_model(model_name):
    """
    Returns True if model is part of the GPT-5 family.
    GPT-5 model names typically start with 'gpt-5'.
    """
    return model_name.startswith("gpt-5")

def ask_gpt(image_path, object_name, prompt_fn):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = prompt_fn(object_name)

    def api_call():
        kwargs = {
            "model": MODEL_TO_USE,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        },
                    ],
                }
            ]
        }

        # GPT-5 uses max_completion_tokens instead of max_tokens
        if is_gpt5_model(MODEL_TO_USE):
            kwargs["max_completion_tokens"] = 20
        else:
            kwargs["max_tokens"] = 5

        return client.chat.completions.create(**kwargs)

    resp = call_with_retries(api_call)
    return resp.choices[0].message.content.strip()


############################################
# 5. Main routine for one prompt_type
############################################

def run_prompt_mode(prompt_key):
    prompt_fn = PROMPT_TEMPLATES[prompt_key]
    results = []

    with open(GROUNDTRUTH_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            folder = row["foldername"]
            filename = row["filename"]

            # Only run for selected folders
            if folder not in TARGET_FOLDERS:
                continue

            no_list = eval(row["no"])
            image_path = os.path.join(IMAGE_ROOT, folder, filename)

            if not os.path.exists(image_path):
                logging.warning(f"Missing image: {image_path}")
                continue

            for obj in no_list:
                logging.info(f"Model={MODEL_TO_USE} Prompt={prompt_key} Image={filename} Object={obj}")

                raw_ans = ask_gpt(image_path, obj, prompt_fn)

                results.append({
                    "model": MODEL_TO_USE,
                    "prompt": prompt_key,
                    "filename": filename,
                    "foldername": folder,
                    "object": obj,
                    "flag": 0,
                    "gpt_raw_answer": raw_ans,
                })

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_path = os.path.join(
        OUTPUT_DIR,
        f"{MODEL_TO_USE}_{prompt_key}_results.json"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logging.info(f"Saved file: {out_path}")


############################################
# 6. MAIN — run for all selected prompts
############################################

def main():
    start = time.time()

    if MODEL_TO_USE not in AVAILABLE_MODELS:
        logging.warning(f"Model '{MODEL_TO_USE}' is not known in 2025 model list.")

    for key in PROMPTS_TO_RUN:
        run_prompt_mode(key)

    end = time.time()
    elapsed = round(end - start, 2)
    logging.info(f"Total runtime: {elapsed} seconds")


if __name__ == "__main__":
    main()

