import json
from transformers import pipeline

# 1. Load a pre-trained model for initial probing
# text-generation model to see how it performs
# textgen_ml = pipeline("text-generation", model="meta-llama/Llama-3-8b-chat-hf")
textgen_mt = pipeline("text-generation", model="mistral/mixtral-8x7b-instruct")

# 2. Load raw, domain-specific data
with open("data/PII.json", 'r') as file:
    df = json.load(file)

# Loop through all rows in the data
for idx, source_text in enumerate(df):
    # ml_output = textgen_ml(source_text, max_length=50)[0]['generated_text']
    mt_output = textgen_mt(source_text, max_length=50)[0]['generated_text']

    print(f"\n=== ROW {idx} ===")
    print("--- DOMAIN-SPECIFIC TEXT ---")
    print(source_text)
    print("\n--- Llama-3 GENERATED TEXT ---")
    print(ml_output)
    print("\n--- Mixtral GENERATED TEXT ---")
    print(mt_output)

# 4. Analysis & Observations (Markdown cells in the notebook)
# Think about the following observations:
# - **Observation 1: Metric-Goal Misalignment.** The generic summary is factually correct but misses the target audience. It's still too technical. This highlights the "Metric-Goal Misalignment" issue from our planning[cite: 20].
# - **Observation 2: Omission of Real-World Scenarios.** The model wasn't trained on "technical-to-business" translation, so it omits the key context of who the summary is for[cite: 18].
# - **Observation 3: Unresolved Limitations.** Even a powerful model like BART has limitations when applied to a new domain without fine-tuning[cite: 16, 17].

