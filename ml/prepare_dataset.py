from datasets import load_dataset
import os


os.makedirs("data/finance", exist_ok=True)

# Download Financial PhraseBank dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")

# Create mock summaries (simplified; adapt for better quality if needed)
dataset = dataset.map(lambda x: {"text": x["sentence"], "summary": x["sentence"][:50] + "..."})

# Save to CSV
dataset["train"].to_csv("data/finance/finance_data.csv")