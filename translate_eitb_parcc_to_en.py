import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import os

# Load the dataset
dataset = load_dataset("Helsinki-NLP/eitb_parcc", split="train")

# Load the model and tokenizer
# Model 'google/madlad400-10b-mt' is very large and may not fit in memory
# despite it has very good translation quality. For large datasets, use a smaller model.
model_name = 'google/madlad400-3b-mt'
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Function to translate text
def translate(text):
    print("ES: " + text)
    input_text = f"<2en> {text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids=input_ids, max_length=128)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("EN: " + translation)
    return translation

# Function to process batch
def process_batch(batch):
    es_texts = [item['es'] for item in batch['translation']]
    eu_texts = [item['eu'] for item in batch['translation']]
    
    translations = []
    for text in tqdm(es_texts, desc="Translating"):
        translations.append(translate(text))
    
    return {
        'es': es_texts,
        'eu': eu_texts,
        'en': translations
    }

# Set up variables
batch_size = 1000
dataset_name = "itzune/eitb_parcc_with_english"  # Replace with your desired dataset name

# Check if the target dataset exists and get the last translated row
try:
    existing_dataset = load_dataset(dataset_name, split="train")
    start_idx = len(existing_dataset)
    print(f"Continuing from index {start_idx}")
except:
    start_idx = 0
    existing_dataset = None
    print("Starting new translation")

# Process the dataset in batches
for i in range(start_idx, len(dataset), batch_size):
    print(f"\nProcessing batch starting at index {i}")
    
    # Get current batch
    end_idx = min(i + batch_size, len(dataset))
    batch = dataset[i:end_idx]
    
    # Process batch
    processed_data = process_batch(batch)
    new_batch_dataset = Dataset.from_dict(processed_data)
    
    # Combine with existing data if any
    if existing_dataset is not None:
        combined_dataset = concatenate_datasets([existing_dataset, new_batch_dataset])
    else:
        combined_dataset = new_batch_dataset
    
    # Save to Hub
    print(f"Saving dataset with {len(combined_dataset)} total rows")
    combined_dataset.push_to_hub(dataset_name, split="train")
    
    # Update existing dataset reference
    existing_dataset = combined_dataset
    
    print(f"Completed batch up to index {end_idx}")

print("\nTranslation and saving completed!")
