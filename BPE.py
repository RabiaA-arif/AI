import wikipediaapi

# Define a custom User-Agent
user_agent = "MyWikipediaScraper/1.0 (contact: myemail@example.com)"

# Initialize Wikipedia API with the User-Agent correctly
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent=user_agent  # Pass user_agent directly
)

# Fetch the article
page_title = "Neurodiversity"
page = wiki.page(page_title)

# Check if the page exists and extract text
if page.exists():
    with open("neurodiversity_wikipedia.txt", "w", encoding="utf-8") as f:
        f.write(page.text)
    print("Wikipedia article saved successfully!")
else:
    print("Page not found!")

#  preprocessing the data

import re

def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers like [1], [2]
    text = re.sub(r'\(.*?\)', '', text)  # Remove text inside parentheses
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = text.lower().strip()  # Convert to lowercase
    return text

# Read the saved Wikipedia article
with open("neurodiversity_wikipedia.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Clean the text
cleaned_text = clean_text(raw_text)

# Save cleaned text
with open("cleaned_neurodiversity.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print("Text preprocessing complete!")


# Train BPE Tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing

# Initialize a BPE Tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Use Whitespace-based pre-tokenization
tokenizer.pre_tokenizer = Whitespace()

# Trainer for BPE
trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"], vocab_size=5000)

# Train on cleaned Wikipedia text
tokenizer.train(files=["cleaned_neurodiversity.txt"], trainer=trainer)

# Save the tokenizer
tokenizer.save("bpe_tokenizer.json")

# print("✅ BPE tokenizer training complete! Saved as 'bpe_tokenizer.json' ")


print("BPE tokenizer training complete! Saved as 'bpe_tokenizer.json'.")

# Load the trained tokenizer
tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

# Example sentence
sentence = "Neurodiversity is an important concept in cognitive science."

# Tokenize the sentence
tokens = tokenizer.encode(sentence)

print("Tokenized Sentence:", tokens.tokens)
