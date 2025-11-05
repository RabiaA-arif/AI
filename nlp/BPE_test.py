from tokenizers import Tokenizer

# Load the trained BPE tokenizer
tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

# Test sentences
test_sentences = [
    "Neurodiversity is an important concept.",
    "Cognitive science explores human intelligence.",
    "People with autism have unique strengths."
]

# Tokenize and display results
for sentence in test_sentences:
    tokens = tokenizer.encode(sentence)
    print(f"Original: {sentence}")
    print(f"Tokenized: {tokens.tokens}")
    print("-" * 50)



#  evaluation of model

def compression_ratio(text, tokenizer):
    original_length = len(text.split())  # Word-based token count
    bpe_length = len(tokenizer.encode(text).tokens)  # BPE token count
    return bpe_length / original_length  # Ratio (lower is better)

# test_text = "Neurodiversity is a concept that embraces neurological differences."


test_text = "Neurodiversity is a concept that embraces neurological differences."
tst_text="neurodivergent and neurotypicalneuroconformingaccording to kassiane asasumasu who coined the terms in the year  neurodivergentneurodivergence"

ratio = compression_ratio(test_text, tokenizer)

print(f"Compression Ratio: {ratio:.2f}")

ratio = compression_ratio(tst_text, tokenizer)
print(f"Compression Ratio: {ratio:.2f}")