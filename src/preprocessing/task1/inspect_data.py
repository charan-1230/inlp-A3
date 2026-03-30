import string

def inspect_data(file_path):
    
    with open(file_path, "r") as f:
        text = f.read()
    # Find unique punctuation characters present
    punctuation_found = sorted(set([ch for ch in text if ch in string.punctuation]))
    print(f"Punctuation characters found in {file_path}: {punctuation_found}")
    
    unique_chars = sorted(set(text))
    print(f"Unique characters in {file_path}: {unique_chars}")
    
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ")
    extra = set(text) - allowed
    
    print(f"Extra characters in {file_path}: {extra}")

    print(f"Count of spaces in {file_path}: {text.count(' ')}")

def count_nines(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    return text.count("9")

plain_file_path = "data/plain.txt"

cipher_file_paths = [
    "data/cipher_00.txt",
    "data/cipher_01.txt",
    "data/cipher_02.txt",
    "data/cipher_03.txt",
    "data/cipher_04.txt"
]

if __name__ == "__main__":

    inspect_data(plain_file_path)

    for path in cipher_file_paths:
        print(f"Count of '9' in {path}: {count_nines(path)}")