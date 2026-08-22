with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Total characters: {len(text)}")
print("Sample text preview:")
print(text[:100])