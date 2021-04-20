import random

with open('Frankenstein.txt', 'r', encoding='utf-8') as f:
    text = f.read()

n = 3
ngrams = {}
def n_gram(n, text):
    for i in range(len(text) - n):
        seq = text[i:i+n]
        seq = ' '.join(seq)
        if seq not in ngrams.keys():
            ngrams[seq] = []
        ngrams[seq].append(text[i+n])
    return ngrams

splited_text = text.split()
ngrams = n_gram(n, splited_text)

curr_sequence = splited_text[0:n]
curr_sequence = ' '.join(curr_sequence)
output = curr_sequence
for i in range(200):
    if curr_sequence not in ngrams.keys():
        break
    possible_chars = ngrams[curr_sequence]
    next_char = possible_chars[random.randrange(len(possible_chars))]
    output += ' ' + next_char
    temp = output.split()
    curr_sequence = temp[len(temp)-n:len(temp)]
    curr_sequence = ' '.join(curr_sequence)

print(output)
