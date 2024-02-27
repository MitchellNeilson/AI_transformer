import torch

# Open dataset
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

# Note: using a very small codebook for simplicity. Unlike tiktoken and sentencepiece
# get all characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# encoding and decoding:
#   encode: translate string to list of integers
#   decode: translate list of integers to string
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# encode entire input text: take all text and convert to very long sequence of integers
data = torch.tensor(encode(text), dtype=torch.long)

# split the data: train + validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# looking at preceding values vs target
#   take a chunk of data [S H A K E S P E A R E]
#   - for the chunk "S", the next target is "H"
#   - for the chunk "S H", the next target is "A"
#   - etc.
#   this helps the transformer network adjust to anything as little as 1 character onwards
#   x = train_data[:block_size]
#   y = train_data[1:block_size+1]
#   for t in range(block_size):
#       context = x[:t+1]
#       target = y[t]

#   note that we want to do many chunks at hte same time so we run in batches
torch.manual_seed(1337)
batch_size = 4 # no. of independent sequences to process in parallel
block_size = 8  # max content length for predictions

def get_batch(split):
    # get small batch of inputs (x) and targets (y)
    data = train_data if split == 'train' else val_data # use training data or validation data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y


xb, yb = get_batch('train')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, : t+1]
        target = yb[b, t]
