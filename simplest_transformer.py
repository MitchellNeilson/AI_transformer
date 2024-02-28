import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy

# Open dataset
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

# Note: using a very small codebook for simplicity. Unlike tiktoken and sentencepiece
# get all characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

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
torch.manual_seed(1337) # seed for random number generation
batch_size = 4 # no. of independent sequences to process in parallel
block_size = 8  # max content length for predictions

def get_batch(split):
    # get small batch of inputs (x) and targets (y)
    data = train_data if split == 'train' else val_data # use training data or validation data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # generates (batch_size) numbers from 0 to (len(data)-block_size)
    x = torch.stack([data[i:i+block_size] for i in ix]) # Take all the 1D tensors and stacks them as rows - we generate independent rows
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y


xb, yb = get_batch('train')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, : t+1]
        target = yb[b, t]
        # print(f"when input is {context.tolist()} the target: {target}")


class BigramLanguageModel(nn.Module):

    # Constructor
    def __init__(self, vocab_size):
        super().__init__()

        # Each token directly reads off logits for next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # every integer in the input will pluck out a row in the embedding table corresponding to the index
        # pytorch then arranges it into a batch x time x channel array

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        # logits = scores for next characters in sequence
        #       Prediction basis
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            # to put into cross entropy we need to reorganize the logits (this is just how pytorch works)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  # stretch out B and T into 1D and save C as a 2nd dimension
            targets = targets.view(B*T)

            # now we need a loss function
            loss = F.cross_entropy(logits, targets)  # measures quality of the logits with respect to the targets

        return logits, loss

    def generate(self, idx, max_new_tokens): # will take a (B, T) array and generate a +1, +2, ... +max_new_tokens
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # gets the predictions
            logits, loss = self(idx) # --> take current indices and make the predictions
            # focus only on the last time step
            logits = logits[:, -1, :] # --> only need to focus on the last (most recent) step - we just need the ones for what comes next
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # this will be a (B, 1) matrix because we only asked for 1 sample
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # integers just get concatenated

        return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)

# idx = torch.zeros((1,1), dtype=torch.long) # the reason we use zeros is because 0 is the newline character so it's reasonable to use
# print(decode(m.generate(idx=idx, max_new_tokens=100)[0].tolist())) # 0th row to take out the single batch dimension that exists - gives a 1D array of all indices we will decode into text

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) # will take the gradients and update the parameters based on the gradients

batch_size = 32
for steps in range(50000):

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx=idx, max_new_tokens=100)[0].tolist()))