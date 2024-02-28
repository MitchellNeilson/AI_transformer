import torch
import torch.nn as nn
from torch.nn import functional as F

# hyper parameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ---------------

torch.manual_seed(1337)

# Open dataset
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

# Note: using a very small codebook for simplicity. Unlike tiktoken and sentencepiece
# get all characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

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

def get_batch(split):
    # get small batch of inputs (x) and targets (y)
    data = train_data if split == 'train' else val_data # use training data or validation data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # generates (batch_size) numbers from 0 to (len(data)-block_size)
    x = torch.stack([data[i:i+block_size] for i in ix]) # Take all the 1D tensors and stacks them as rows - we generate independent rows
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y


# Self attention: the idea
# ===============================================================================================
#   Think of a directed graph
#   - every node has a vector of info
#   - every node gets to aggregate information via a weighted sum from all nodes pointing to it, in a data dependent manner (what is actually stored in each node at a point in time)
#   - Node setup:
#       - The first node is only pointing to itself
#       - Second node has first node pointing to it and is pointing to itself
#       - Third node has 1 and 2 pointing to it and is pointing to itself
#       - etc. up until the 8th node
#   - In self attention:
#       - there is no notion of space: it acts over a set of vectors
#       - by default the nodes have no idea of where they are in space - this is why we need to manually code the positions
#       - the nodes are just out there to communicate with one another
# Note that in our situation the elements accross the batch dimension do not commmunicate
#   - we have different pools where inside the pool the nodes communicate but there is no communication between each pool
# The structure of the graph means future tokens do not communicate with past tokens
#   - This is not a general idea. Some instances require this communication - this requires an encoder block of self attention (no zeroing out for the softmax)
#   - In decoder blocks we create a lower-triangle structure to ensure future tokens are not commmunicated with
# Self attention vs cross attention:
#   - Self attention
#       - keys, queries and values come from the same source (x). So the nodes are SELF ATTENDING
#   - Cross attention:
#       - queries produced from x, but keys and queries come from external source (nodes on the side)
# Read up on scaled dot-product attention for more
# ===============================================================================================
# SELF ATTENTION BASIS
# We COULD can use matrix multiplications we can do averages in an easy and incremental fashion
# or we could use softmax:
#       - exponentiates each element in array and divides by the sum
#       e.g.) if we have:
#       [ 1    -inf -inf   -inf]
#       [ 1     1   -inf   -inf]
#       [ 1     1     1    -inf]
#       [ 1     1     1      1 ]
#       Then exponentiation would be
#       [ e 0 0 0]                  [ 1      0    0   0  ]
#       [ e e 0 0]    div by sum    [ 0.5   0.5   0   0  ]
#       [ e e e 0]    ------->      [ 0.33 0.33 0.33  0  ]
#       [ e e e e]                  [ 0.25 0.25 0.25 0.25]
#       Which gives us the weights we need for out code
#       We can use this to weight affinities based on past (the zeros block out the future) and expected future tokens
# ===============================================================================================
# SELF ATTENTION FURTHER
# -- Every token at each position emits 2 vectors: a query and a key
#       - Query is "what are we looking for?"
#       - Key is "what do I contain?"
# -- We need to do a dot product between the keys and the queries --> this will give us the necessary weights as seen above
# -- Gives us information about a specific token
# ===============================================================================================


# Linear models won't just multiply with fixed weights (bias=false)
# k becomes (B,T,head)
# q becomes (B,T,head)
# all the queries will dot product with all the keys
#   - But before we multiply, we need to transpose the last 2 dimensions of k so it will work
#   - We essentially have a (B,T,head) @ (B,head,T) which gives a (B,T,T)
#   - the weights now is a function of keys and queries of the nodes
#   - we now provide a key-query communication. The channels provide good insight into what properties the nodes may/may not have
# when doing aggregations for the weights, we calculate a vector v that we aggregate instead of the raw x
#   - vector x has the information we need
#   - vector v has information about what the node is interested in, what it has, and what can be communicated
#   - therefore vector v is very important

class Head(nn.Module):
    ''''One head of self-attention'''''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)  # Read up on scaled dot-product attention for more
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=1) # (B,T,T)
        wei = self.dropout(wei)
        # perform weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

# MULTI HEAD ATTENTION
# Also implement multiple attentions in parallel:
class MultiHeadAttention(nn.Module):
    ''' multiple heads of self-attention in parallel '''
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

class FeedForward(nn.Module):
    ''' a simple linear layer followed by a non-linearity '''
    # All tokens do this independently
    # Once tokens have gathered the data, they will think on it individually

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed), # projection pathway back into residual
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    ''' Transformer block: communication followed by computation '''

    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension
        #n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        # incorporate layer normalization on rows
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    # Constructor
    def __init__(self):
        super().__init__()

        # Each token directly reads off logits for next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # every integer in the input will pluck out a row in the embedding table corresponding to the index
        # pytorch then arranges it into a batch x time x channel array

        # we need to take the indices (which are based on identity of tokens) and the position of the tokens in the embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # self.sa_head = Head(n_embed) # <-- this is only 1 communication channel

        # self.sa_heads = MultiHeadAttention(4, n_embed//4) # now we have 4 communication channels: 4 heads of 8-dimensional self-attention
        # self.ffwd = FeedForward(n_embed)

        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embed)

        self.lm_head = nn.Linear(n_embed, vocab_size) # linear layer required

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        # logits = scores for next characters in sequence
        #       Prediction basis
        token_embeddings = self.token_embedding_table(idx)  # (B,T,C)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embeddings + position_embeddings # x is a (B, T, C) matrix that holds both the token identities and the POSITIONS at which these tokens occur --> needed because we want the blocks to take not of the entire sequences
        # x = self.sa_heads(x) # apply one head of self-attention (B,T,C)
        # x = self.ffwd(x) # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

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
            # crop the idx to the last (block_size) tokens
            idx_cond = idx[:, -block_size:]
            # gets the predictions
            logits, loss = self(idx_cond) # --> take current indices and make the predictions
            # focus only on the last time step
            logits = logits[:, -1, :] # --> only need to focus on the last (most recent) step - we just need the ones for what comes next
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # this will be a (B, 1) matrix because we only asked for 1 sample
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # integers just get concatenated

        return idx



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # will take the gradients and update the parameters based on the gradients

# The training loop
for iter in range(max_iters):

    # every once in a while evaluate loss on training and valuation sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample batch of data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx=idx, max_new_tokens=500)[0].tolist()))
