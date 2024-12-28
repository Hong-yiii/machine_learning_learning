import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Attention Model
class Head(nn.Module): 
    # one head of self attention

    def __init__(self, head_size): 
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        #since it is not a param, we assign it using register_buffer
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) #(B,T,C)
        q = self.query(x) #(B,T,C)

        # computing the affinities
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class FeedForward(nn.Module): 
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # implementing pre-norm formulation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Addition for gradients to flow through easier using residual connections
        x = x + self.ffwd(self.ln2(x))
        return x

# microGPT Model
class microGPT(nn.Module):

    def __init__(self, n_embd): 
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #n_embed is the number of embedding dimensions 
        # we also want to embed in the positions of token
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #embed from 0 to blksize - 1
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final Layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # language model head, maps back to vocab space
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):   
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) matrix
        x = tok_emb + pos_emb # note of broadcasting across Batch dim
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Set the model to evaluation mode and disable gradients
        self.eval()
        with torch.no_grad():
            for _ in tqdm(range(max_new_tokens), desc="Generating Tokens"):
                # Crop idx into last block_size tokens
                idx_cond = idx[:, -block_size:]
                
                # Get the predictions
                logits, _ = self(idx_cond)
                
                # Focus only on the last time step
                logits = logits[:, -1, :]  # Shape: (B, C)
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)  # Shape: (B, C)
                
                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # Shape: (B, 1)
                
                # Append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1)  # Shape: (B, T+1)
        return idx

model = microGPT(n_embd=384)
m = model.to(device)
# Print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("CUDA is available. Using GPU.")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")



# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

from tqdm import tqdm
import matplotlib.pyplot as plt

# Initialize lists to store loss values
train_losses = []
val_losses = []
iterations = []

# Create a tqdm progress bar
with tqdm(range(max_iters), desc="Training", unit="iter") as pbar:
    for iter in pbar:

        # Every eval_interval steps, evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            train_loss = losses['train']
            val_loss = losses['val']
            
            # Append losses to the lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            iterations.append(iter)

            # Update the progress bar description
            pbar.set_postfix({
                "Step": iter,
                "Train Loss": f"{train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}"
            })

        # Sample a batch of data
        xb, yb = get_batch('train')

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

# After training, plot the losses
plt.figure(figsize=(10, 5))
plt.plot(iterations, train_losses, label='Train Loss')
plt.plot(iterations, val_losses, label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.savefig('training_plot.png')  # Save the plot to a file
plt.close()  # Close the plot to free up resources
print("Training plot saved as 'training_plot.png'.")

# save the model 
# Save the trained model's state_dict
torch.save(model.state_dict(), 'microGPT_model.pth')
print("Model saved as 'microGPT_model.pth'.")


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))