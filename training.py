import torch
import torch.nn as nn
import mmap
import random
import pickle
from torch.nn import functional as F
import argparse

parser = argparse.ArgumentParser(description='This is a demonstration program')
parser.add_argument('-batch_size', type = str, required=True, help='Please provide a batch_size')
args = parser.parse_args()
print(f'batch_size:{args.batch_size}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
block_size = args.batch_size
batch_size = 128
max_iters = 1000
#eval_interval = 2500
learning_rate = 3e-4
eval_iters = 100
dropout = 0.2 #arunca noduri random in network pentru a evita overfilling
n_embed = 384
n_layer = 1
n_head = 1

chars = ""
with open("vocab.txt", 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
        
vocab_size = len(chars)

string_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_string = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

encoded_txt = encode('hello')
decoded = decode(encoded_txt)

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]

def get_random_chunk(split):
    filename = "train_split.txt" if split == 'train' else "val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0,  (file_size) - block_size*batch_size)
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size - 1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r','')
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
    return data

def get_batch(split):
    data  = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    #print(ix)
    x=torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
x, y = get_batch('train')
print('input=')
print(x)
print('target')
print(y)

@torch.no_grad()# tot ce se intampla aici sa nu ia seama la gradienti
def estimate_loss():
    out = {}
    model.eval() # trecem in modul de evaluare
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters): #ia un batch trece prin model, retine pierderile
            x, y = get_batch(split)
            logits, loss = model.forward(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean() # media pierderilor
    model.train()# pune modelul in modul de antrenare
    return out

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print("when input is ", context, "target is", target)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False) # transforma n_embed to head_size
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # ne asiguram c a programul nu triseaza cand se uita inainte
        wei = F.softmax(wei, dim = -1) # transformam in probabilitati
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # 4 heads in paralel
        self.proj = nn.Linear(head_size * num_heads, n_embed) 
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) #concatenam fiecare head -> (B, T, F)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
     def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
     def forward(self,x):
        return self.net(x)

class Block(nn.Module):
     def __init__(self, n_embed, n_head):
         super().__init__()
         head_size = n_embed // n_head
         self.sa = MultiHeadAttention(n_head, head_size)
         self.ffwd = FeedForward(n_embed)
         self.ln1 = nn.LayerNorm(n_embed)
         self.ln2 = nn.LayerNorm(n_embed)


     def forward(self,x):
         y = self.sa(x)
         x = self.ln1(x+y)
         y = self.ffwd(x)
         x = self.ln2(x+y)
         return x

class GPTLanguageModel(nn.Module):
     def __init__(self, vocab_size):
         super().__init__()
         self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
         self.position_embedding_table = nn.Embedding(block_size, n_embed)
         #se initializeaza un tabel unde pt fiecare element din vocab se retin toate probabilitatile
         # ca urmatorul element sa fie alta litera
         self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)]) 
         self.in_f = nn.LayerNorm(n_embed) # final layer norm
         self.in_head = nn.Linear(n_embed, vocab_size)
         self.apply(self.__init__weights)


     def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
        
     def forward(self, index, targets=None):
         B, T = index.shape
         tok_emb = self.token_embedding_table(index)
         pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # un lookup table
         x = tok_emb + pos_emb
         x = self.blocks(x)
         x = self.in_f(x)
         logits = self.in_head(x)
         if targets is None:
             loss = None
         else:
             B, T, C = logits.shape#Batch, Time(lungimea secventei), Channels(vocab_size)
             logits = logits.view(B*T, C)
             targets = targets.view(B*T) #se faec sa se potriveasca pentru inmultire
             loss = F.cross_entropy(logits, targets)# cat de gresit s-a ghicit
         return logits, loss

     def generate(self, index, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling (better quality than pure sampling)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index
model = GPTLanguageModel()
print('loading model parameters...')
#with open('model-01.pkl', 'rb') as f:
 #     model = pickle.load(f)
#print('loaded successfully')
m = model.to(device)


context = torch.zeros((1,1), dtype=torch.long, device = device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)  

optimizer = torch.optim.AdamW(model.parameters(),  lr = learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f'step:{iter}, train loss:{losses['train']:.2f}, val loss : {losses['val']:.2f}') 
    xb, yb = get_batch('train')
    logits, loss = model.forward(xb, yb)#se face predictia si compara cu yb ca sa calculeze cat de gresit a fost
    optimizer.zero_grad(set_to_none=True)
    loss.backward()#se calculeaza gradientul astfel incat sa vedem cum loss ul sa scada
    optimizer.step()#actualizam weights
print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model saved')

