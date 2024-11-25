import torch
import torch.nn as nn
from torch.nn import functional as F


#Load the dataset from terminal
#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#or
# curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

batch_size=64
block_size=256
max_iters=5000
eval_interval=500
learning_rate=3e-4
eval_iters=200
n_embed=30
dropout=0.2
n_layer=3
device ='cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1222)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab=sorted(list(set(text)))
vocab_size=len(vocab)
stoi={char:i for i,char in enumerate(vocab)}
itos={i:char for char,i in stoi.items()}
encoder=lambda s: [stoi[c] for c in s]
decoder=lambda l: ''.join(itos[i] for i in l)

data=torch.tensor(encoder(text),dtype=torch.long) 
break_point=int(0.9*len(data))
train_data=data[ :break_point]
val_data=data[break_point:]

def get_batch(split):
  split_data=train_data if split=='train' else val_data
  ix=torch.randint(len(split_data)-block_size,(batch_size,))
  x=torch.stack([split_data[i:i+block_size] for i in ix])
  y=torch.stack([split_data[i+1:i+block_size+1] for i in ix])
  x,y=x.to(device),y.to(device)
  return x,y


@torch.no_grad()
def estimate_loss():
    out={}
    model.eval() #modify model to evaluation layer
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train() #modify model back to train
    return out



class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embed,head_size,bias=False)
        self.query=nn.Linear(n_embed,head_size,bias=False)
        self.value=nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)

        wei=q@k.transpose(-2,-1)*(C**0.5)
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1)
        wei=self.dropout(wei)

        out=wei@v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection=nn.Linear(num_heads*head_size,n_embed)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads], dim=-1)
        out=self.dropout(self.projection(out))

        return out

class FeedForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_embed,4*n_embed),nn.ReLU(),nn.Linear(4*n_embed,n_embed),nn.Dropout(dropout))

    def forward(self,x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self,n_embed,n_head):
        super().__init__()
        head_size=n_embed//n_head
        self.sa=MultiHeadAttention(n_head,n_embed)
        self.ffw=FeedForward(n_embed)
        self.ln1=nn.LayerNorm(n_embed)
        self.ln2=nn.LayerNorm(n_embed)

    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffw(self.ln2(x))
        return x

    
class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table=nn.Embedding(vocab_size,n_embed)
    self.position_embedding_table=nn.Embedding(block_size,n_embed)
    self.linear=nn.Linear(n_embed,vocab_size)
    # self.sa_head=Head(n_embed)
    # self.ffw=FeedForward(n_embed)
    # self.sa_head=MultiHeadAttention(4,n_embed//4)
    # self.blocks=nn.Sequential(Block(n_embed,n_head=4), Block(n_embed,n_head=4), Block(n_embed,n_head=4),nn.LayerNorm(n_embed))
    self.blocks=nn.Sequential(*[Block(n_embed,n_head=4) for i in range(n_layer)])
    self.ln_f=nn.LayerNorm(n_embed)

  def forward(self,idx,targets=None):
    B,T=idx.shape
    token_embed=self.token_embedding_table(idx) #(b,t,c)
    positional_embed=self.position_embedding_table(torch.arange(T,device=device)) #(t,c)
    x=token_embed+positional_embed #(b,t,c) #understand how brodcast works in detail
    x=self.blocks(x)
    logits=self.linear(x) #(b,t,c) this c and above c in token_embed are not equal
    if targets is None:
      loss=None
    else:
      B,T,C=logits.shape
      logits=logits.view(B*T,C)
      targets=targets.view(B*T) #or (-1)
      loss=F.cross_entropy(logits,targets) #negative log likelihood

    return logits,loss

  def generate(self,idx,max_new_tokens):
    #idx is (B,T)
    for i in range(max_new_tokens):
      #get prediciton
      idx_cond=idx[:,-block_size:]
      logits,loss=self(idx_cond)
      #focus only on last time step
      logits=logits[:,-1,:] #becomes (B,C)
      probs=F.softmax(logits,dim=-1) #(B,C)
      idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
      idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)
    return idx


model=Decoder()
m=model.to(device)

optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)

for i in range(max_iters):
    if i%eval_interval==0:
        loss=estimate_loss()
        print(f"step : {i}, train_loss : {loss['train']:.4f}, val_loss : {loss['val']:.4f}")
    xb,yb=get_batch('train')
    logits,loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()




print(decoder(m.generate(torch.zeros((1,1),dtype=torch.long),max_new_tokens=1000)[0].tolist())) #[0] gives batch dimension from (B,T)