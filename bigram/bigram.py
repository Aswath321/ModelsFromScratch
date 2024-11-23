import torch
import torch.nn as nn
from torch.nn import functional as F


#Load the dataset from terminal
#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#or
# curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

batch_size=32
block_size=8
max_iters=100000
eval_interval=300
learning_rate=1e-2
eval_iters=200
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

class BigramLanguageModel(nn.Module):
  def __init__(self,vocab_size):
    super().__init__()
    self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)

  def forward(self,idx,targets=None):
    # print(idx.shape)
    logits=self.token_embedding_table(idx) #B,T,C Batch,Time(Block_size),channels(vocab_size)
    # print(logits.shape)
    #according to documentation pytorch wants logits in the form b,c,t and not b,t,c
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
      logits,loss=self(idx)
      #focus only on last time step
      logits=logits[:,-1,:] #becomes (B,C)
      probs=F.softmax(logits,dim=-1) #(B,C)
      idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
      idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)
    return idx


model=BigramLanguageModel(vocab_size)
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




print(decoder(m.generate(torch.zeros((1,1),dtype=torch.long),max_new_tokens=100)[0].tolist())) #[0] gives batch dimension from (B,T)