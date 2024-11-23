import torch 
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)


class PositionalEmbeddings(nn.Module):
    def __init__(self,sentence_len,d_model,dropout):
        super().__init__()
        self.sentence_len=sentence_len
        self.d_model=d_model
        self.dropout=nn.Dropout(dropout)

        pe=torch.zeros(sentence_len,d_model)
        position=torch.arange(0,sentence_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalisation(nn.Module):

    def __init__(self,eps=10**-6):
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.alpha*((x-mean)/(std+self.eps))+self.beta

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Droupout(dropout)
        self.linear2=nn.Linear(d_ff,d_model)

    def forward(self,x):
        self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,d_model,h,dropout):
        super().__init__()
        self.d_model=d_model
        self.h=h
        self.dropout=nn.Dropout(dropout)
        assert d_model%h==0, "d_model is not divisible by h"

        self.d_k=d_model//h
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)

    @staticmethod 
    def attention(query,key,value,mask,dropout):
        d_k=query.shape[-1]

        #(batch,h,seq_len,d_k)-->(batch,h,seq_len,seq_len)
        attention_scores=(query@key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores=attention_scores.softmax(dim=-1) #(batch,h,seq_len,seq_len)

        if dropout is not None:
            attention_scores=dropout(attention_scores)

        return (attention_scores@value),attention_scores


    def forward(self,q,k,v,mask):
        query=self.w_q(q) #(batch,seq_len,d_model)-->(batch,seq_len,d_model)
        key=self.w_k(k) #(batch,seq_len,d_model)-->(batch,seq_len,d_model)
        value=self.w_v(v) #(batch,seq_len,d_model)-->(batch,seq_len,d_model)

        #(batch,seq_len,d_model)-->(batch,seq_len,h,d_k)-->(batch,h,seq_len,d_k)
        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores=MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        #(batch,h,seq_len,d_k)-->(batch,seq_len,h,d_k)-->(batch,seq_len,d_model)
        x=x.transpose(1,2).contiguos().view(x.shape[0],-1,self.h*self.d_k)

        #(batch,seq_len,d_model)-->(batch,seq_len,d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self,dropout):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalisation()

    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block,feed_forward_block,dropout):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList(ResidualConnection(dropout) for i in range(2))

    def forward(self,x,src_mask):
        x=self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))
        x=self.residual_connections[1](x,self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalisation()

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block,cross_attention_block,feed_forward_block,dropout):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.Module(ResidualConnection(dropout) for i in range(2))

    def forward(self,x,encoder_output,src_mask,target_mask):
        x=self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,target_mask))
        x=self.residual_connections[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connections[2](x,self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalisation()

    def forward(self,x,encoder_output,src_mask,target_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,target_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)


class Transformer(nn.Module):
    def __init__(self,encoder,decoder,src_embed,target_embed,src_pos,target_pos,projection_layer):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.target_embed=target_embed
        self.src_pos=src_pos
        self.target_pos=target_pos
        self.projection_layer=projection_layer

    def encode(self,src,src_mask):
        src=self.src_embed(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)

    def decode(self,encoder_output,src_mask,trg,trg_mask):
        trg=self.target_embed(trg)
        trg=self.target_pos(trg)
        return self.decoder(trg,encoder_output,src_mask,trg_mask)

    def project(self,x):
        self.projection_layer(x)



def build_transformer(src_vocab_size,tgt_vocab_size,src_seq_len,tgt_seq_len,d_model=512,N=6,h=8,dropout=0.1,d_ff=2048):
    #create embedding layers
    src_embed=InputEmbeddings(src_vocab_size,d_model)
    tgt_embed=InputEmbeddings(tgt_vocab_size,d_model)

    #create positional embedding
    src_pos=PositionalEmbeddings(src_seq_len,d_model,dropout)
    tgt_pos=PositionalEmbeddings(tgt_seq_len,d_model,dropout)

    #create encoder blocks
    encoder_blocks=[]
    for i in range(N):
        encoder_self_attention=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForward(d_model,d_ff,dropout)
        encoder_block=EncoderBlock(encoder_self_attention,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    
    #create decoder blocks
    decoder_blocks=[] 
    for i in range(N):
        decoder_self_attention=MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForward(d_model,d_ff,dropout)
        decoder_block=DecoderBlock(decoder_self_attention,decoder_cross_attention,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    #create encoder and decoder
    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))

    #projection
    projection_layer=ProjectionLayer(d_model,tgt_vocab_size)

    #build transformer
    transformer=Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)


    #initialise parameters
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform(p)

    
    return transformer


    











