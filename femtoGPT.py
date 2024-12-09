import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
  def __init__(self, embed_dim, num_heads, causal=True):
    super().__init__()
    self.W_qkv = nn.Parameter(torch.randn(embed_dim, 3 * embed_dim) * .02)
    self.W_out = nn.Parameter(torch.randn(embed_dim, embed_dim) * .02)

    self.num_heads = num_heads
    self.causal = causal

  def forward(self, x):
    B, T, C = x.shape
    qkv = x @ self.W_qkv  # BTC @ C(H*C) => BT(C*H)
    Q, K, V = qkv.chunk(3, dim=-1)

    Q = Q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # => BHTC
    K = K.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
    V = V.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

    attention = Q @ K.transpose(-2, -1) / C ** .5

    if self.causal:
      t = torch.arange(T, device=x.device)
      mask = t[None, None, :, None] < t[None, None, None, :]
      attention.masked_fill_(mask, -torch.inf)

    attention = attention.softmax(dim=-1)

    y = attention @ V
    y = y.transpose(1, 2).reshape(B, T, C)

    return y @ self.W_out


class FFN(nn.Sequential):
  def __init__(self, in_channels, inner_channels):
    super().__init__(
      nn.Linear(in_channels, inner_channels),
      nn.ReLU(),
      nn.Linear(inner_channels, in_channels),
    )


class Residual(nn.Module):
  def __init__(self, *modules):
    super().__init__()
    self.module = nn.Sequential(*modules)

  def forward(self, x):
    return self.module(x) + x


class GPT(nn.Module):
  def __init__(
      self,
      num_layers=6,
      context_length=1024,
      embed_dim=512,
      num_heads=8,
      ffn_dim=2048,
      vocab_size=20_000,
      norm=nn.RMSNorm,
      ffn=FFN
  ):
    super().__init__()
    self.embeddings = nn.Embedding(vocab_size, embed_dim)
    self.pos_embeddings = nn.Embedding(context_length, embed_dim)
    
    self.blocks = nn.Sequential(
      *(nn.Sequential(
        Residual(
          norm(embed_dim),
          Attention(embed_dim=embed_dim, num_heads=num_heads),
        ),
        Residual(
          norm(embed_dim),
          ffn(in_channels=embed_dim, inner_channels=ffn_dim),
        )
      ) for i in range(num_layers))
    )

    self.classifier = nn.Linear(embed_dim, vocab_size)

  def forward(self, x):
    x = self.embeddings(x) + self.pos_embeddings(x)
    x = self.blocks(x)
    return self.classifier(x)


class Tokenizer:
  def __init__(self, text):
    self.vocab = sorted(set(text))
    self.indices = {c: n for n, c in enumerate(self.vocab)}

  def encode(self, text):
    return [self.indices[c] for c in text]

  def decode(self, indices):
    return [self.vocab[i] for i in indices]


def train(data=None, *, device='mps', context_length=8, batch_size=8, epochs=10, **kwargs):
  tokenizer = Tokenizer(data)

  model = GPT(**kwargs, vocab_size=len(tokenizer.vocab))
  model.to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

  indices = torch.tensor(tokenizer.encode(data), device=device)
  samples = indices[: context_length * (len(indices) // context_length)].view(-1, context_length)

  for epoch in range(epochs):
    print(f"Epoch {epoch}")
    model.train()
    for batch in samples.split(batch_size):
      logits = model(batch[:, :-1])
      loss = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        batch[:, 1:].reshape(-1)
      )

      with torch.no_grad():
        preds = logits.argmax(-1).view(-1).cpu().tolist()
        print(f"\n\n" + '=' * 200 + '\n\n' + ''.join(list(tokenizer.decode(preds))))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


if __name__ == '__main__':
  text = open(__file__, 'r').read()
  train(
    text,
    embed_dim=128,
    num_heads=8,
    ffn_dim=256,
    context_length=32,
    epochs=100
  )
