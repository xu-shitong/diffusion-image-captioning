from PIL import Image
import pandas as pd
import copy
from torchvision.datasets import CocoCaptions
from transformers import (
  DistilBertTokenizer, DistilBertForMaskedLM, DistilBertConfig,
  CLIPProcessor, CLIPModel as CLIP, CLIPConfig
)
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import tqdm
import matplotlib.pyplot as plt
import os

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)
print("using device: ", dev)

# download pretrained model and tokenizer
def save_model_tokenizer(tokenizer_class, model_class, name):
  if tokenizer_class is not None:
    tokenizer = tokenizer_class.from_pretrained(name)
    tokenizer.save_pretrained(f"./tokenizers/{name}-local")
  if model_class is not None:
    model = model_class.from_pretrained(name)
    model.save_pretrained(f"./models/{name}-local/")

save_model_tokenizer(CLIPProcessor, CLIP, "openai/clip-vit-base-patch32")

# hyperparameters
BATCH_SIZE = 64
MAX_LENGTH = 64 # max text length
LEARNING_RATE = 5e-5
EPOCH_NUM = 1
ROUNDING_WEIGHT = 0.3 # weight of rounding term, the probability of regenerated sequence 
LOSS_FUNC = nn.functional.l1_loss

# diffusion hyperparameter
BETA_MIN = 0.0001
BETA_MAX = 0.02
STEP_TOT = 2000 # total noise adding steps
COSIN_SCHEDULE = False # if alpha sequence is scheduled in cosin instead of linear patten
SAMPLE_SIZE = 1 # number of sample steps in each diffuse sequence
X_0_PREDICTION = True # if model predicts x_0 or x_{t-1}

class DistilBertModel(nn.Module):
  def __init__(self, embedding, config=None) -> None:
    '''
    inputs:
      embedding: clip embedding module
      config
    '''
    super().__init__()

    self.model = DistilBertForMaskedLM(config).to(device)

    self.embedding = copy.deepcopy(embedding).to(device)
    projection_weight = embedding.weight.data.clone().detach().to(device)
    self.projection = nn.Linear(projection_weight.shape[1], projection_weight.shape[0])
    self.projection.weight.data = projection_weight
    self.projection.bias.data = torch.zeros(self.projection.bias.data.shape, device=device)
    self.projection.requires_grad_(False)
    self.embedding.requires_grad_(False)
    
    self.model.set_input_embeddings(nn.Sequential())
    self.model.set_output_embeddings(nn.Sequential())

  def parameters(self):
    return self.model.parameters()
  
  def forward(self, x, mask):
    '''
    input:
      x: [x_t ... x_t, image_clip, text_clip], shape: [sample_size * batch_size, seq_len + 2, dim]

    return 
      vocab_out, shape: [sample_size * batch_size, seq_len, vocab_size]
      feature_out, shape: [sample_size * batch_size, seq_len + 2, dim]
    '''
    
    x_out = self.model(x, mask)[0]    
    return self.projection(x_out[:, :-2, :]), x_out

clip_processor = CLIPProcessor.from_pretrained("./tokenizers/openai/clip-vit-base-patch32-local")
clip = CLIP.from_pretrained("./models/openai/clip-vit-base-patch32-local")

configuration = DistilBertConfig(vocab_size=clip_processor.tokenizer.vocab_size, dim=clip.projection_dim, n_heads=8)
model = DistilBertModel(clip.get_submodule("text_model.embeddings.token_embedding"), config=configuration)

# parameter only include model, no embedding layer
# trainer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
trainer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

if COSIN_SCHEDULE:
  def scheduler(t):
    s = 0.008 # smalle value prevent beta_t too small, from Improved DDPM paper
    return torch.cos(torch.pi / 2 * (t/STEP_TOT + s) / (1 + s)) ** 2
  ts = torch.arange(STEP_TOT).to(device)
  alpha_cumprod = scheduler(ts) / scheduler(torch.zeros(1, device=device))
else:
  betas = torch.hstack([torch.zeros(1), torch.linspace(BETA_MIN, BETA_MAX, STEP_TOT)]).to(device)
  alphas = 1 - betas
  alpha_cumprod = torch.cumprod(alphas[:-1], 0)
def diffuse_t(x, t):
  '''
  input:
    x_shape: [batch_size, seq_len, dim]
    t shape: [sample num] 
      NOTE: not necessary have hyperparameter sample_size number of element, to allow single diffuse generation

  return shape [sample_num * batch_size, seq_len, dim]
  '''
  batch_size, seq_len, dim = x.shape
  sample_shape = (t.numel(), *(1, ) * len(x.shape))

  noise = torch.normal(0, 1, x.shape).to(device)
  mean = torch.sqrt(alpha_cumprod[t].reshape(sample_shape)) * x 
  epsilon = noise * torch.sqrt(1 - alpha_cumprod[t]).reshape(sample_shape)
  return (mean + epsilon).reshape((t.numel() * batch_size, seq_len, dim))

def generate_diffuse_pair(x_0, t, t_next=None):
  '''
  input:
    x_0 shape: [batch_size, seq_len, dim],
    t shape: [sample_num] 
      NOTE: not necessary have hyperparameter sample_size number of element, to allow single diffuse generation
  
  return (net input, net target)
    net input shape: [sample_num * batch_size, seq_len, dim]
    net target shape: if t_next is None then [batch_size, seq_len, dim] else [sample_num * batch_size, seq_len, dim]
  '''
  if X_0_PREDICTION:
    # predict x_0
    return (diffuse_t(x_0, t), x_0)

  # predict x_{t_next}
  return (diffuse_t(x_0, t), diffuse_t(x_0, t_next))

def loss(model, x_t, x_1, x_tgt, x_0, image_clip, text_clip, mask, idx, loss_func):
  ''' 
  input: 
    model, 
    x_t, x_tgt shape: [sample_num * batch_size, seq_len, dim]
      NOTE: x_tgt only used when X_0_PREDICTION is False
    x_1, x_0 shape: [batch_size, seq_len, dim]
    image_clip, text_clip shape: [batch_size, dim]
    mask shape: [batch_size, seq_len + 2]
    idx shape: [batch_size, seq_len]
    loss_func

  return triple loss terms
  '''
  assert x_t.shape == (SAMPLE_SIZE * BATCH_SIZE, MAX_LENGTH, 512)
  assert x_1.shape == x_0.shape == (BATCH_SIZE, MAX_LENGTH, 512)
  assert image_clip.shape == text_clip.shape == (BATCH_SIZE, 512)
  assert mask.shape == (BATCH_SIZE, MAX_LENGTH + 2)
  assert idx.shape == (BATCH_SIZE, MAX_LENGTH)
  
  repeat_shape = (SAMPLE_SIZE, *(1, ) * (len(x_t.shape) - 1))
  image_clip = image_clip.unsqueeze(1) # shape [ batch_size, 1, dim]
  text_clip = text_clip.unsqueeze(1) # shape same as above

  # x_t restore loss
  x_t_prob, x_t_hidden = model(torch.hstack([x_t, image_clip.repeat(repeat_shape), text_clip.repeat(repeat_shape)]), mask.repeat((SAMPLE_SIZE, 1)))
  if X_0_PREDICTION:
    x_t_loss = loss_func(x_t_hidden[:, :-2, :], x_0.repeat(repeat_shape))
  else:
    assert x_tgt.shape == x_t.shape
    x_t_loss = loss_func(x_t_hidden[:, :-2, :], x_tgt)

  # x_1 restore loss
  x_1_prob, x_1_hidden = model(torch.hstack([x_1, image_clip, text_clip]), mask)
  x_1_loss = loss_func(x_1_hidden[:, :-2, :], x_0)

  # output sequence probability loss, applied to both x_1 and x_t restore
  idx = idx.unsqueeze(dim=-1)
  x_t_prob_loss = -(nn.functional.softmax(x_t_prob, dim=-1)).gather(-1, idx.repeat(repeat_shape)).log().mean()
  x_1_prob_loss = -(nn.functional.softmax(x_1_prob, dim=-1)).gather(-1, idx).log().mean()
  
  return x_t_loss, x_1_loss, ROUNDING_WEIGHT * (x_t_prob_loss + x_1_prob_loss)


class Flickr8kCLIPDataset(torch.utils.data.Dataset):
  def __init__(self, clip_processor) -> None:
    self.caption = pd.read_csv("captions.txt")
    self.tokenizer = clip_processor

    self.train_dataset = torch.utils.data.TensorDataset(torch.load("image_all_final.pickle"), torch.load("text_all_final.pickle"))

  def __len__(self):
    return len(self.caption)

  def __getitem__(self, idx):
    image_clip, text_clip = self.train_dataset[idx]
    tokens = self.tokenizer(text=self.caption.loc[idx]["caption"], images=None, return_tensors="pt", padding='max_length', truncation=True, max_length=MAX_LENGTH)

    return {
      "image_clip": image_clip, 
      "text_clip": text_clip, 
      "input_ids": tokens["input_ids"].squeeze().to(device), 
      "attention_mask": tokens["attention_mask"].squeeze().to(device)
    }

# TODO: COCO dataset

train_dataset = Flickr8kCLIPDataset(clip_processor)
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE)


# training
# model = torch.load("model_continue1.pickle")["net"]
# trainer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
model.train()
print("start training")
for epoch in range(EPOCH_NUM):
  acc_loss = 0
  with tqdm.tqdm(train_loader, unit="batch") as tepoch: 
    for sample_num, x in enumerate(tepoch):
  # for x in train_loader:
      x_0 = model.embedding(x["input_ids"])
      repeat_shape = (SAMPLE_SIZE, *(1, ) * (len(x_0.shape) - 1))
      t = torch.randint(0, STEP_TOT, repeat_shape, device=device)
      if X_0_PREDICTION:
        x_t = diffuse_t(x_0, t)
        x_tgt = None
      else:
        x_t, x_tgt = generate_diffuse_pair(x_0, t, torch.max(t - 30, torch.zeros(t.shape, device=device, dtype=torch.int64)))
      x_1 = diffuse_t(x_0, torch.ones(1, dtype=torch.int64, device=device))

      trainer.zero_grad()
      x_t_loss, x_1_loss, prob_loss = loss(
        model, 
        x_t, x_1, x_tgt, x_0, 
        x["image_clip"], x["text_clip"], 
        torch.hstack([x["attention_mask"], torch.ones((BATCH_SIZE, 2), device=device)]), 
        x["input_ids"], 
        LOSS_FUNC
      )
      l = x_t_loss + x_1_loss + prob_loss
      l.backward()
      trainer.step()

      acc_loss += l

      tepoch.set_description(f"sample {sample_num}")
      tepoch.set_postfix(
                         x_t_hidden=x_t_loss.item(),
                         x_1_loss=x_1_loss.item(),
                         prob_loss=prob_loss.item(),
                         tot_loss=l.item())
      break

  print(f"epoch {epoch} average loss: {acc_loss / len(train_loader)}, last loss x_t_loss, x_1_loss, prob_loss: {x_t_loss.item(), x_1_loss.item(), prob_loss.item()}")
  break

torch.save(model.cpu(), "model.pickle")