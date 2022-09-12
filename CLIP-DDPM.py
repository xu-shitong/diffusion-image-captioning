
"""# Imports"""

from PIL import Image
import pandas as pd
import copy
from torchvision.datasets import CocoCaptions
from transformers import (
  DistilBertTokenizer, DistilBertForMaskedLM, DistilBertConfig,
  CLIPProcessor, CLIPModel as CLIP, CLIPConfig,
  activations, PreTrainedTokenizer
)
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import tqdm
import math

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)
print("using device: ", dev)

# Import packages
import os,sys,humanize,psutil,GPUtil

# Define function
def mem_report():
  print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
  
  GPUs = GPUtil.getGPUs()
  for i, gpu in enumerate(GPUs):
    print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

mem_report()

# # download pretrained model and tokenizer
# def save_model_tokenizer(tokenizer_class, model_class, name):
#   if tokenizer_class is not None:
#     tokenizer = tokenizer_class.from_pretrained(name)
#     tokenizer.save_pretrained(f"./tokenizers/{name}-local")
#   if model_class is not None:
#     model = model_class.from_pretrained(name)
#     model.save_pretrained(f"./models/{name}-local/")

# # save_model_tokenizer(CLIPProcessor, CLIP, "openai/clip-vit-base-patch32")
# save_model_tokenizer(DistilBertTokenizer, DistilBertForMaskedLM, "distilbert-base-uncased")

"""# Hyperparameters"""

def series_sum_batch_average(x_hat, x):
  return (x_hat - x).abs().sum(dim=1).mean()

# hyperparameters
BATCH_SIZE = 8
MAX_LENGTH = 16 # max text length
LEARNING_RATE = 5e-5
EPOCH_NUM = 12
ROUNDING_WEIGHT = 3e-1 # weight of rounding term, the probability of regenerated sequence 
# LOSS_FUNC = nn.functional.l1_loss
LOSS_FUNC = series_sum_batch_average # loss function used between embedding 
# CLIP_ADDING_METHOD = "add" # CLIP feature are added as position embedding to sequence of word embedding
CLIP_ADDING_METHOD = "concat" # CLIP feature are appended to sequence of word embedding, use together with CLIP_MASK
CLIP_MASK = torch.tensor([1, 1], device=device) # mask indicating if [image, text] clip feature is used 
TRAIN_EMBEDDING = False # if model use pretrained distilbert embedding, or learn a 16 embedding for each word and project to 768 before pass to bert
if TRAIN_EMBEDDING:
  IN_CHANNEL = 16
else:
  IN_CHANNEL = 768

# diffusion hyperparameter
BETA_MIN = 0.0001
BETA_MAX = 0.02
STEP_TOT = 1000 # total noise adding steps
COSIN_SCHEDULE = True # if alpha sequence is scheduled in cosin instead of linear patten
SAMPLE_SIZE = 100 # number of sample steps in each diffuse sequence
X_0_PREDICTION = True # if model predicts x_0 or x_{t-1}

MODEL_NAME = f"batch{BATCH_SIZE}_maxlen{MAX_LENGTH}_round{'%.0E' % ROUNDING_WEIGHT}_loss{LOSS_FUNC.__name__}_clip{CLIP_ADDING_METHOD}_clipmask{CLIP_MASK[0].item()}{CLIP_MASK[1].item()}_train-embed{TRAIN_EMBEDDING}_samplesize{SAMPLE_SIZE}"

"""# Define Dataset"""

# # TODO: use self defined tokenizer
# from spacy.lang.en import English
# from collections import Counter
# import itertools

# captions = pd.read_csv("captions.txt")["caption"]
# nlp = English()

# sentence_lst = []

# for sentences in captions:
#   word_lst = [x.text.lower() for x in nlp.tokenizer(sentences)]
#   spl = [[]]
#   for x, y in itertools.groupby(word_lst, lambda z: z == '.'):
#       spl[-1].extend(y)
#       if x: spl.append([])
#   sentence_lst.extend(spl[:-1])

# counter = Counter()
# for input_ids in sentence_lst:
#     counter.update(input_ids)
# vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}
# for k, v in counter.items():
#     if v > 10:
      # vocab_dict[k] = len(vocab_dict)

class Flickr8kCLIPDataset(torch.utils.data.Dataset):
  def __init__(self, captions, tokenizer) -> None:
    self.caption = captions
    self.tokenizer = tokenizer

    self.train_dataset = torch.utils.data.TensorDataset(torch.load("image_all_final.pickle").to(device), torch.load("text_all_final.pickle").to(device))

  def __len__(self):
    return len(self.caption)

  def __getitem__(self, idx):
    image_clip, text_clip = self.train_dataset[idx]
    if isinstance(self.tokenizer, PreTrainedTokenizer):
      tokens = self.tokenizer(text=self.caption.loc[idx], return_tensors="pt", padding='max_length', truncation=True, max_length=MAX_LENGTH)
    else:
      ids = [0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in self.caption.loc[idx][:MAX_LENGTH-2]] + [1] 
      pad_length = max(0, MAX_LENGTH - len(ids))
      tokens = dict()
      tokens["input_ids"] = torch.tensor(ids + [vocab_dict['UNK']] * pad_length)
      tokens["attention_mask"] = torch.tensor([1] * len(ids) + [0] * pad_length)

    return {
      "image_clip": image_clip, 
      "text_clip": text_clip, 
      "input_ids": tokens["input_ids"].squeeze().to(device), 
      "attention_mask": tokens["attention_mask"].squeeze().to(device)
    }

# TODO: COCO dataset

if TRAIN_EMBEDDING:
  tokenizer = vocab_dict
  VOCAB_SIZE = len(vocab_dict)
else:
  tokenizer = DistilBertTokenizer.from_pretrained("./tokenizers/distilbert-base-uncased-local/", local_files_only=True)
  VOCAB_SIZE = tokenizer.vocab_size

train_dataset = Flickr8kCLIPDataset(pd.read_csv("captions.txt")["caption"], tokenizer)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

mem_report()

"""# Model, trainer and loss function"""

class DistilBertModel(nn.Module):
  def __init__(self, embedding=None, projection=None, config=None) -> None:
    '''
    inputs:
      embedding: clip embedding module
      config
    '''
    super().__init__()

    self.model = DistilBertForMaskedLM(config).to(device)

    if TRAIN_EMBEDDING:
      self.embedding = nn.Embedding(VOCAB_SIZE, IN_CHANNEL, device=device).requires_grad_(True)
      self.lm_head = nn.Linear(IN_CHANNEL, VOCAB_SIZE, bias=False, device=device).requires_grad_(True)

      self.input_projection = nn.Linear(IN_CHANNEL, 768, device=device).requires_grad_(True)
      self.output_projection = nn.Linear(768, IN_CHANNEL, device=device).requires_grad_(True)
    else:
      self.embedding = copy.deepcopy(embedding.requires_grad_(False))
      self.lm_head = copy.deepcopy(projection.requires_grad_(False))
      self.lm_head.bias.data = torch.zeros(self.lm_head.bias.data.shape, device=device).requires_grad_(False)
    
    self.model.set_input_embeddings(nn.Sequential())
    self.model.set_output_embeddings(nn.Sequential())

    self.image_linear = nn.Linear(512, 768, device=device)
    self.text_linear = nn.Linear(512, 768, device=device)

    if CLIP_ADDING_METHOD == "concat":
      self.segment_embedding = nn.Embedding(2, 768, device=device)

  def parameters(self):
    base_list = list(self.model.parameters()) + list(self.image_linear.parameters()) + list(self.text_linear.parameters())
    if TRAIN_EMBEDDING:
      base_list += list(self.embedding.parameters()) + list(self.lm_head.parameters()) \
                  + list(self.input_projection.parameters()) + list(self.output_projection.parameters())

    if CLIP_ADDING_METHOD == "concat":
      return base_list + list(self.segment_embedding.parameters())
    elif CLIP_ADDING_METHOD == "add":
      return base_list
    else:
      raise NotImplementedError(CLIP_ADDING_METHOD)

  def forward(self, x, image_clip, text_clip, mask):
    '''
    input:
      x: [x_t ... x_t], shape: [sample_size * batch_size, seq_len, IN_CHANNEL]
      image_clip, text_clip shape: [sample_size * batch_size, 1, clip_dim]
      mask shape: [sample_size * batch_size, seq_len] 
    
    return 
      vocab_out, shape: [sample_size * batch_size, seq_len, vocab_size]
      feature_out, shape: [sample_size * batch_size, seq_len, IN_CHANNEL]
    '''
    sample_batch_multi, _, _ = x.shape

    assert x.shape == (sample_batch_multi, MAX_LENGTH, IN_CHANNEL)
    assert image_clip.shape == text_clip.shape == (sample_batch_multi, 1, 512)
    assert mask.shape == (sample_batch_multi, MAX_LENGTH)

    if TRAIN_EMBEDDING:
      x = self.input_projection(x)
    
    if CLIP_ADDING_METHOD == "concat":
      mask = torch.hstack([mask, CLIP_MASK.repeat((mask.shape[0], 1))])
      x = torch.hstack([x, self.image_linear(image_clip), self.text_linear(text_clip)])
      x = x + self.segment_embedding(torch.tensor([0] * MAX_LENGTH + [1] * 2, device=device))
    elif CLIP_ADDING_METHOD == "add":
      x = x + self.image_linear(image_clip) + self.text_linear(text_clip)
    else:
      raise NotImplementedError(CLIP_ADDING_METHOD)

    x_out = self.model(x, mask)[0]
    if TRAIN_EMBEDDING:
      x_out = self.output_projection(x_out)

    assert x_out.shape == (sample_batch_multi, mask.shape[-1], IN_CHANNEL)
    return self.lm_head(x_out[:, :MAX_LENGTH, :]), x_out
    
if TRAIN_EMBEDDING:
  configuration = DistilBertConfig()
  model = DistilBertModel(config=configuration)
else:
  origin = DistilBertForMaskedLM.from_pretrained("./models/distilbert-base-uncased-local", local_files_only=True).to(device)
  configuration = DistilBertConfig()
  model = DistilBertModel(origin.get_input_embeddings(), origin.get_output_embeddings(), config=configuration)

# parameter only include model, no embedding layer
# trainer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
trainer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

if COSIN_SCHEDULE:
  def scheduler(t):
    s = 0.008 # smalle value prevent beta_t too small, from Improved DDPM paper
    return torch.cos(math.pi / 2 * (t/STEP_TOT + s) / (1 + s)) ** 2
  ts = torch.arange(STEP_TOT).to(device)
  alpha_cumprod = scheduler(ts) / scheduler(torch.zeros(1, device=device))
else:
  betas = torch.hstack([torch.zeros(1), torch.linspace(BETA_MIN, BETA_MAX, STEP_TOT)]).to(device)
  alphas = 1 - betas
  alpha_cumprod = torch.cumprod(alphas[:-1], 0)
def diffuse_t(x, t):
  '''
  input:
    x_shape: [batch_size, seq_len, IN_CHANNEL]
    t shape: [sample num] 
      NOTE: not necessary have hyperparameter sample_size number of element, to allow single diffuse generation

  return shape [sample_num * batch_size, seq_len, IN_CHANNEL]
  '''
  batch_size, seq_len, _ = x.shape
  sample_shape = (t.numel(), *(1, ) * len(x.shape))

  noise = torch.normal(0, 1, x.shape).to(device)
  mean = torch.sqrt(alpha_cumprod[t].reshape(sample_shape)) * x 
  epsilon = noise * torch.sqrt(1 - alpha_cumprod[t]).reshape(sample_shape)
  return (mean + epsilon).reshape((t.numel() * batch_size, seq_len, IN_CHANNEL))

def generate_diffuse_pair(x_0, t, t_next=None):
  '''
  input:
    x_0 shape: [batch_size, seq_len, IN_CHANNEL],
    t shape: [sample_num] 
      NOTE: not necessary have hyperparameter sample_size number of element, to allow single diffuse generation
  
  return (net input, net target)
    net input shape: [sample_num * batch_size, seq_len, IN_CHANNEL]
    net target shape: if t_next is None then [batch_size, seq_len, IN_CHANNEL] else [sample_num * batch_size, seq_len, IN_CHANNEL]
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
    x_t, x_tgt shape: [sample_num * batch_size, seq_len, IN_CHANNEL]
      NOTE: x_tgt only used when X_0_PREDICTION is False
    x_1, x_0 shape: [batch_size, seq_len, IN_CHANNEL]
    image_clip, text_clip shape: [batch_size, clip_dim]
    mask shape: [batch_size, seq_len]
    idx shape: [batch_size, seq_len]
    loss_func

  return triple loss terms
  '''
  assert x_t.shape == (SAMPLE_SIZE * BATCH_SIZE, MAX_LENGTH, IN_CHANNEL)
  assert x_1.shape == x_0.shape == (BATCH_SIZE, MAX_LENGTH, IN_CHANNEL)
  assert image_clip.shape == text_clip.shape == (BATCH_SIZE, 512)
  assert mask.shape == (BATCH_SIZE, MAX_LENGTH)
  assert idx.shape == (BATCH_SIZE, MAX_LENGTH)
  
  repeat_shape = (SAMPLE_SIZE, *(1, ) * (len(x_t.shape) - 1))
  image_clip = image_clip.unsqueeze(1) # shape [ batch_size, 1, clip_dim]
  text_clip = text_clip.unsqueeze(1) # shape same as above

  # x_t restore loss
  x_t_prob, x_t_hidden = model(x_t, image_clip.repeat(repeat_shape), text_clip.repeat(repeat_shape), mask.repeat((SAMPLE_SIZE, 1)))
  if X_0_PREDICTION:
    x_t_loss = loss_func(x_t_hidden[:, :MAX_LENGTH, :], x_0.repeat(repeat_shape))
  else:
    assert x_tgt.shape == x_t.shape
    x_t_loss = loss_func(x_t_hidden[:, :MAX_LENGTH, :], x_tgt)

  # x_1 restore loss
  x_1_prob, x_1_hidden = model(x_1, image_clip, text_clip, mask)
  x_1_loss = loss_func(x_1_hidden[:, :MAX_LENGTH, :], x_0)

  # output sequence probability loss, applied to both x_1 and x_t restore
  idx = idx.unsqueeze(dim=-1)
  x_t_prob_loss = -(nn.functional.softmax(x_t_prob, dim=-1)).gather(-1, idx.repeat(repeat_shape)).log().sum(dim=1).mean()
  x_1_prob_loss = -(nn.functional.softmax(x_1_prob, dim=-1)).gather(-1, idx).log().sum(dim=1).mean()
  
  return x_t_loss, x_1_loss, ROUNDING_WEIGHT * (x_t_prob_loss + x_1_prob_loss)

mem_report()

"""# Training"""

# training 
# model = torch.load("batch8_maxlen16_round3E-01_lossseries_sum_batch_average_clipconcat_clipmask10_train-embedFalse_samplesize100.pickle").to(device)
# model.model.add_module("activation", activations.GELUActivation())
# trainer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
summary = open("summary.txt", "a")

model.train()
print("start training")
for epoch in range(EPOCH_NUM):
  acc_x_t = 0
  acc_x_1 = 0
  acc_prob = 0
  # with tqdm.tqdm(train_loader, unit="batch") as tepoch: 
  #   for batch_num, x in enumerate(tepoch):
  for batch_num, x in enumerate(train_loader):

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
        x["attention_mask"], 
        x["input_ids"], 
        LOSS_FUNC
      )
      
      l = x_t_loss + x_1_loss + prob_loss
      l.backward()
      trainer.step()
      
      acc_x_t += x_t_loss 
      acc_x_1 += x_1_loss 
      acc_prob += prob_loss

      # tepoch.set_description(f"batch {batch_num}")
      # tepoch.set_postfix(
      #                    x_t_hidden=x_t_loss.item(),
      #                    x_1_loss=x_1_loss.item(),
      #                    prob_loss=prob_loss.item(),
      #                    tot_loss=l.item())

      if batch_num % 25 == 0:
        summary.write(f"batch {batch_num} avg x_t_loss, x_1_loss, prob_loss: {acc_x_t / (batch_num + 1)}, {acc_x_1 / (batch_num + 1)}, {acc_prob / (batch_num + 1)}\n")

  print(f"epoch {epoch} average x_t_loss, x_1_loss, prob_loss: {acc_x_t / len(train_loader)}, {acc_x_1 / len(train_loader)}, {acc_prob / len(train_loader)}")
summary.close()

mem_report()

"""# Evaluate"""

# trial on inference
# model = torch.load("model-200-sample-pretrain-embedding.pickle").to(device)
# model.model.add_module("activation", activations.GELUActivation())
model.eval()
idx = 11
origin_text = train_dataset.caption.loc[idx]
print("origin text: ", origin_text)

sample = train_dataset[idx]
image_clip = sample["image_clip"][None, None, :]
text_clip = sample["text_clip"][None, None, :]
x_0 = model.embedding(sample["input_ids"].unsqueeze(0))
t = 500
print(f"t = {t}")
x_t = diffuse_t(x_0, torch.tensor([t], dtype=torch.int64, device=device))
mask = sample["attention_mask"].unsqueeze(0)

# multi-step inference
restored = x_t
for i in range(5):
  out, restored = model(restored[:, :MAX_LENGTH, :], image_clip, text_clip, mask)
  print("inferred: ", train_dataset.tokenizer.decode(out.argmax(dim=-1)[0])[:len(origin_text) + 10])

# effectiveness of model on large t
print("text t effectiveness")
for i in range(1, STEP_TOT, 100):
  x_t = diffuse_t(x_0, torch.tensor([i], dtype=torch.int64, device=device))
  out, _ = model(x_t, image_clip, text_clip, mask) 

  print("t: ", i, "restore: ", train_dataset.tokenizer.decode(out.argmax(dim=-1)[0])[:len(origin_text) + 20])

torch.save(model.cpu(), f"{MODEL_NAME}.pickle")