from torchvision import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel as CLIP
import tqdm
from torchtext.data.metrics import bleu_score
import re
from torch import nn
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
import matplotlib.pyplot as plt
import math

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)
print("using device: ", dev)

# hyperparameters
DEBUG = False
BATCH_SIZE = 8
MAX_LENGTH = 16 # max text length
LEARNING_RATE = 5e-5
# END_LEARNING_RATE = 5e-5 # learning rate is linearly reduced to end_learning_rate
END_LEARNING_RATE = LEARNING_RATE # no changing learning rate

def cosine_annealing():
  sub_epoch = 5
  x = torch.arange(0, sub_epoch)
  return END_LEARNING_RATE + (LEARNING_RATE - END_LEARNING_RATE) * (1 + torch.cos(x / sub_epoch * math.pi)) / 2
# SCHEDULER = torch.logspace
SCHEDULER = torch.linspace
# SCHEDULER = cosine_annealing # scheduler of learning rateTRAIN_SET_RATIO = 0.95
EARLY_STOP_RATIO = 1.05
EPOCH_NUM = 15
DYNAMIC_ROUNDING_WEIGHT = -1 # weight of rounding term with respect to x_t loss, <0 means not using 
ROUNDING_WEIGHT = 0.3 # weight of rounding term, the probability of regenerated sequence, not used if using dynamic rounding

def series_sum_sample_mean(x_hat, x):
  return (x_hat - x).abs().sum(dim=1).mean()

def series_sum(x_hat, x):
  return (x_hat - x).abs().sum() / BATCH_SIZE / 768 / 100

def mse_series_mean(x_hat, x):
  return ((x_hat - x) ** 2).sum(dim=[-2, -1]).sqrt().mean()

def mse_series_sum(x_hat, x):
  return ((x_hat - x) ** 2).sum(dim=[-2, -1]).sqrt().sum() / BATCH_SIZE

LOSS_FUNC = series_sum_sample_mean
# LOSS_FUNC = series_sum
# LOSS_FUNC = mse_series_mean
# LOSS_FUNC = mse_series_sum # loss function used between embedding 
# CLIP_ADDING_METHOD = "add" # CLIP feature are added as position embedding to sequence of word embedding
CLIP_ADDING_METHOD = "concat" # CLIP feature are appended to sequence of word embedding
# # CLIP_MASK = None
# CLIP_MASK = torch.tensor([1, 0], device=device) # mask indicating if [image, text] clip feature is used, None means use classification free guidance
CLASSIFIER_FREE_WEIGHT = 0
# CLASSIFIER_FREE_WEIGHT = 0.3 # classifier guidance, 0 means no guidance
CLASSIFIER_FREE_PROB = 0.2
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
X_T_STEP_INTERVAL = 100
USE_X_T_LOSS = True
USE_X_1_LOSS = True # if using x_1 loss
USE_PROB_LOSS = True # if using prob loss

MODEL_NAME = f"epoch{EPOCH_NUM}_loss{LOSS_FUNC.__name__}_lr{'%.0E' % LEARNING_RATE}-{'%.0E' % END_LEARNING_RATE}_scheduler{SCHEDULER.__name__}_round{'%.0E' % ROUNDING_WEIGHT}_dynamic{DYNAMIC_ROUNDING_WEIGHT}\
_clip{CLIP_ADDING_METHOD}_class_weight{'%.0E' % CLASSIFIER_FREE_WEIGHT}_class_prob{'%.0E' % CLASSIFIER_FREE_PROB}_train-embed{TRAIN_EMBEDDING}\
_samplesize{SAMPLE_SIZE}_x_0_predict{X_0_PREDICTION}_X_INTERVAL{X_T_STEP_INTERVAL}_use_x_t{USE_X_T_LOSS}_use_x_1{USE_X_1_LOSS}_use_prob{USE_PROB_LOSS}"

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

  def forward(self, x, image_clip, text_clip, mask, concat_mask):
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
    assert concat_mask.shape == (sample_batch_multi, 2)

    # mask of which sample is classifier free guided, true if guided
    guidance_sample_index = (concat_mask[:, 1] == 1)

    if TRAIN_EMBEDDING:
      x = self.input_projection(x)
    
    if CLIP_ADDING_METHOD == "concat":
      classifier_guided_mask = torch.hstack([mask, torch.tensor([1, 1], device=device).repeat(sample_batch_multi, 1)])
      non_classifier_mask = torch.hstack([mask, torch.tensor([1, 0], device=device).repeat(sample_batch_multi, 1)])

      x = torch.hstack([x, self.image_linear(image_clip), self.text_linear(text_clip)])
      x = x + self.segment_embedding(torch.tensor([0] * MAX_LENGTH + [1] * 2, device=device))

      classifier_guided_x = non_classifier_x = x
    elif CLIP_ADDING_METHOD == "add":
      classifier_guided_mask = non_classifier_mask = mask

      non_classifier_x = x + self.image_linear(image_clip)
      classifier_guided_x = non_classifier_x + self.text_linear(text_clip)
    else:
      raise NotImplementedError(CLIP_ADDING_METHOD)

    # no classifier guidance part
    x_out = self.model(non_classifier_x, non_classifier_mask)[0]
    if CLASSIFIER_FREE_WEIGHT > 0 and not guidance_sample_index.sum() == 0:
      # classifier guided
      x_out[guidance_sample_index] = \
        (1 + CLASSIFIER_FREE_WEIGHT) * self.model(classifier_guided_x[guidance_sample_index], classifier_guided_mask[guidance_sample_index])[0] \
        - CLASSIFIER_FREE_WEIGHT * x_out[guidance_sample_index]
    
    if TRAIN_EMBEDDING:
      x_out = self.output_projection(x_out)

    assert x_out.shape == (sample_batch_multi, non_classifier_mask.shape[-1], IN_CHANNEL)
    return self.lm_head(x_out[:, :MAX_LENGTH, :]), x_out
    
if TRAIN_EMBEDDING:
  configuration = DistilBertConfig()
  model = DistilBertModel(config=configuration)
else:
  origin = DistilBertForMaskedLM.from_pretrained("./models/distilbert-base-uncased-local", local_files_only=True).to(device)
  configuration = DistilBertConfig()
  model = DistilBertModel(origin.get_input_embeddings(), origin.get_output_embeddings(), config=configuration)

class CocoClipDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.coco_val = datasets.CocoDetection(
            root = "./coco_2014_caption/val2014", 
            annFile = "./coco_2014_caption/val2014_caption.json")

        self.clip_processor = CLIPProcessor.from_pretrained("./tokenizers/openai/clip-vit-base-patch32-local")
        self.clip = CLIP.from_pretrained("./models/openai/clip-vit-base-patch32-local")

    def __len__(self):
        return len(self.coco_val)
    
    def __getitem__(self, idx):
        img, info = self.coco_val[idx]
        inputs = self.clip_processor(text="", images=img, return_tensors="pt", padding=True)
        image_embed = self.clip.get_image_features(inputs["pixel_values"])
        image_embed = image_embed / image_embed.norm(p=2, dim=-1, keepdim=True)

        return {
            "image_clip": image_embed.to(device),
            "text": [i["caption"] for i in info],
        }

dataset = CocoClipDataset()

tokenizer = DistilBertTokenizer.from_pretrained("./tokenizers/distilbert-base-uncased-local/", local_files_only=True)

import os
import sys
print(os.path.basename(sys.argv[1]))
print(f"{MODEL_NAME}")

assert os.path.basename(sys.argv[1]) == f"{MODEL_NAME}.pickle"

model = torch.load(
  sys.argv[1]
  ).to(device)
model.model.add_module("activation", activations.GELUActivation())
model.eval()
acc_bleu = 0
with torch.no_grad():
  with tqdm.tqdm(dataset, unit="batch") as tepoch: 
    for j, x in enumerate(tepoch):

      restored = torch.randn((1, MAX_LENGTH + 2, 768), device=device)

      # each prediction involves multiple generation steps
      for i in range(5):
        out, restored = model(restored[:, :MAX_LENGTH, :], x["image_clip"].unsqueeze(1), torch.zeros_like(x["image_clip"], device=device).unsqueeze(1), torch.ones((1, MAX_LENGTH), device=device), torch.tensor([[1, 0]], device=device))

      # append final strings to each answer bin
      indexes = nn.functional.softmax(out, dim=-1).argmax(dim=-1)
      indexes = indexes.unique_consecutive(dim=-1)

      ans_strs = [re.split("\.| ", tokenizer.decode(indexes[0]))[:MAX_LENGTH]]
      
      GT_list = [[['[CLS]'] + re.split("\.| ", caption.strip().lower())[:MAX_LENGTH-2] + ['[SEP]'] for caption in x["text"]]]

      acc_bleu += bleu_score(ans_strs, GT_list)

      if j > 1000:
        break

print(acc_bleu / 1000)