import pandas as pd
import copy
from transformers import (
    DistilBertTokenizer, DistilBertForMaskedLM, DistilBertConfig,
    BertTokenizer, BertModel as Bert
)
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import tqdm
import matplotlib.pyplot as plt

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)
print("using device: ", dev)

# read pandas data
train_path = './train.csv'
valid_path = './valid.csv'
test_path = './test.csv'

train_df = pd.read_csv(train_path).dropna()
valid_df = pd.read_csv(valid_path).dropna()
test_df = pd.read_csv(test_path).dropna()

# download pretrained model and tokenizer
def save_model_tokenizer(tokenizer_class, model_class, name):
  tokenizer = tokenizer_class.from_pretrained(name)
  tokenizer.save_pretrained(f"./tokenizers/{name}-local")
  model = model_class.from_pretrained(name)
  model.save_pretrained(f"./models/{name}-local/")

save_model_tokenizer(DistilBertTokenizer, DistilBertForMaskedLM, "distilbert-base-uncased")
# save_model_tokenizer(BertTokenizer, Bert, "bert-base-cased")

# hyperparameters
batch_size = 16
max_length = 128 # max text length
learning_rate = 1e-4
epoch_num = 4
linear_probe = False

# diffusion hyperparameter
beta_min = 0.0001
beta_max = 0.02
step_tot = 2000 # total noise adding steps
sample_size = 3 # number of sample steps in each diffuse sequence
x_0_prediction = False # if model predicts x_0 or x_{t-1}

class DistilBertModel(nn.Module):
  def __init__(self, config=None) -> None:
    super().__init__()

    self.model = DistilBertForMaskedLM.from_pretrained("./models/distilbert-base-uncased-local", local_files_only=True, config=config).to(device)
    
    self.embedding = copy.deepcopy(self.model.get_input_embeddings().requires_grad_(False))
    self.projection = copy.deepcopy(self.model.get_output_embeddings().requires_grad_(False))
    self.model.set_input_embeddings(nn.Sequential())
    self.model.set_output_embeddings(nn.Sequential())

    # print(self.model.config)

  def parameters(self):
    return list(model.model.parameters()) + list(model.embedding.parameters()) + list(model.projection.parameters())
  
  def forward(self, x, mask):
    '''
    return 
      feature_out, shape: [batch_size, seq_len, dim]
      vocab_out, shape: [batch_size, seq_len, vocab_size]
    '''
    
    x_out = self.model(x, mask)[0]
    return self.projection(x_out), x_out

class EncoderModel(nn.Module): # ABANDONED: mask shape not known meaning
  def __init__(self, 
               layer_dim=512, 
               nhead=8, 
               activation='gelu',
               dropout=0.1,
               num_layer=6,
               train_embedding=False) -> None:
    super().__init__()

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=layer_dim, 
        nhead=nhead,
        dim_feedforward=2048,
        activation=activation,
        dropout=dropout,
        batch_first=False,
        norm_first=False,
        device=device)
    self.model = nn.TransformerEncoder(
        encoder_layer, 
        num_layers=num_layer,
        norm=None)
    self.embedding = nn.Embedding(
        30522,
        layer_dim, 
        padding_idx=None, 
        max_norm=None, 
        norm_type=2.0, 
        scale_grad_by_freq=False, 
        sparse=False, 
        device=device)
    if not train_embedding:
      self.embedding.requires_grad_(False)
    
  def forward(self, x, mask):
    return self.model(x, mask)

class BertModel(nn.Module): # ABANDONED
  def __init__(self, train_embedding=False) -> None:
    super().__init__()

    self.model = Bert.from_pretrained("./models/bert-base-cased-local/", local_files_only=True)

    self.embedding = self.model.get_input_embeddings()
    if not train_embedding:
      self.embedding.requires_grad_(False)
    self.model.set_input_embeddings(nn.Sequential())

  def forward(self, x, mask):
    return self.model(x, mask)

configuration = DistilBertConfig()
model = DistilBertModel(config=configuration)
# model = EncoderModel(train_embedding=train_embedding)
# model = BertModel(train_embedding=train_embedding)

if linear_probe:  
  # TODO: linear probation not supported
  NotImplementedError()
  # trainer = optim.Adam(model.projection.parameters(), lr=learning_rate)
else:
  # parameter only include model, no embedding layer
  trainer = optim.Adam(model.parameters(), lr=learning_rate)

betas = torch.hstack([torch.zeros(1), torch.linspace(beta_min, beta_max, step_tot)]).to(device)
alphas = 1 - betas
alpha_cumprod = torch.cumprod(alphas[:-1], 0)
def diffuse_t(x, t):
  '''
  x_shape: [batch_size, seq_len, dim]
  t shape: [sample num]

  return shape [batch_size * sample_num, seq_len, dim]
  '''
  batch_size, seq_len, dim = x.shape
  sample_shape = (sample_size, *(1, ) * len(x.shape))

  noise = torch.normal(0, 1, x.shape).to(device)
  mean = torch.sqrt(alpha_cumprod[t].reshape(sample_shape)) * x 
  epsilon = noise * torch.sqrt(1 - alpha_cumprod[t]).reshape(sample_shape)
  return (mean + epsilon).reshape((sample_size * batch_size, seq_len, dim))

def generate_diffuse_pair(x_0, repeat_shape, t, t_next=-1):
  '''
  x_0 shape: [batch_size, seq_len, dim]
  t shape: [sample_num]
  repeat shape: (sample_num, 1, 1, ...)
  
  return (net input, net target)
    shape [batch_size * sample_num, seq_len, dim]
  '''
  if t_next == -1:
    # predict x_0
    return (diffuse_t(x_0, t), x_0.repeat(repeat_shape))

  # predict x_{t_next}
  return (diffuse_t(x_0, t), diffuse_t(x_0, t_next))

def loss(model, x_input, x_tgt, mask, loss_func):
  _, x_hat = model(x_input, mask)
  return loss_func(x_hat, x_tgt)

# define dataset 
class DPMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, input_df):
        self.tokenizer = tokenizer
        self.texts = input_df['text'].tolist()

    def collate_fn(self, batch):
        # function for batch allocation
        texts = []

        for b in batch:
            texts.append(b)

        encodings = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)

        return {"input_ids": encodings["input_ids"].to(device), "attention_mask": encodings["attention_mask"].to(device)}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

tokenizer = DistilBertTokenizer.from_pretrained("./tokenizers/distilbert-base-uncased-local/", local_files_only=True)
# tokenizer = BertTokenizer.from_pretrained("./tokenizers/bert-base-cased-local", local_files_only=True)

train_dataset = DPMDataset(tokenizer, train_df)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=train_dataset.collate_fn)

# training
model.train()
print("start training")
for epoch in range(epoch_num):
  acc_loss = 0
  # with tqdm.tqdm(train_loader, unit="batch") as tepoch: 
  #   for epoch, x in enumerate(tepoch):
  for x in train_loader:
      x_0 = model.embedding(x["input_ids"])
      repeat_shape = (sample_size, *(1, ) * (len(x_0.shape) - 1))
      t = torch.randint(0, step_tot, repeat_shape, device=device)
      if x_0_prediction:
        x_input, x_tgt = generate_diffuse_pair(x_0, repeat_shape, t)
      else:
        x_input, x_tgt = generate_diffuse_pair(x_0, repeat_shape, t, torch.max(t - 30, 0))

      trainer.zero_grad()
      l = loss(model, x_input, x_tgt, x["attention_mask"].repeat(repeat_shape), nn.L1Loss())
      l.backward()
      trainer.step()

      acc_loss += l
      break

      # tepoch.set_description(f"Epoch {epoch}")
      # tepoch.set_postfix(Loss=l)

  print(f"epoch {epoch} average loss: {acc_loss / len(train_loader)}")
  break

torch.save({"net": model.to(torch.device("cpu"))}, "model.pickle")