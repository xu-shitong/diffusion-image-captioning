import pandas as pd
import copy
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)

# download pandas data
train_path = './train.csv'
valid_path = './valid.csv'
test_path = './test.csv'

train_df = pd.read_csv(train_path).dropna()
valid_df = pd.read_csv(valid_path).dropna()
test_df = pd.read_csv(test_path).dropna()

# download model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained("./tokenizers/distilbert-base-uncased-local")
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
model.save_pretrained("./models/distilbert-base-uncased-local/")

# hyperparameters
batch_size = 16
max_length = 256 # max text length
learning_rate = 0.01
beta_min = 0.0001
beta_max = 0.02
step_tot = 1000

# configuration = DistilBertConfig()
# print(configuration)
betas = torch.linspace(beta_min, beta_max, step_tot).to(device)
alphas = 1 - betas
alpha_cumprod = torch.cumprod(alphas[:-1], 0)
def diffuse_t(x, t):
  noise = torch.normal(0, 1, x.shape).to(device)
  return torch.sqrt(alpha_cumprod[t]) * x + noise * torch.sqrt(1 - alpha_cumprod[t])

def loss(model, x, mask, t, loss_func):
  '''
  model: torch model accept x shape as input
  x: x_0
  alpha_cumprod: bar_alpha list
  loss_func: "l1" or "l2" loss function between x_0 and predicted x_0
  '''
  noised = diffuse_t(x, t)
  x_0_hat = model(input_ids=noised, attention_mask=mask, output_hidden_states=True)[1][0]
  return loss_func(x_0_hat, x)


# Initializing a model from the configuration
model = DistilBertForMaskedLM.from_pretrained("./models/distilbert-base-uncased-local", local_files_only=True).to(device)
embedding = model.get_input_embeddings().requires_grad_(False)
model.set_input_embeddings(nn.Sequential())

trainer = optim.Adam(model.parameters(), lr=learning_rate)

# define dataset
class DPMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, input_df, embedding):
        self.tokenizer = tokenizer
        self.texts = input_df['text'].tolist()
        self.embedding = embedding

    def collate_fn(self, batch):
        # function for batch allocation
        texts = []

        for b in batch:
            texts.append(b)

        encodings = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)

        return {"embeddings": self.embedding(encodings["input_ids"]).to(device), "attention_mask": encodings["attention_mask"].to(device)}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

tokenizer = DistilBertTokenizer.from_pretrained("./tokenizers/distilbert-base-uncased-local", local_files_only=True)
train_dataset = DPMDataset(tokenizer, train_df, embedding)
# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=train_dataset.collate_fn)
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, collate_fn=train_dataset.collate_fn)

# training
model.train()
for x in train_loader:
  acc_loss = 0
  for t in range(1, step_tot + 1, 30):
    trainer.zero_grad()
    l = loss(model, x["embeddings"], x["attention_mask"], t, nn.L1Loss())
    l.backward()
    trainer.step()

    acc_loss += l
    break
  print(f"average loss: {acc_loss / len(train_dataset)}")
  break
  
# save model
torch.save({"net": model.to(torch.device("cpu"))}, "model.pickle")