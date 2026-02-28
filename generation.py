import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from trainer import Brain  # imports your model class

# load config
with open("config.pkl", "rb") as f:
    config = pickle.load(f)

# extract everything
blockSize = config["blockSize"]
nEmbd = config["nEmbd"]
nHead = config["nHead"]
nLayer = config["nLayer"]
vocabSize = config["vocabSize"]
stoi = config["stoi"]
itos = config["itos"]

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# rebuild and load model
model = Brain(vocabSize, nEmbd, nHead, nLayer, blockSize)
model.load_state_dict(torch.load("model.pth"))
model.eval()

temp = float(input("Enter a temprature, defaults to 0.7 >>> ") or 0.7)

# generate
while True:
    prompt = input("Enter a prompt >>> ")
    context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0)
    output = decode(model.generate(context, max_new_tokens=500, temperature=temp)[0].tolist())
    print("\n--- OUTPUT ---\n")
    print(output)