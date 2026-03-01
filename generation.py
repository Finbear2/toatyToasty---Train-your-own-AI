import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from trainer import Brain  # imports your model class

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# load config
with open(input("Enter config file path, defaults to models/config.pkl >>> ") or "models/config.pkl", "rb") as f:
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
model = Brain(vocabSize, nEmbd, nHead, nLayer, blockSize).to(device)
model.load_state_dict(torch.load(input("Enter model path, defaults to models/model.pth >>>") or "models/model.pth"))
model.eval()

temp = float(input("Enter a temprature, defaults to 0.4 >>> ") or 0.4)
maxTokens = int(input("Enter max amount of tokens that should be generated, defaults to 500 >>> ") or 500)

# generate
while True:
    prompt = input("Enter a prompt >>> ")
    context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    output = decode(model.generate(context, max_new_tokens=maxTokens, temperature=temp)[0].tolist())
    print("\n--- OUTPUT ---\n")
    print(output,"\n")