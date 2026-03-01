import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import sys

# RENEMBER

# it's a massive neural nework!
# numbers go in →
# multiply by weights →
# do some maths →
# numbers come out →
# compare to correct answer →
# adjust weights →
# repeat :)

# The actual model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# Kinda the neuron
class Head(nn.Module):
    # Initiliases everything
    def __init__(self, head_size, nEmbd, blockSize):
        super().__init__()
        self.key = nn.Linear(nEmbd, head_size, bias=False) # What imformation do I contain
        self.query = nn.Linear(nEmbd, head_size, bias=False) # What imformation am I looking for 
        self.value = nn.Linear(nEmbd, head_size, bias=False) # What imforamtion do I actually pass forward 
        self.register_buffer('tril', torch.tril(torch.ones(blockSize, blockSize))) # Allows the model to only look backwards not forwards, almost like a cheating prevention 

    def forward(self, x):
        B, T, C = x.shape # B - Batch size, T - Sequence lrngth, C - Embedding dimensions
        k = self.key(x) # Searching through the key
        q = self.query(x) # Searching through the query

        # attention scores
        # Transpose swaps the last two dimensions
        # q @ is the matrix manipulation which compares against every key
        # * C**-0.5 divides by square root of C to stop the numbers getting stupidly large and breaking softmax
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Sets all future positions to negative infinty, means it can't look forward
        wei = F.softmax(wei, dim=-1) # Converts soresto probabilities that add up to 1, negative infinity becomes zero
        v = self.value(x) # Gets the imformation to pass on
        return wei @ v # Multiplies attention probabilities by values. Each character gets a weighted mix of information from all previous characters based on how relevant they are.

# Combines the heads together
class MultiHeadAttention(nn.Module):
    # Initiliases everything
    def __init__(self, numHeads, headSize, nEmbd, blockSize):
        super().__init__()
        self.heads = nn.ModuleList([Head(headSize, nEmbd, blockSize) for _ in range(numHeads)]) # Creates mutliple heads
        self.proj = nn.Linear(nEmbd, nEmbd) # Puts all the outputs into the kinda right shape

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # Collects all the heads output andglues them side by side
        return self.proj(out) # Mixes all the imformation together

# Thinks about what imformation it's gathered
class FeedForward(nn.Module):
    # Initiliases everything
    def __init__(self, nEmbd):
        super().__init__()
        #Sequential runs the layers in order
        self.net = nn.Sequential(
            nn.Linear(nEmbd, 4 * nEmbd), # Exapnds it dimensions to give it more room to spot patterns
            nn.ReLU(), # Negative number = 0, positive number = same. Basically kills any not usefull patterns
            nn.Linear(4 * nEmbd, nEmbd), # Reduces the size again
        )

    def forward(self, x):
        return self.net(x) # Runs the sequential

# Just contains alot of stuff
class Block(nn.Module):
    # Initiliases everything
    def __init__(self, nEmbd, n_head, blockSize):
        super().__init__()
        head_size = nEmbd // n_head # Divides it into an integer because you can't have half a dimension
        self.sa = MultiHeadAttention(n_head, head_size, nEmbd, blockSize) # The multihead
        self.ffwd = FeedForward(nEmbd) # The feedforward
        self.ln1 = nn.LayerNorm(nEmbd) # Layer Norm 1
        self.ln2 = nn.LayerNorm(nEmbd) # Layer Norm 2

    def forward(self, x):
        # Feeds everything forward
        x = x + self.sa(self.ln1(x)) # You pass throguh the x again for the risidual conection
        x = x + self.ffwd(self.ln2(x))
        return x

# Joins everything together
class Brain(nn.Module):
    # Initiliases everything
    def __init__(self, vocabSize, nEmbd, nHead, nLayer, blockSize):
        super().__init__()
        self.token_embedding = nn.Embedding(vocabSize, nEmbd) # Converts letter ints to vectors
        self.position_embedding = nn.Embedding(blockSize, nEmbd) # Converts the position into a vector representing different positions
        self.blocks = nn.Sequential(*[Block(nEmbd, nHead, blockSize) for _ in range(nLayer)]) # Makes a block for each layer and runs them in sequence
        self.ln_f = nn.LayerNorm(nEmbd) # Normalizes everything
        self.lm_head = nn.Linear(nEmbd, vocabSize) # Matric multiplication, takes in n_emd numbers and spits out vocab size numbers
        self.blockSize = blockSize

    def forward(self, idx, targets=None):
        B, T = idx.shape # Unpacks everything
        tok_emb = self.token_embedding(idx) # Converts characters to vectors
        pos_emb = self.position_embedding(torch.arange(T, device=device).to(device)) # just generates [0, 1, 2, 3...] up to T, then converts those positions to vectors
        x = tok_emb + pos_emb # Combines what and where you know
        x = self.blocks(x) # Runs through all transofrmer blocks
        x = self.ln_f(x) # Final layernorm before output
        logits = self.lm_head(x) # Converts to vocabulary size predictions

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) # Mesaures how bad the predictions were
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.5):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.blockSize:] # Crops size to block size
            logits, _ = self(idx_cond) # Grabs only the last characters predictions
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # Samples randomly
            idx = torch.cat((idx, idx_next), dim=1) # Appends new character and starts again
        return idx

if __name__ == "__main__":

    # Hyper parameters
    blockSize = int(input("Please enter block size/context window, defaults to 32 >>> ") or 32) # Context window
    batchSize = int(input("Please enter batch size, defaults to 16 >>> ") or 16) # How many sequences to train on at once
    maxIters = int(input("Please enter max iterations/training steps, defaults to 5000 >>> ") or 5000) # Training steps
    learningRate = float(input("Please enter learning rate, defaults to 1e-2 >>> ") or 1e-2) # High LR = More chaos
    nEmbd = int(input("Please enter amount of embedding dimensions, defaults to 64 >>> ") or 64) # Embedding dimensions, How long each coordinate is
    nHead = int(input("Please enter amount of attention heads, defaults to 2 >>> ") or 2) # Attention heads
    nLayer = int(input("Please enter amount of tranformer layers, defaults to 2 >>> ") or 2) # Transformer layers
    dropout = float(input("Please enter dropout, defaults to 0 >>> ") or 0) # No dropout = raw chaos
    temp = float(input("Please enter a temprature, this can be changed in genertaion.py and is only used for test iutput, defaults to 0.4 >>> ") or 0.4)
    targetLoss = float(input("Please enter target loss for model, defaults to 1.1 >>>") or 1.1)

    # Load the training data
    print("\nLoading all text from data folder into training data...")

    # Check if data folder is empty
    if not os.listdir("data"):
        print("No text data found in data folder, please add .txt files!\nExiting program...")
        sys.exit() # Exit program early

    text=list()
    for file in os.scandir("data"):
        if file.is_file() and file.name.endswith(".txt"):
            with open(file.path, "r", encoding="utf-8") as f:
                text.append(f.read())
                print(f"Loaded {file.name} into training data...")
    text = "\n\n\n---Next Book---\n\n\n".join(text) # Join every peice of text together
    print("Finished loading training data!")

    # Build the vocabulary
    chars = sorted(list(set(text)))
    vocabSize = len(chars)
    print(f"\nVocab made!\nVocab size: {vocabSize}\nDataset size: {len(text)}")

    # Encoder and Decoder for the llm
    stoi = {ch: i for i, ch in enumerate(chars)} # Letter to Int
    itos = {i: ch for i, ch in enumerate(chars)} # Int to Letter
    # Make the encoding and ecoding functions 
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # Split the data into train and eval, 90% training & 10% testing
    data = torch.tensor(encode(text), dtype=torch.long) # Put encoded data in a tensor
    n = int(0.9 * len(data)) # get 90% of the data
    trainingData = data[:n] # Everything below 90%
    evalData = data[n:] # Everything above 90%


    # Data loader
    def getBatch(type):
        data = trainingData if type == "train" else evalData # Get data based off type 
        ix = torch.randint(len(data) - blockSize, (batchSize,)) # Get {batch size} amount of positions of blocks at {block size} [4521, 234, 8876, 1203, ...]
        x = torch.stack([data[i:i+blockSize] for i in ix]) # Get the current block
        y = torch.stack([data[i+1:i+blockSize+1] for i in ix]) # Get the next block

        # x = "i'm not dea"
        # y = "'m not dead"

        return x.to(device), y.to(device)

    # Create the model
    print("\nCreating model...")
    model = Brain(vocabSize, nEmbd, nHead, nLayer, blockSize).to(device)
    print("Created model!")

    # Get and display the total parameters
    totalParams = sum(p.numel() for p in model.parameters())
    print(f"Parameter total: {totalParams}")

    if input("\nDo you want to continue with training model, this will take some time [Y/n] >>> ").lower() == "n":
        print("Exiting program...")
        sys.exit() # Exiting program early 
    else:
        print("Continuing with training model!")

    # TRAIN IT HEHEHEHE

    print("\nBeginging training process. Please wait as this may take a while, DON'T CLOSE THE TERMINAL...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)

    # Change the learning rate overtime so it can fine tune it and increases speed
    # 5 minute training becomes 1 minute in testing
    # 800k model now outpreforms a 17m parameter model 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=maxIters, eta_min=1e-5
    )

    for i in range(maxIters):
        xb, yb = getBatch("train") # Get random training batch
        logits, loss = model(xb, yb) # Runs the forward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # Turn the knobs
        scheduler.step()

        # Alert user of stuff happening
        if i % 100 == 0:
            print(f"step {i}: loss {loss.item():.4f}...")

        if loss.item() <= targetLoss:
            print("Reached target loss, stopping training!")
            break

    print("Finished training!")

    # GENERATION

    context = torch.zeros((1, 1), dtype=torch.long).to(device)
    output = decode(model.generate(context, max_new_tokens=500, temperature=temp)[0].tolist())
    print("\n--- Model finished, now generating first message. ---\n")
    print(output)

    print("\nSaving model to models/model.pth")
    if os.path.isfile("models/model.pth"):
        if input("This will over-ride current file occupying models/model.pth, do you want to do this? [y/N] >>> ").lower() == "y":
            torch.save(model.state_dict(), "models/model.pth")
            print("Saved model to models/model.pth!")
        else:
            print("Not saving model!")
    else:
        torch.save(model.state_dict(), "models/model.pth")
        print("Saved model to models/model.pth!")

    config = {
        "blockSize": blockSize,
        "nEmbd": nEmbd,
        "nHead": nHead,
        "nLayer": nLayer,
        "vocabSize": vocabSize,
        "stoi": stoi,
        "itos": itos
    }

    print("\nSaving config file to models/config.pkl...")
    if os.path.isfile("models/config.pkl"):
        if input("This will over-ride current file occupying models/model.pth, do you want to do this? [y/N] >>> ").lower() == "y":
            with open("models/config.pkl", "wb") as f:
                pickle.dump(config, f)
                print("Saved model config to models/config.pkl")
        else:
            print("Not saving config!")
    else:
        with open("models/config.pkl", "wb") as f:
            pickle.dump(config, f)
            print("Saved model config to models/config.pkl")

    print("\nModel creation finished, prompt the model with genertaion.py by running it and entering model path, config path,\ndesired temprature in responses and max token length")
    