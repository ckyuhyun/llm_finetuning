# PyTorch for implementing LLM (No GPU)
import torch

# Neural network modules and functions from PyTorch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as pt
from sklearn.model_selection import train_test_split

import time
import numpy as np
import pandas as pd
import urllib.request

# Configuration object for model parameters
MASTER_CONFIG = {}


# url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# filename = "tinyshakespeare.txt"
# urllib.request.urlretrieve(url, filename)


class data_preprocessing:
    def __init__(self):
        self.itos = None
        self.stoi = None
        self.dataset = None
        self.vocab = None

    def tokenizing(self):
        # TODO : need to implement with SentencePiece https://github.com/google/ sentencepiece
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}


    def get_unique_vocab_from_file(self, file_name: str):
        with open(file_name, 'r') as fp:
            self.dataset = fp.read()

        self.vocab = sorted(list(set(self.dataset)))


class encoder_decoder(data_preprocessing):
    def __init__(self):
        super().__init__()

    def data_preprocessing(self, file_name:str):
        self.get_unique_vocab_from_file(file_name)
        self.tokenizing()

    def encode(self, ss):
        return [self.stoi[ch] for ch in ss]

    def decode(self, ll):
        return ''.join([self.itos[i] for i in ll])

    def convert_data_to_tensor(self):
        return torch.tensor(self.encode(self.dataset), dtype=torch.int8)


class LLM(encoder_decoder):
    def __init__(self):
        super().__init__()

    def get_batches(self, split, batch_size, context_window, config=MASTER_CONFIG):
        tensor_dataset = self.convert_data_to_tensor()

        data_length = len(tensor_dataset)
        train = tensor_dataset[: int(.8 * data_length)]
        test = tensor_dataset[int(.8 * data_length):int(.9 * data_length)]
        val = tensor_dataset[int(.9 * data_length):]

        # Determine which split to use
        batch_data = train if split == 'train' else test if split == 'test' else val

        # Pick random starting points within the data
        ix = torch.randint(0, (batch_data.size(0) - context_window - 1), (batch_size,))

        x = torch.stack([batch_data[i: i + context_window] for i in ix]).long()
        y = torch.stack([batch_data[i + 1: i + context_window + 1] for i in ix]).long()

        return x, y

    @torch.no_grad()
    def evaluate_loss(self, model, config=MASTER_CONFIG):
        # placeholder for the evaluation results
        out = {}

        # Set the model to evaluate model.
        # model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference
        model.eval()

        # Iterate through training and validation splits
        for split in ["train", "val"]:
            # Placeholder for individual losses
            losses = []

            # Generate 10 batches for evaluation
            for _ in range(10):
                # Get input sequences (xb) and target sequences (yb)
                xb, yb = self.get_batches(split, config['batch_size'], config['context_window'])

                # Perform model inference and calculate the loss
                _, loss = model(xb, yb)

                # Append the loss to the list
                losses.append(loss.item())

            out[split] = np.mean(losses)

        # Set the model back to training mode
        model.train()

        return out

    def train(self, model, optimizer, schedular=None, config=MASTER_CONFIG, print_logs=False):
        # Placeholder for storing losses
        losses = []

        # Start tracking time
        start_time = time.time()

        # Iterate through epoch
        for epoch in range(config.get('epochs')):
            # Zero out gradients
            optimizer.zero_grad()

            # Obtain batches for training
            xs, ys = self.get_batches(split='train', batch_size=MASTER_CONFIG['batch_size'], context_window=MASTER_CONFIG.get('context_window'))

            # Forward pass through the model to calculate logits and loss
            logits, loss = model(xs, targets=ys)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # If a learning rate scheduler is provided, adjust the learning rate
            if schedular:
                schedular.step()

            if epoch % config.get('log_interval') == 0:
                # Calculate batch time
                batch_time = time.time() - start_time

                # Evaluate loss on validation set
                x = self.evaluate_loss(model)

                losses += [x]

                # Print progress logs if specified
                if print_logs:
                    print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch) / config['log_interval'] :.3f}")

                # Reset the timer
                start_time = time.time()

                if schedular:
                    print("lr:", schedular.get_lr())

            print(f"Validation loss: {losses[-1]['val']}")


class SimpleBrokenModel(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
        super().__init__()
        self.config = config

        # Embedding layer to convert character indicis to vectors
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])

        # Linear layers for modeling relationships between features
        # (to be updated with SwiGLU activation function as in LLaMA)
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size'])
        )
        # Print the total number of model parameters
        # print("Model parameters:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        # Embedding layer converts character indices to vectors
        x = self.embedding(idx)

        # Linear layers for modeling relationships between features
        a = self.linear(x)

        logits = F.softmax(a, dim=-1)

        # If targets are provided, calculate and return the cross-entropy loss
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        else:
            return logits




def run():
    # Update MASTER_CONFIG with training parameters
    MASTER_CONFIG.update({
        'batch_size': 8,  # Number of batches to be processed at each random split
        'context_window': 16,  # Number of characters in each input (x) and target (y) sequence of each batch
        'd_model': 128,
        'vocab_size': 65,
        'epochs': 100,  # Number of training epochs
        'log_interval': 10,  # Log information every 10 batches during training
        'batch_size': 32,  # Increase batch size to 32
    })
    llm = LLM()
    llm.data_preprocessing(file_name="tinyshakespeare.txt")

    xs, ys = llm.get_batches(split='train', batch_size=MASTER_CONFIG['batch_size'], context_window=MASTER_CONFIG.get('context_window'))

    model = SimpleBrokenModel(MASTER_CONFIG)
    logits, loss = model(xs, ys)

    optimizer = torch.optim.Adam(
        model.parameters()
    )

    llm.train(model, optimizer)







    #print(f'Loss: {loss.item()}')


    # llm = LLM()
    # llm.get_unique_vocab_from_file("tinyshakespeare.txt")
    # xs, ys = llm.get_batches(split='train', batch_size=MASTER_CONFIG['batch_size'], context_window=MASTER_CONFIG['context_window'])
    #
    # decoded_samples = [(llm.decode(xs[i].tolist()), llm.decode(ys[i].tolist())) for i in range(len(xs))]
    #
    # print(decoded_samples)


if __name__ == "__main__":
    run()
