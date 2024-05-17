# Import core libraries for deep learning and scientific computing, neural network building blocks
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F #Functional Utilities
import torch.optim as optim  #For Optimizer

# Import libraries for data manipulation and analysis
import pandas as pd
import csv

# Import libraries for progress monitoring and visualization
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Import libraries for logging and experimentation tracking
import wandb  

# Import libraries for utility functions
import random  
import heapq  

# Import Libraries for tanking argument from command line
import argparse

# Import warnings
import warnings


parser = argparse.ArgumentParser()
parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='DL-Assignment3')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='cs23m026')
parser.add_argument('-d', '--datapath', help='give data path e.g. /kaggle/input/vocabs/Dataset', type=str, default='D:/DL_A3/Dataset')
parser.add_argument('-l', '--lang', help='languge', type=str, default='hin')
parser.add_argument('-e', '--epochs', help="Number of epochs to train network.", type=int, default=10)
parser.add_argument('-b', '--batch_size', help="Batch size used to train network.", type=int, default=32)
parser.add_argument('-dp', '--dropout', help="dropout probablity in Ecoder & Decoder", type=float, default=0.3)
parser.add_argument('-nl', '--num_layers', help="number of layers in encoder & decoder", type=int, default=2)
parser.add_argument('-bw', '--beam_width', help="Beam Width for beam Search", type=int, default=1)
parser.add_argument('-cell', '--cell_type', help="Cell Type of Encoder and Decoder", type=str, default="LSTM", choices=["LSTM", "RNN", "GRU"])
parser.add_argument('-emb_size', '--embadding_size', help="Embadding Size", type=int, default=256)
parser.add_argument('-hdn_size', '--hidden_size', help="Hidden Size", type=int, default=512)
parser.add_argument('-lp', '--length_penalty', help="Length Panelty", type=float, default=0.6)
parser.add_argument('-bi_dir', '--bidirectional', help="Bidirectional", type=int, choices=[0, 1], default=1)
parser.add_argument('-tfr', '--teacher_forcing_ratio', help="Teacher Forcing Ratio", type=float, default=0.5)
parser.add_argument('-o', '--optimizer', help = 'choices: ["sgd", "adagrad", "adam", "rmsprop"]', type=str, default = 'adam', choices= ["sgd", "rmsprop", "adam", "adagrad"])
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate for training', type=float, default=0.001)
parser.add_argument('-p', '--console', help='print training_accuracy + loss, validation_accuracy + loss for every epochs', choices=[0, 1], type=int, default=1)
parser.add_argument('-wl', '--wandb_log', help='log on wandb', choices=[0, 1], type=int, default=0)
parser.add_argument('-eval', '--evaluate', help='get test accuarcy and test loss', choices=[0, 1], type=int, default=1)
parser.add_argument('-t_random', '--translate_random', help='get 10 Random words and their translations from test data', choices=[0, 1], type=int, default=0)



# This function determines the appropriate device ("cpu" or "cuda") to use for training.
def set_device():
    """Sets the training device to either "cpu" or "cuda" based on availability.

    Returns:
        str: The chosen device ("cpu" or "cuda").
    """
    device = "cpu"  # Default device is CPU

    # Check if a CUDA GPU is available
    if torch.cuda.is_available():
        device = "cuda"  # Use GPU if available for faster training

    return device  # Return the chosen device


def load_data(b_p, lang='hin'):
    """
    Loads training, validation, and test data from CSV files.

    Args:
        lang (str, optional): Language code (default: 'hin'). Defaults to 'hin'.

    Returns:
        dict: A dictionary containing the loaded data and maximum sequence lengths.
    """

    # Define base paths based on language
    base_path = f'{b_p}/{lang}'
    
    train_path, val_path, test_path = f'{base_path}/{lang}_train.csv', f'{base_path}/{lang}_valid.csv', f'{base_path}/{lang}_test.csv'

    # Load data using a single loop with list comprehension
    data_lists = []
    for path in [train_path, val_path, test_path]:
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file) #read csv file
            data_lists.append([[f"{row[0]}$", f"#{row[1]}$"] for row in reader]) 
      
    data_set = []
    for i in range(0, 6):
        data_set.append([list_item[i%2] for list_item in data_lists[i//2]])
    
    train_x, train_y, val_x, val_y, test_x, test_y = data_set[0], data_set[1], data_set[2], data_set[3], data_set[4], data_set[5]


  # Convert data to NumPy arrays
    train_x, train_y = np.array(train_x), np.array(train_y)
    val_x, val_y = np.array(val_x), np.array(val_y)
    test_x, test_y = np.array(test_x), np.array(test_y)

    # Find maximum sequence lengths (combined for efficiency)
    max_decoder_length = max(len(s) for s in np.concatenate((train_y, val_y, test_y)))
    max_encoder_length = max(len(s) for s in np.concatenate((train_x, val_x, test_x)))

    # Return data as a dictionary
    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "max_decoder_length": max_decoder_length,
        "max_encoder_length": max_encoder_length
    }

def create_corpus(dictionary : dict):
    """
    Creates vocabulary dictionaries for input and output sequences.

    Args:
        dict : A dictionary containing train_y, val_y, test_y
    Returns:
        dict: A dictionary containing vocabulary information.
    """
    train_y = dictionary["train_y"]
    val_y = dictionary["val_y"]
    test_y = dictionary["test_y"]

    # Define English vocabulary
    english_vocab = "#$abcdefghijklmnopqrstuvwxyz"

    # Combine target sequences from all datasets to create a complete vocabulary
    all_chars = set.union((set(char for word in train_y for char in word)),
                            set(char for word in val_y for char in word),
                            set(char for word in test_y for char in word))
    all_chars.add('')
    all_chars = sorted(all_chars)

    # Create input vocabulary dictionary (includes the empty string)
    input_corpus_dict = {char: idx+1 for idx, char in enumerate(english_vocab)}
    input_corpus_dict[''] = 0
    input_corpus_length = len(input_corpus_dict)
    

    # Create output vocabulary dictionary (includes the empty string)
    output_corpus_dict = {char: idx for idx, char in enumerate(all_chars)}
    output_corpus_length = len(output_corpus_dict)

    # Create dictionaries for reversed lookups (character -> index)
    reversed_input_corpus = {v: k for k, v in input_corpus_dict.items()}
    reversed_output_corpus = {v: k for k, v in output_corpus_dict.items()}

    # Return a dictionary containing all vocabulary information
    return {
        "input_corpus_length": input_corpus_length,
        "output_corpus_length": output_corpus_length,
        "input_corpus_dict": input_corpus_dict,
        "output_corpus_dict": output_corpus_dict,
        "reversed_input_corpus": reversed_input_corpus,
        "reversed_output_corpus": reversed_output_corpus
    }

def create_tensor(data_dict, corpus_dict):
    """
    Creates PyTorch tensors for training and validation data.

    Args:
        data_dict (dict) : Dictionary contaning datasets
        corpus_dict (dict): Dictionary containing vocabulary information.

    Returns:
        dict: A dictionary containing PyTorch tensors for training and validation.
    """

    # Get maximum sequence length
    max_len = max(data_dict["max_encoder_length"], data_dict["max_decoder_length"])

    # Function to convert sequences to tensors with padding
    def create_padded_tensor(sequences, vocab_dict, max_len):
        tensor = np.zeros((max_len, len(sequences)), dtype='int64')
        for i, seq in enumerate(sequences):
            for j, char in enumerate(seq):
                tensor[j, i] = vocab_dict.get(char, 0)  # Use default of 0 for missing characters
        return torch.tensor(tensor)

    # Create tensors for training data
    train_input = create_padded_tensor(data_dict["train_x"], corpus_dict["input_corpus_dict"], max_len)
    train_output = create_padded_tensor(data_dict["train_y"], corpus_dict["output_corpus_dict"], max_len)

    # Create tensors for validation data
    val_input = create_padded_tensor(data_dict["val_x"], corpus_dict["input_corpus_dict"], max_len)
    val_output = create_padded_tensor(data_dict["val_y"], corpus_dict["output_corpus_dict"], max_len)

    # Create tensors for testing data
    test_input = create_padded_tensor(data_dict["test_x"], corpus_dict["input_corpus_dict"], max_len)
    test_output = create_padded_tensor(data_dict["test_y"], corpus_dict["output_corpus_dict"], max_len)

    # Return dictionary containing tensors
    return {
        "train_input": train_input,
        "train_output": train_output,
        "val_input": val_input,
        "val_output": val_output,
        "test_input" : test_input,
        "test_output" : test_output
    }


def preprocess_data(datapath:str, lang : str):
    dictionary1 = load_data(datapath, lang)
    dictionary2 = create_corpus(dictionary1)
    dictionary3 = create_tensor(dictionary1, dictionary2) 
    dictionary4 = {
        "train_input": dictionary3["train_input"],
        "train_output": dictionary3["train_output"],
        "val_input": dictionary3["val_input"],
        "val_output": dictionary3["val_output"],
        "test_input" : dictionary3["test_input"],
        "test_output" : dictionary3["test_output"],
        "input_corpus_length" : dictionary2["input_corpus_length"],
        "output_corpus_length" : dictionary2["output_corpus_length"],
        "input_corpus_dict" : dictionary2["input_corpus_dict"],
        "output_corpus_dict" : dictionary2["output_corpus_dict"],
        "reversed_input_corpus" : dictionary2["reversed_input_corpus"],
        "reversed_output_corpus" : dictionary2["reversed_output_corpus"],
        "train_x" : dictionary1["train_x"],
        "train_y" : dictionary1["train_y"],
        "val_x" : dictionary1["val_x"],
        "val_y" : dictionary1["val_y"],
        "test_x" : dictionary1["test_x"],
        "test_y" : dictionary1["test_y"],
        "max_decoder_length" : dictionary1["max_decoder_length"],
        "max_encoder_length" : dictionary1["max_encoder_length"]
    }   

    return dictionary4

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
    
    def dot_score(self, hidden_state, encoder_state):
        """
        Calculate dot product attention scores.
        Args:
            hidden_state (Tensor): The hidden state of the decoder. Shape: (batch_size, 1, hidden_size)
            encoder_state (Tensor): The output of the encoder. Shape: (batch_size, max_length, hidden_size)
        Returns:
            scores (Tensor): Attention scores. Shape: (batch_size, 1, max_length)
        """
        # Compute dot product between decoder hidden state and encoder output
        scores = torch.sum(hidden_state * encoder_state, dim=2)
        return scores

    def forward(self, hidden, encoder_output):
        """
        Forward pass of the attention mechanism.
        Args:
            hidden (Tensor): The hidden state of the decoder. Shape: (batch_size, 1, hidden_size)
            encoder_output (Tensor): The output of the encoder. Shape: (batch_size, max_length, hidden_size)
        Returns:
            attention_weights (Tensor): Attention weights over encoder outputs. Shape: (batch_size, 1, max_length)
        """
        # Calculate attention scores using dot product
        scores = self.dot_score(hidden, encoder_output)
        
        # Transpose the scores for softmax computation
        scores = scores.t()
        
        # Apply softmax to get attention weights and add an extra dimension for compatibility
        attention_weights = F.softmax(scores, dim=1).unsqueeze(1)
        
        return attention_weights
    

class Encoder(nn.Module):
    """
    Encoder class for sequence-to-sequence model.

    Args:
        PARAM (dict): A dictionary containing encoder parameters.
            - "encoder_input_size" (int): Size of the encoder input.
            - "embedding_size" (int): Size of the embedding.
            - "hidden_size" (int): Size of the hidden state.
            - "num_layers" (int): Number of layers.
            - "drop_prob" (float): Dropout probability.
            - "cell_type" (str): Type of RNN cell (LSTM, GRU, RNN).
            - "bidirectional" (bool): Whether the RNN is bidirectional.
    """

    def __init__(self, PARAM):
        super(Encoder, self).__init__()
        self.input_size = PARAM["encoder_input_size"]
        self.embedding_size = PARAM["embedding_size"]
        self.hidden_size = PARAM["hidden_size"]
        self.num_layers = PARAM["num_layers"]
        self.drop_prob = PARAM["drop_prob"]
        self.cell_type = PARAM["cell_type"]
        self.bidirectional = PARAM["bidirectional"]

        self.dropout = nn.Dropout(self.drop_prob)
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)

        cell_map = {
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
            "RNN": nn.RNN
        }
        self.cell = cell_map[self.cell_type](
            self.embedding_size, self.hidden_size, self.num_layers,
            dropout=self.drop_prob, bidirectional=self.bidirectional
        )
    
    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoder states.
            torch.Tensor: Hidden state (if cell type is RNN or GRU).
            torch.Tensor: Cell state (if cell type is LSTM).
        """
        embedding = self.embedding(x)
        drops = self.dropout(embedding)
        if self.cell_type == "RNN" or self.cell_type == "GRU":
            encoder_states, hidden = self.cell(drops)
            if self.bidirectional:
                encoder_states = encoder_states[:, :, :self.hidden_size] + encoder_states[:, :, self.hidden_size:]
            return encoder_states, hidden
        elif self.cell_type == "LSTM":
            encoder_states, (hidden, cell) = self.cell(drops)
            if self.bidirectional:
                encoder_states = encoder_states[:, :, :self.hidden_size] + encoder_states[:, :, self.hidden_size:]
            return encoder_states, hidden, cell

        
class Decoder(nn.Module):
    """
    Decoder class for sequence-to-sequence model with attention mechanism.

    Args:
        params (dict): A dictionary containing decoder parameters.
            - "decoder_input_size" (int): Size of the decoder input.
            - "embedding_size" (int): Size of the embedding.
            - "hidden_size" (int): Size of the hidden state.
            - "decoder_output_size" (int): Size of the decoder output.
            - "num_layers" (int): Number of layers.
            - "drop_prob" (float): Dropout probability.
            - "cell_type" (str): Type of RNN cell (LSTM, GRU, RNN).
            - "bidirectional" (bool): Whether the RNN is bidirectional.
    """

    def __init__(self, params):
        super(Decoder, self).__init__()

        # Initialize decoder parameters
        self.input_size = params["decoder_input_size"]
        self.embedding_size = params["embedding_size"]
        self.hidden_size = params["hidden_size"]
        self.output_size = params["decoder_output_size"]
        self.num_layers = params["num_layers"]
        self.drop_prob = params["drop_prob"]
        self.cell_type = params["cell_type"]
        self.bidirectional = params["bidirectional"]

        # Dropout layer
        self.dropout = nn.Dropout(self.drop_prob)

        # Embedding layer
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)

        # Linear layer for combining context and decoder output
        self.concatlayer = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # Final output layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # Apply LogSoftmax for probability distribution
        self.log_softmax = nn.LogSoftmax(dim=1)

        # Attention layer
        self.attn = Attention(self.num_layers)

        # Select appropriate RNN cell based on cell_type parameter
        self.cell_map = {
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
            "RNN": nn.RNN
        }
        self.cell = self.cell_map[self.cell_type](
            self.embedding_size, self.hidden_size, self.num_layers,
            dropout=self.drop_prob
        )

    def forward(self, x, encoder_states, hidden, cell):
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): Input tensor.
            encoder_states (torch.Tensor): Encoder states.
            hidden (torch.Tensor): Hidden state.
            cell (torch.Tensor): Cell state (if cell type is LSTM).

        Returns:
            torch.Tensor: Predictions.
            torch.Tensor: Hidden state.
            torch.Tensor: Cell state (if cell type is LSTM).
            torch.Tensor: Attention weights.
        """
        # Embed input word
        x = x.unsqueeze(0)
        embedding = self.embedding(x)
        drops = self.dropout(embedding)

        # Pass through RNN cell based on type
        if self.cell_type == "RNN" or self.cell_type == "GRU":
            outputs, (hidden) = self.cell(drops, hidden)
        elif self.cell_type == "LSTM":
            outputs, (hidden, cell) = self.cell(drops, (hidden, cell))
        else:
            raise ValueError("Invalid cell type during forward pass: {}".format(self.cell_type))

        # Attention mechanism
        attention_weights = self.attn(outputs, encoder_states)
        context = attention_weights.bmm(encoder_states.transpose(0, 1))

        # Squeeze outputs and context
        outputs = outputs.squeeze(0)
        context = context.squeeze(1)

        # Concatenate decoder output and context
        concat_input = torch.cat((outputs, context), 1)

        # Apply activation and linear layer
        concat_output = torch.tanh(self.concatlayer(concat_input))
        predictions = self.log_softmax(self.fc(concat_output))

        # Return predictions, hidden state (and cell for LSTM) and attention weights
        if self.cell_type == "LSTM":
            return predictions, hidden, cell, attention_weights.squeeze(1)
        else:
            return predictions, hidden, attention_weights.squeeze(1)


class Seq2Seq(nn.Module):
    """
    Sequence-to-sequence model for translation tasks.

    Args:
        encoder: The encoder module.
        decoder: The decoder module.
        params (dict): A dictionary containing model parameters.
        processed_data (dict): A dictionary containing processed data needed for translation.
    """

    def __init__(self, encoder, decoder, params, processed_data):
        super(Seq2Seq, self).__init__()
        self.cell_type = params["cell_type"]
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = params["tfr"]
        self.target_vocab_size = processed_data['output_corpus_length']

    def forward(self, src, target):
        """
        Forward pass of the Seq2Seq model.

        Args:
            src (torch.Tensor): Source input tensor.
            target (torch.Tensor): Target output tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        batch_size = src.shape[1]
        target_len = target.shape[0]
        x = target[0, :]
        target_vocab_size = self.target_vocab_size
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # Get encoder output and hidden state (or cell for LSTM)
        if self.cell_type == "LSTM":
            encoder_output, hidden, cell = self.encoder(src)
            cell = cell[:self.decoder.num_layers]
        else:
            encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.num_layers]

        # Iterate through target sequence
        for i in range(1, target_len):
            # Decode using previous word or teacher forcing
            if self.cell_type == "LSTM":
                output, hidden, cell, _ = self.decoder(x, encoder_output, hidden, cell)
            else:
                output, hidden, _ = self.decoder(x, encoder_output, hidden, None)

            outputs[i] = output

            # Use teacher forcing with a probability
            if random.random() < self.teacher_forcing_ratio:
                next_word = target[i]
            else:
                next_word = output.argmax(dim=1)

            # Update input word for next iteration
            x = next_word

        return outputs

 
def set_optimizer(name, model, learning_rate):
    """
    Creates an optimizer object based on the specified name and learning rate.
    Args:
        name (str): Name of the optimizer (e.g., "adam", "sgd", "rmsprop", "adagrad").
        model (nn.Module): The PyTorch model to be optimized.
        learning_rate (float): The learning rate to use for training.
    Returns:
        torch.optim.Optimizer: The created optimizer object.
    """

    # Define the optimizer based on the provided name
    optimizer = None
    if name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif name == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    else:
        # Raise an error if the optimizer name is invalid
        raise ValueError(f"Invalid optimizer name: {name}")

    # Ensure an optimizer was created
    if optimizer is None:
        raise ValueError("Failed to create optimizer. Please check the provided name.")

    return optimizer


def beam_search(PARAM, model, word, device, processed_data):
    """
    Performs beam search to generate a sequence of output tokens.

    Args:
        PARAM (dict): A dictionary containing model parameters.
        model: The Seq2Seq model.
        word (str): Input word to be translated.
        device: The device on which the model runs.
        processed_data (dict): A dictionary containing processed data needed for translation.

    Returns:
        str: Translated sentence.
    """
    # Unpack processed data
    input_corpus_dict = processed_data["input_corpus_dict"]
    output_corpus_dict = processed_data["output_corpus_dict"]
    max_encoder_length = processed_data["max_encoder_length"]
    reversed_output_corpus = processed_data["reversed_output_corpus"]

    # Initialize data tensor for input word
    data = np.zeros((max_encoder_length + 1, 1), dtype=np.int32)
    for i, char in enumerate(word):
        data[i, 0] = input_corpus_dict[char]
    data[i + 1, 0] = input_corpus_dict['$']  # Add end-of-sentence marker
    data = torch.tensor(data, dtype=torch.int32).to(device)

    # Encode input word
    with torch.no_grad():
        if PARAM["cell_type"] == "LSTM":
            outputs, hidden, cell = model.encoder(data)
            cell = cell[:PARAM['num_layers']]
        elif PARAM["cell_type"] == "GRU" or PARAM["cell_type"] == "RNN":
            outputs, hidden = model.encoder(data)
    hidden = hidden[:PARAM['num_layers']]

    # Initialize beam search
    output_start = output_corpus_dict['#']
    start_token = np.array(output_start).reshape(1,)
    hidden_par = hidden.unsqueeze(0)
    initial_sequence = torch.tensor(start_token).to(device)
    beam = [(0.0, initial_sequence, hidden_par)]

    # Perform beam search
    for i in range(len(output_corpus_dict)):
        candidates = []
        for score, seq, hidden in beam:
            if seq[-1].item() == output_corpus_dict['$']:
                candidates.append((score, seq, hidden))
                continue
            
            reshape_last = np.array(seq[-1].item()).reshape(1, )
            hdn = hidden.squeeze(0) 
            x = torch.tensor(reshape_last).to(device)
            if PARAM["cell_type"] == "LSTM":
                output, hidden, cell, attention_wt = model.decoder(x, outputs, hdn, cell)
            if PARAM["cell_type"] == "RNN" or PARAM["cell_type"] == "GRU":
                output, hidden, attention_wt = model.decoder(x, outputs, hdn, None)
            
            probabilities = F.softmax(output, dim=1)
            topk_probs, topk_tokens = torch.topk(probabilities, k=PARAM["beam_width"])
    
            for prob, token in zip(topk_probs[0], topk_tokens[0]):
                new_seq = torch.cat((seq, token.unsqueeze(0)), dim=0)
                ln_ns = len(new_seq)
                ln_pf = ((ln_ns - 1) / 5)
                candidate_score = score + torch.log(prob).item() / (ln_pf ** PARAM["length_penalty"])
                candidates.append((candidate_score, new_seq, hidden.unsqueeze(0)))
    
        beam = heapq.nlargest(PARAM["beam_width"], candidates, key=lambda x: x[0])

    # Retrieve best sequence and decode
    best_score, best_sequence, _ = max(beam, key=lambda x: x[0]) 
    translated_sentence = ''.join([reversed_output_corpus[token.item()] for token in best_sequence[1:]])[:-1]

    return translated_sentence


def run_epoch(model, data_loader, optimizer, criterion, processed_data):
    """
    Train the Seq2Seq model for one epoch.

    Args:
        model (nn.Module): Seq2Seq model to train.
        data_loader (List): List containing training_data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function for calculating training loss.

    Returns:
        tuple(float, float): Training accuracy and average loss.
    """

    model.train()  # Set model to training mode
    total_loss, total_words, correct_predictions = 0, 0, 0

    with tqdm(total=len(data_loader[0]), desc='Training') as pbar:  # Gradient accumulation
        for _ , (source, target) in enumerate(zip(data_loader[0], data_loader[1])):
            source, target = source.to(device), target.to(device)  # Move data to device
            optimizer.zero_grad()

            # Forward pass
            output = model(source, target)
            target = target.reshape(-1)  # Reshape target for loss calculation
            output = output.reshape(-1, output.shape[2])  # Reshape output
            
            #Ignore the padding
            pad_mask = (target != processed_data['output_corpus_dict'][''])
            target = target[pad_mask]
            output = output[pad_mask]

            # Calculate loss
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()  # Update model parameters

            # Calculate total loss, total words, correct_predictions
            total_loss += loss.item()
            total_words += target.size(0)
            correct_predictions += torch.sum(torch.argmax(output, dim = 1) == target).item()
            pbar.update(1)

    # Calculate Accuracy and Avg Loss
    accuracy = correct_predictions / total_words
    avg_loss = total_loss / len(data_loader[0])

    return accuracy, avg_loss


def evaluate_character_level(model, val_data_loader, loss_fn, processed_data):
    """
    Evaluate the Seq2Seq model on character-level data.

    Args:
        model (nn.Module): Seq2Seq model to evaluate.
        val_data_loader (DataLoader): Data loader for validation data.
        loss_fn (nn.Module): Loss function for calculating validation loss.

    Returns:
        tuple(float, float): Validation accuracy and average loss.
    """

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        total_loss = 0
        total_words = 0
        correct_predictions = 0

        with tqdm(total=len(val_data_loader[0]), desc='Validation') as pbar:
            for src, tar in zip(val_data_loader[0], val_data_loader[1]):
                target, source = tar.to(device), src.to(device)

                # Apply model
                output = model(source, target)

                # Reshape target and output
                target = target.reshape(-1)
                output = output.reshape(-1, output.shape[2])
                
                # Ignore the padding 
                pad_mask = (target != processed_data['output_corpus_dict'][''])
                target = target[pad_mask]
                output = output[pad_mask]

                #Calculate total_loss, total_words, correct_predictions
                val_loss = loss_fn(output, target)
                total_loss += val_loss.item()
                total_words += target.size(0)
                correct_predictions += torch.sum(torch.argmax(output, dim=1) == target).item()
                pbar.update(1)
        
    accuracy = correct_predictions / total_words
    avg_loss = total_loss / len(val_data_loader[0])

    return accuracy, avg_loss


def evaluate_model_beam_search(params, model, device, processed_data):
    """
    Evaluates the model using beam search and returns accuracy and correct predictions.

    Args:
        model (torch.nn.Module): The machine translation model to evaluate.
        val_data (torch.Tensor): The validation data tensor.
        vx (list): List of source words for beam search.
        vy (list): List of target words for beam search.
        device (str): Device to use for computation (e.g., 'cpu' or 'cuda').
        processed_data (dict): Preprocessed data dictionary.

    Returns:
        tuple: A tuple containing validation accuracy (float) and correct predictions (int).
    """

# Set the model to evaluation mode
    model.eval()

    # Disable gradient computation during inference
    with torch.no_grad():
        # Initialize counters
        total_words = 0
        correct_predictions = 0
        
        # Iterate through the validation data with tqdm progress bar
        with tqdm(total=len(processed_data["val_x"]), desc='Beam_Search') as pbar:
            for word, target_word in zip(processed_data["val_x"], processed_data["val_y"]):
                # Increment the total words counter
                total_words += 1
                
                # Perform beam search to predict the next word
                predicted_word = beam_search(params, model, word, device, processed_data)
                
                # Check if the predicted word matches the target word
                if predicted_word == target_word[1:-1]:  # Remove start and end tokens
                    correct_predictions += 1
                
                # Update the progress bar
                pbar.update(1)

    # Calculate accuracy
    accuracy = correct_predictions / total_words

    # Return accuracy and number of correct predictions
    return accuracy, correct_predictions


def training(PARAM, processed_data, device, wandb_log = 0):
    #initialize wandb with project
    if wandb_log == 1:
        wandb.init(project='DL-Assignment3')
        wandb.run.name = 'Testing'
    
    # Set learning Rate, epochs, batch_size
    learning_rate = PARAM["learning_rate"]
    epochs = PARAM["epochs"]
    batch_size = PARAM["batch_size"]

    #  copy encoder and decoder to device
    encoder = Encoder(PARAM).to(device)
    decoder = Decoder(PARAM).to(device)

#     # Initialize model
    model = Seq2Seq(encoder, decoder, PARAM, processed_data).to(device)
    print(model)

    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = set_optimizer(PARAM["optimizer"], model, learning_rate)

    # Split dataset into batches
    train_batches_x = torch.split(processed_data["train_input"], batch_size, dim=1)
    train_batches_y = torch.split(processed_data["train_output"], batch_size, dim=1)
    val_batches_x = torch.split(processed_data["val_input"], batch_size, dim=1)
    val_batches_y = torch.split(processed_data["val_output"], batch_size, dim=1)

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch :: {epoch+1}/{epochs}")
        
        # Train the model on training data
        data_loader = [train_batches_x, train_batches_y]
        accuracy, avg_loss = run_epoch(model, data_loader, optimizer, loss_function, processed_data)  # Average loss per batch

        # Evaluate model character wise
        val_data_loader = [val_batches_x, val_batches_y]
        val_accuracy, val_avg_loss = evaluate_character_level(model, val_data_loader, loss_function, processed_data)

        # Evaluate model word wise
        val_accuracy_beam, val_correct_pred_beam = evaluate_model_beam_search(PARAM, model, device, processed_data)
        total_words = processed_data["val_input"].shape[1] 

        print(f"Epoch : {epoch+1} Train Accuracy: {accuracy*100:.4f}, Train Loss: {avg_loss:.4f}\nValidation Accuracy: {val_accuracy*100:.4f}, Validation Loss: {val_avg_loss:.4f}, \nValidation Acc. With BeamSearch: {val_accuracy_beam*100:.4f}, Correctly Predicted : {val_correct_pred_beam}/{total_words}")

        # Log on wandb
        if wandb_log:
            wandb.log(
                    {
                        'epoch': epoch+1,
                        'training_loss' : avg_loss,
                        'training_accuracy' : accuracy,
                        'validation_loss' : val_avg_loss,
                        'validation_accuracy_using_char' : val_accuracy,
                        'validation_accuracy_using_word' : val_accuracy_beam,
                        'correctly_predicted' : val_correct_pred_beam
                    }
                )
    return model, val_accuracy_beam

# Function to get argument from command line
def get_hyper_perameters(arguments, processed_data):
    HYPER_PARAM = {
        "encoder_input_size": processed_data["input_corpus_length"],
        "embedding_size": arguments.embadding_size,
        "hidden_size": arguments.hidden_size,
        "num_layers": arguments.num_layers,
        "drop_prob": arguments.dropout,
        "cell_type": arguments.cell_type,
        "decoder_input_size": processed_data["output_corpus_length"],
        "decoder_output_size": processed_data["output_corpus_length"],
        "beam_width" : arguments.beam_width,
        "length_penalty" : arguments.length_penalty,
        "bidirectional" : True if arguments.bidirectional else False,
        "learning_rate" : arguments.learning_rate,
        "batch_size" : arguments.batch_size,
        "epochs" : arguments.epochs,
        "optimizer" : arguments.optimizer,
        "tfr" : arguments.teacher_forcing_ratio,
    }

    return HYPER_PARAM

# Function to get model accuracy and loss on test data
def evaluate_model(params, model, device, processed_data):
    """
    Evaluates the model using beam search on test data and returns accuracy and correct predictions.

    Args:
        params : Hyper Parameters used for parameters
        model (torch.nn.Module): The machine translation model to evaluate.
        device (str): Device to use for computation (e.g., 'cpu' or 'cuda').
        processed_data (dict): Preprocessed data dictionary.

    Returns:
        tuple: A tuple containing test accuracy (float) and correct predictions (int).
    """

# Set the model to evaluation mode
    model.eval()

    # Disable gradient computation during inference
    with torch.no_grad():
        # Initialize counters
        total_words = 0
        correct_predictions = 0
        
        # Iterate through the validation data with tqdm progress bar
        with tqdm(total=len(processed_data["test_x"]), desc='Evaluating Model') as pbar:
            for word, target_word in zip(processed_data["test_x"], processed_data["test_y"]):
                # Increment the total words counter
                total_words += 1
                
                # Perform beam search to predict the next word
                predicted_word = beam_search(params, model, word, device, processed_data)
#                 print(target_word, predicted_word)
                # Check if the predicted word matches the target word
                if predicted_word == target_word[1:-1]:  # Remove start and end tokens
                    correct_predictions += 1
                
                # Update the progress bar
                pbar.update(1)

    # Calculate accuracy
    accuracy = correct_predictions / total_words

    # Return accuracy and number of correct predictions
    return accuracy, correct_predictions

# Function and their helpler function that print 10 random translated data from test data 
def encode_input(word, processed_data):
    """
    Encode the input word into a tensor representation.

    Args:
        word (str): Input word to be encoded.
        input_corpus_dict (dict): Dictionary mapping characters to indices.

    Returns:
        data (torch.Tensor): Encoded tensor representation of the input word.
    """
    max_encoder_length = processed_data["max_encoder_length"]
    input_corpus_dict = processed_data["input_corpus_dict"]
    data = np.zeros((max_encoder_length + 1,1), dtype= int)

    for i, char in enumerate(word):
        data[i, 0] = input_corpus_dict[char]
    data[i + 1, 0] = input_corpus_dict['$']  # Add end-of-sentence marker
    
    data = torch.tensor(data,dtype = torch.int64).to(device)
    return data

def generate_predictions(model, word, PARAM, device, processed_data):
    """
    Generate prediction based on the encoder outputs using the provided model.

    Args:
        model (torch.nn.Module): The trained model for prediction.
        outputs (torch.Tensor): Encoder outputs.
        hidden (torch.Tensor): Hidden state of the encoder.
        cell (torch.Tensor): Cell state of the encoder (for LSTM).
        PARAM (dict): Dictionary containing model parameters.
        device (torch.device): Device on which the model will run (e.g., CPU or GPU).
        processed_data (dict): Dictionary containing preprocessed data.

    Returns:
        pred (str): Predicted output sequence.
        attentions (torch.Tensor): Attention weights for each decoding step.
    """
    input_corpus_dict = processed_data["input_corpus_dict"]
    output_corpus_dict = processed_data["output_corpus_dict"]
    reversed_output_corpus = processed_data["reversed_output_corpus"]
    max_encoder_length = processed_data["max_encoder_length"]
    data = encode_input(word, processed_data).to(device)
    
    outputs,hidden,cell=None,None,None
    with torch.no_grad():
        if PARAM['cell_type'] == 'LSTM':
            outputs, hidden, cell = model.encoder(data)
            cell =  cell[:PARAM['num_layers']]
        else:
            outputs, hidden = model.encoder(data)
    hidden =  hidden[:PARAM['num_layers']]
    
    # Initialize decoder input with start-of-sequence marker
    x = torch.tensor([output_corpus_dict['#']]).to(device)

    attentions = torch.zeros(max_encoder_length + 1, 1, max_encoder_length + 1)

    # Decode output sequence
    pred = ""
    for t in range(1, len(output_corpus_dict)):
        if PARAM['cell_type'] == 'LSTM':
            output, hidden, cell, attn = model.decoder(x, outputs, hidden, cell)
        else:
            output, hidden, attn = model.decoder(x, outputs, hidden, None)

        # Get predicted character
        character = reversed_output_corpus[output.argmax(1).item()]
        attentions[t] = attn

        # Append character to prediction
        if character != '$':
            pred += character
        else:
            break

        # Update decoder input
        x = torch.tensor([output.argmax(1)]).to(device)

    # Return prediction and attention weights
    return pred, attentions[:t+1]

def random_test_words(processed_data, model, HYPER_PARAM, device):
    """
    Generate predictions and attention maps for a random set of test words.

    Args:
        processed_data (dict): Dictionary containing preprocessed test data.
        model (torch.nn.Module): The trained model for prediction.
        HYPER_PARAM (dict): Dictionary containing model hyperparameters.
        device (torch.device): Device on which the model will run (e.g., CPU or GPU).

    Returns:
        translation_dict (dict): Dictionary containing word-to-predicted-translation mapping.
        attention_dict (dict): Dictionary containing attention matrices for each translation.
    """
    random_words = random.sample(list(processed_data["test_x"]), 10)
    translation_dict, attention_dict = {}, {}
    
    for word_pair in random_words:
        # Get prediction and attention for each word
        input_word = word_pair[:-1]  # Remove end-of-sentence marker
        pred, attention = generate_predictions(model, input_word, HYPER_PARAM, device, processed_data)
        
        # Store translation and attention for the word
        translation_dict[input_word] = ' ' + pred
        attention_dict[input_word] = attention
    
    return translation_dict, attention_dict

# Main Function
if __name__ == "__main__":
    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Get arguments from command line
    arguments = parser.parse_args()

    # Function call set avilable device (GPU/ CPU)
    device = set_device()

    # Get Data
    processed_data = preprocess_data(arguments.datapath, arguments.lang)

    # Hyper Parameter Dict 
    params = get_hyper_perameters(arguments, processed_data)
    print(params)

    # Train the Model 
    model, acc = training(params, processed_data, device, wandb_log = arguments.wandb_log)

    # Evaluate Model 
    if arguments.evaluate:
        accuracy, correct_pred = evaluate_model(params, model, device, processed_data)
        total_words = len(processed_data["test_x"])
        msg = f"Test Accuracy : {accuracy*100:.4f}, Correct_pred : {correct_pred}/{total_words}"
        print(msg)

    if arguments.translate_random:
        print("10 Random Words Translated from Test Data")
        translation_dict, _ = random_test_words(processed_data, model, params, device)
        for key in translation_dict.keys():
            msg = f"{key} ==> {translation_dict[key]}"
            print(msg)
    
    
