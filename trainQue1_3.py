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
parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='CS6910-Assignment3')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='cs23m026')
parser.add_argument('-d', '--datapath', help='give data path e.g. /kaggle/input/vocabs/Dataset', type=str, default='/kaggle/input/vocabs/Dataset')
parser.add_argument('-l', '--lang', help='language', type=str, default='hin')
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


class Encoder(nn.Module):
    """
    Encoder class for sequence-to-sequence models.
    Args:
        PARAM (dict): Encoder hyperparameters.
            - input_size (int): Size of the input vocabulary.
            - embedding_size (int): Dimensionality of word embeddings.
            - hidden_size (int): Size of the hidden state in RNN cells.
            - num_layers (int): Number of stacked RNN layers.
            - drop_prob (float): Dropout probability for regularization.
            - cell_type (str): Type of RNN cell (LSTM, GRU, RNN).
            - bidirectional (bool): Whether to use a bidirectional RNN.
    """

    def __init__(self, PARAM):
        super(Encoder, self).__init__()

        # Hyperparameters
        self.input_size = PARAM["encoder_input_size"]
        self.embedding_size = PARAM["embedding_size"]
        self.hidden_size = PARAM["hidden_size"]
        self.num_layers = PARAM["num_layers"]
        self.drop_prob = PARAM["drop_prob"]
        self.cell_type = PARAM["cell_type"]
        self.bidirectional = PARAM["bidirectional"]

        # Layers
        self.dropout = nn.Dropout(self.drop_prob)
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)

        # Select RNN cell based on cell_type
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
        Forward pass of the Encoder.
        Args:
            x : Input sequence of word indices.
        Returns:
            torch.Tensor or tuple : Hidden state (or hidden & cell states for LSTMs)
        """

        embedding = self.embedding(x) # embadding layer 
        drops = self.dropout(embedding) # Dropout on embadding 
        if self.cell_type == "RNN" or self.cell_type == "GRU": 
            _, hidden = self.cell(drops) 
            return hidden
        elif self.cell_type == "LSTM":
            _, (hidden, cells) = self.cell(drops)
            return hidden, cells
        else:
            raise ValueError(f"Invalid RNN cell type: {self.cell_type}") # Raise a error on invalid cell type


class Decoder(nn.Module):
    """
    Decoder class for sequence-to-sequence models.

    Args:
        PARAM (dict): Decoder hyperparameters.
            - input_size (int): Size of the decoder vocabulary.
            - embedding_size (int): Dimensionality of word embeddings.
            - hidden_size (int): Size of the hidden state in RNN cells.
            - output_size (int): Size of the output vocabulary.
            - num_layers (int): Number of stacked RNN layers.
            - drop_prob (float): Dropout probability for regularization.
            - cell_type (str): Type of RNN cell (LSTM, GRU, RNN).
            - bidirectional (bool): Whether to use a bidirectional RNN.
    """

    def __init__(self, PARAM):
        super(Decoder, self).__init__()

        # Hyperparameters
        self.input_size = PARAM["decoder_input_size"]
        self.embedding_size = PARAM["embedding_size"]
        self.hidden_size = PARAM["hidden_size"]
        self.output_size = PARAM["decoder_output_size"]
        self.num_layers = PARAM["num_layers"]
        self.drop_prob = PARAM["drop_prob"]
        self.cell_type = PARAM["cell_type"]
        self.bidirectional = PARAM["bidirectional"]

        # Layers
        self.dropout = nn.Dropout(self.drop_prob)
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.cell_map = {
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
            "RNN": nn.RNN
        }
        self.cell = self.cell_map[self.cell_type](
            self.embedding_size, self.hidden_size, self.num_layers,
            dropout=self.drop_prob, bidirectional=self.bidirectional
        )

        # Final linear layer for output prediction
        self.fc = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), self.output_size)

    def forward(self, x, hidden, cell=None):
        """
        Forward pass of the Decoder.

        Args:
            x (torch.Tensor): Input sequence of word indices (single token for teacher forcing).
            hidden (torch.Tensor): Hidden state from the encoder.
            cell (torch.Tensor, optional): Cell state for LSTMs (default: None).

        Returns:
            tuple(torch.Tensor): Predicted output logits, hidden state (and cell state for LSTMs).
        """

        x = x.unsqueeze(0)  # Add batch dimension for single token
        embedding = self.embedding(x)
        drops = self.dropout(embedding)

        if self.cell_type == "RNN" or self.cell_type == "GRU":
            outputs, hidden = self.cell(drops, hidden)
        elif self.cell_type == "LSTM":
            outputs, (hidden, cell) = self.cell(drops, (hidden, cell))
        predictions = self.fc(outputs).squeeze(0)  # Remove batch dimension

        if self.cell_type == "LSTM":
            predictions = F.log_softmax(predictions, dim=1)
            return predictions, hidden, cell
        return predictions, hidden


class Seq2Seq(nn.Module):
    """
    Seq2Seq model for sequence-to-sequence tasks.

    Args:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        param (dict): Model hyperparameters.
            - tfr (float): Teacher forcing ratio for training.
        processed_data (dict) : containing all information of processed data
    """

    def __init__(self, encoder, decoder, param, p_data):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = param["tfr"]  # Teacher forcing ratio
        self.processed_data = p_data

    def forward(self, src, target):
        """
        Forward pass of the Seq2Seq model.

        Args:
            src (torch.Tensor): Source sequence of word indices.
            target (torch.Tensor): Target sequence of word indices.

        Returns:
            torch.Tensor: Predicted output logits for each target word.
        """

        batch_size = src.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.processed_data["output_corpus_length"]

        # Initialize outputs tensor
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # Get encoder hidden state(s)
        if self.encoder.cell_type == "LSTM":
            encoder_hidden, cell = self.encoder(src)
        elif self.encoder.cell_type == "GRU" or self.encoder.cell_type == "RNN":
            encoder_hidden = self.encoder(src)

        # Start with first target word
        x = target[0]

        for t in range(1, target_len):
            # Decode with teacher forcing or predicted output
            if self.encoder.cell_type == "LSTM":
                y, encoder_hidden, cell = self.decoder(x, encoder_hidden, cell) 
            else:
                y, encoder_hidden = self.decoder(x, encoder_hidden, None)  

            outputs[t] = y
            if random.random() < self.teacher_forcing_ratio:
                x = target[t]
            else:
                x = y.argmax(dim=1)

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


def beam_search(params, model, word, device, processed_data):
    """
    Beam search decoding for sequence-to-sequence models.

    Args:
        params (dict): Model hyperparameters.
            - encoder_cell_type (str): Type of RNN cell (LSTM, GRU, RNN).
            - beam_width (int): Beam width for beam search decoding.
            - length_penalty (float): Penalty for longer sequences.
        model (nn.Module): Seq2Seq model for sequence translation.
        word (str): Input word to translate.
        device (torch.device): Device to use for computations (CPU or GPU).
        max_encoder_length (int): Maximum length of the encoder input sequence.
        input_corpus_dict (dict): Dictionary mapping input characters to integer indices.
        output_corpus_dict (dict): Dictionary mapping integer indices to output characters.
        reverse_output_corpus (dict): Dictionary mapping output characters to integer indices (for reversing prediction).

    Returns:
        str: Translated sentence.
    """

    input_corpus_dict = processed_data["input_corpus_dict"]
    output_corpus_dict = processed_data["output_corpus_dict"]
    max_encoder_length = processed_data["max_encoder_length"]
    reversed_output_corpus = processed_data["reversed_output_corpus"]
    # Preprocess input sentence
    data = torch.zeros((max_encoder_length + 1, 1), dtype=torch.int32).to(device)
    for i, char in enumerate(word):
        data[i, 0] = input_corpus_dict[char]
    data[i + 1, 0] = input_corpus_dict['$']  # Add end-of-sentence marker

    # Encode input sentence
    with torch.no_grad():
        if params["cell_type"] == "LSTM":
            hidden, cell = model.encoder(data)
        else:
            hidden = model.encoder(data)

        # Initialize beam search
        start_token = output_corpus_dict['#']  # Start-of-sentence symbol
        initial_sequence = torch.tensor([start_token]).to(device)
        hidden = hidden.unsqueeze(0)  # Add batch dimension
        beam = [(0.0, initial_sequence, hidden)]  # List of (score, sequence, hidden state) tuples

    # Decode loop
        for _ in range(len(output_corpus_dict)):
            candidates = []  # List for storing candidate sequences
            for score, seq, hidden in beam:
                # Check for end-of-sentence token
                if seq[-1].item() == output_corpus_dict['$']:
                    candidates.append((score, seq, hidden))
                    continue

                # Get last token and hidden state
                last_token = seq[-1].unsqueeze(0).to(device)
                hidden = hidden.squeeze(0)

                # Decode step with last token
                if params["cell_type"] == "LSTM":
                    output, hidden, cell = model.decoder(last_token, hidden, cell)
                else:
                    output, hidden = model.decoder(last_token, hidden, None)

            # Get top-k probable tokens
                probabilities = F.softmax(output, dim=1)
                topk_probs, topk_tokens = torch.topk(probabilities, k=params["beam_width"])

                # Expand beam with top-k candidate sequences
                for prob, token in zip(topk_probs[0], topk_tokens[0]):
                    new_seq = torch.cat((seq, token.unsqueeze(0)), dim=0)
                    length_penalty = ((len(new_seq) - 1) / 5) ** params["length_penalty"]
                    candidate_score = score + torch.log(prob).item() / length_penalty
                    candidates.append((candidate_score, new_seq, hidden.unsqueeze(0)))

            # Select top-k beam candidates for next iteration
            beam = heapq.nlargest(params["beam_width"], candidates, key=lambda x: x[0])

        # Get best sequence from beam search
        best_score, best_sequence, _ = max(beam, key=lambda x: x[0])

        # Convert predicted token indices to characters and reverse order
        translated_sentence = ''.join([reversed_output_corpus[token.item()] for token in best_sequence[1:]])[:-1]  # Remove start token and end token

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

def training(PARAM, processed_data, device, wandb_log = 0):
    # initilize wandb with project
    if wandb_log == 1:
        wandb.init(project='DL-Assignment3')
        wandb.run.name = 'Training'
    
    # Set Learning Rate, epochsm batch_size
    learning_rate = PARAM["learning_rate"]
    epochs = PARAM["epochs"]
    batch_size = PARAM["batch_size"]

    # Copy encoder and decoder to device
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

        # print epochs
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

    # Train the Model 
    model, acc = training(params, processed_data, device, wandb_log = arguments.wandb_log)

    # Evaluate Model 
    if arguments.evaluate:
        accuracy, correct_pred = evaluate_model(params, model, device, processed_data)
        total_words = len(processed_data["test_x"])
        msg = f"Test Accuracy : {accuracy}, Correct_pred : {correct_pred}/{total_words}"
        print(msg)
    
    
