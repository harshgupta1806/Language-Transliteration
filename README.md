
---

# RNN BASED Seq2Seq MODEL

This repository offers a Python implementation of a Seq2Seq model built with PyTorch. It provides flexibility by allowing you to experiment with various RNN cell architectures (LSTM, RNN, GRU) in both the encoder and decoder for sequence prediction tasks.

## Classes

### Encoder.
-   **Args:**
    -  **PARAM (dict): Encoder hyperparameter** <br />
            - `input_size (int)`: Size of the input vocabulary.<br />
            - `embedding_size (int)`: Dimensionality of word embeddings.<br />
            - `hidden_size (int)`: Size of the hidden state in RNN cells.<br />
            - `num_layers (int)`: Number of stacked RNN layers.<br />
            - `drop_prob (float)`: Dropout probability for regularization.<br />
            - `cell_type (str)`: Type of RNN cell (LSTM, GRU, RNN).<br />
            - `bidirectional (bool)`: Whether to use a bidirectional RNN.<br />
  
- **Methods**:
  - `__init__()`: Initializes the encoder.
  - `forward()`: Performs forward pass through the encoder.

### Decoder
-   **Args:**
    -   **PARAM (dict): Decoder hyperparameters**
            - `input_size (int)`: Size of the decoder vocabulary.<br />
            - `embedding_size (int)`: Dimensionality of word embeddings.<br />
            - `hidden_size (int)`: Size of the hidden state in RNN cells.<br />
            - `output_size (int)`: Size of the output vocabulary.<br />
            - `num_layers (int)`: Number of stacked RNN layers.<br />
            - `drop_prob (float)`: Dropout probability for regularization.<br />
            - `cell_type (str)`: Type of RNN cell (LSTM, GRU, RNN).<br />
            - `bidirectional (bool)`: Whether to use a bidirectional RNN.<br />
  
- **Methods**:
  - `__init__()`: Initializes the decoder.
  - `forward()`: Performs forward pass through the decoder.

### Seq2Seq
-   **Args:**
    -   `encoder (Encoder)`: Encoder module. <br />
        `decoder (Decoder)`: Decoder module.<br />
        `param (dict)`: Model hyperparameters.<br />
            - `tfr (float)`: Teacher forcing ratio for training.<br />
        `processed_data (dict)` : containing all information of processed data<br />

- **Methods**:
  - `__init__()`: Initializes the Seq2Seq model.
  - `forward()`: Performs forward pass through the model.
    
---

## Installation

To run the training script, ensure you have Python 3 installed along with the following dependencies:

- torch
- numpy
- pandas
- tqdm
- wandb
- argparse

You can install these dependencies using pip:

```bash
pip install torch numpy pandas tqdm wandb argparse
```

---

## Usage

To train the Seq2Seq model with different RNN cell types, use the `train.py` script with the following command-line arguments:

| Argument            | Description                                       | Default Value     |
|---------------------|---------------------------------------------------|-------------------|
| -wp, --wandb_project    |Project name used to track experiments in Weights & Biases dashboard | CS6910-Assignment3 |
| -we, --wandb_entity | Wandb Entity used to track experiments in the Weights & Biases dashboard. | cs23m026 |
| -d, --datapath | give data path e.g. /kaggle/input/vocabs/Dataset_Name | '/kaggle/input/vocabs/Dataset' |
| -l, --lang  | language                                       | hin               |
| -e, --epochs   | Number of epochs to train network. | 10                 |
| -b, --batch_size    | Batch size used to train network.              | 32            |
| -dp, --dropout      | dropout probablity in Ecoder & Decoder         | 0.3               |
| -nl, --num_layers      | number of layers in encoder & decoder         | 2              |
| -lr, --learning_rate| Learning rate                                     | 0.01              |
| -bw, --beam_width   | Beam Width for beam Search                       | 1                |
| -cell, --cell_type   | Cell Type of Encoder and Decoder (LSTM, RNN, GRU) | LSTM            |
| -emp_size, --embedding_size    | Embadding Size | 256 |
| -hdn_size, --hidden_size | Hidden Size                             | 512                 |
| -lp, --length_penalty | Length penalty for beam search                  | 0.6               |
| -tfr, --teacher_forcing_ratio | Teacher forcing ratio                          | 0.5        |
| -bi_dir, --bidirectional | Use bidirectional encoder                        | True          |
| -o, --optimizer    | Optimizers :- (sgd, adagrad, adam, rmsprop)| adam                |
| -p, --console | Print training Accuracy, training_loss, validations accuracy, validation_loss for every epoch, Choice :- 0 (for not printing), 1 (for printing) | 1 |
| -wl, --wandb_log | log on wandb, Choice :- 0 (for not log on wandb), 1 (for log) | 0 |
| -eval, --evaluate | print test accuracy and test loss, Choices 0 (for not evaluating), 1 (for evaluating) | 0 |

Example command to run the training script:

```bash
python train_vanilla.py -d your/dataset/path/up/to/aksharantar_sampled -l hin
```
#### Datapath contains path till parent directory of language folder in unzipped dataset (language folder must not included in datapath, seperate argument -l/--lang must used to set language folder)

---

### Output Metrices

The output metrics provided during both training and validation are as follows:

- **Training Accuracy (Character-level)**: Reflects the accuracy of predictions at the character level on the training dataset.
- **Training Average Loss**: Represents the average loss computed during the training process.
- **Validation Accuracy (Character-level)**: Indicates the accuracy of predictions at the character level on the validation dataset.
- **Validation Average Loss**: Denotes the average loss calculated during the validation phase.
- **Word Accuracy on Validation (Using Beam Search)**: Measures the accuracy of predictions at the word level on the validation data utilizing beam search.
- **Correct Predictions**: The count of accurately predicted samples out of the total validation dataset.

These metrics offer valuable insights into the performance of the Seq2Seq model throughout both training and validation stages. Character-level accuracy evaluates the precision of individual character predictions, while word-level accuracy assesses the correctness of entire output sequences.

---
---

# Seq2Seq MODEL WITH ATTENTION

Within this repository, you'll discover a Python implementation of a sequence-to-sequence (Seq2Seq) model with an attention mechanism tailored for sequence prediction tasks. Developed using PyTorch, this Seq2Seq model is equipped with an attention mechanism designed to spotlight pertinent segments of the input sequence while decoding.


