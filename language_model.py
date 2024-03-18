import random

import torch
import torch.nn as nn
import string
import time
import unidecode
import matplotlib.pyplot as plt

from utils import char_tensor, random_training_set, time_since, random_chunk, CHUNK_LEN
from evaluation import compute_bpc
from model.model import LSTM


def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden, cell = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str
    all_characters = string.printable

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, (hidden, cell) = decoder(prime_input[p], (hidden, cell)) 
    inp = prime_input[-1]

    for p in range(predict_len):
        output, (hidden, cell) = decoder(inp, (hidden, cell))

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted


def train(decoder, decoder_optimizer, inp, target):
    hidden, cell = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0
    criterion = nn.CrossEntropyLoss()

    for c in range(CHUNK_LEN):
        output, (hidden, cell) = decoder(inp[c], (hidden, cell))
        loss += criterion(output, target[c].view(1))

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / CHUNK_LEN



def tuner(n_epochs=3000, print_every=100, plot_every=10, hidden_size=128, n_layers=2,
          lr=0.005, start_string='A', prediction_length=100, temperature=0.8):
    # YOUR CODE HERE
    #     TODO:
    #         1) Implement a `tuner` that wraps over the training process (i.e. part
    #            of code that is ran with `default_train` flag) where you can
    #            adjust the hyperparameters
    #         2) This tuner will be used for `custom_train`, `plot_loss`, and
    #            `diff_temp` functions, so it should also accomodate function needed by
    #            those function (e.g. returning trained model to compute BPC and
    #            losses for plotting purpose).

    ################################### STUDENT SOLUTION #######################

    all_characters = string.printable
    n_characters = len(all_characters)
    decoder = LSTM(n_characters, hidden_size, n_characters, n_layers)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    start = time.time()
    all_losses = []
    loss_avg = 0

    for epoch in range(1, n_epochs + 1):
        loss = train(decoder, decoder_optimizer, *random_training_set())
        loss_avg += loss

        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
            print(generate(decoder, start_string, prediction_length, temperature), '\n')

        if epoch % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0

    return decoder, all_losses

######################################################################################



def plot_loss(lr_list):
    # YOUR CODE HERE
    #     TODO:
    #         1) Using `tuner()` function, train X models where X is len(lr_list),
    #         and plot the training loss of each model on the same graph.
    #         2) Don't forget to add an entry for each experiment to the legend of the graph.
    #         Each graph should contain no more than 10 experiments.
    ###################################### STUDENT SOLUTION ##########################

    plt.figure()
    for lr in lr_list:
        _, all_losses = tuner(lr=lr)

        plt.plot(all_losses, label=f'lr={lr}')

    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    ##################################################################################

def diff_temp(temp_list):
    # YOUR CODE HERE
    #     TODO:
    #         1) Using `tuner()` function, try to generate strings by using different temperature
    #         from `temp_list`.
    #         2) In order to do this, create chunks from the test set (with 200 characters length)
    #         and take first 10 characters of a randomly chosen chunk as a priming string.
    #         3) What happen with the output when you increase or decrease the temperature?
################################ STUDENT SOLUTION ################################

    TEST_PATH = './data/dickens_test.txt'
    string = unidecode.unidecode(open(TEST_PATH, 'r').read())
    chunk_len = 200
    chunks = [string[i:i + chunk_len] for i in range(0, len(string), chunk_len)]

    for temp in temp_list:

        decoder, _ = tuner(temperature=temp)

        # Select a random chunk and use the first 10 characters as a priming string
        prime_str = random.choice(chunks)[:10]

        # Generate a string using the current temperature
        generated_str = generate(decoder, prime_str=prime_str, temperature=temp)

        print(f'Temperature: {temp}, Generated String: {generated_str}')

    ##################################################################################

def custom_train(hyperparam_list):
    """
    Train model with X different set of hyperparameters, where X is 
    len(hyperparam_list).

    Args:
        hyperparam_list: list of dict of hyperparameter settings

    Returns:
        bpc_dict: dict of bpc score for each set of hyperparameters.
    """
    TEST_PATH = './data/dickens_test.txt'
    string = unidecode.unidecode(open(TEST_PATH, 'r').read())
    bpc_dict = {}

    for hyperparam in hyperparam_list:
        # Extract hyperparameters
        n_epochs = hyperparam.get('n_epochs', 3000)
        print_every = hyperparam.get('print_every', 100)
        plot_every = hyperparam.get('plot_every', 10)
        hidden_size = hyperparam.get('hidden_size', 128)
        n_layers = hyperparam.get('n_layers', 2)
        lr = hyperparam.get('lr', 0.005)
        start_string = hyperparam.get('start_string', 'A')
        prediction_length = hyperparam.get('prediction_length', 100)
        temperature = hyperparam.get('temperature', 0.8)

        # Train the model
        decoder, _ = tuner(n_epochs=n_epochs, print_every=print_every, plot_every=plot_every,
                           hidden_size=hidden_size, n_layers=n_layers, lr=lr,
                           start_string=start_string, prediction_length=prediction_length,
                           temperature=temperature)

        # Compute BPC
        bpc = compute_bpc(decoder, string)
        bpc_dict[str(hyperparam)] = bpc

    return bpc_dict
########################################################################################