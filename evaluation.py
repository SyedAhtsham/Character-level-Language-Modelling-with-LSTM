import torch
import torch.nn as nn
from utils import char_tensor

def compute_bpc(model, string):
    """
    Given a model and a string of characters, compute bits per character
    (BPC) using that model.

    Args:
        model: RNN-based model (RNN, LSTM, GRU, etc.)
        string: string of characters

    Returns:
        BPC for that set of string.
    """
    # Set the model to evaluation mode
    model.eval()

    # Convert the input string to a tensor
    input_tensor = char_tensor(string)

    # Initialize the hidden state
    hidden = model.init_hidden()

    # Use torch.no_grad() to disable gradient computation during inference
    with torch.no_grad():
        # Iterate through each character in the input string
        total_loss = 0
        for char_input, char_target in zip(input_tensor, input_tensor[1:]):
            # Forward pass
            output, hidden = model(char_input.view(1, -1), hidden)

            # Compute cross-entropy loss
            loss = nn.CrossEntropyLoss()(output.squeeze(), char_target)

            # Accumulate the total loss
            total_loss += torch.log2(loss.item())

        # Calculate bits per character (BPC)
        bpc = total_loss / len(string)

    return bpc
