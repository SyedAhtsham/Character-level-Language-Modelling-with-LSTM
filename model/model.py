import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initialize the LSTM model with specified input, hidden, and output sizes,
        and the number of layers in the LSTM stack.
        """
        # Call the superclass constructor
        super(LSTM, self).__init__()

        # Set the model hyperparameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Embedding layer to convert input to hidden size vectors
        self.embedding_layer = nn.Embedding(input_size, hidden_size)

        # LSTM layer to process sequential data
        self.lstm_layer = nn.LSTM(hidden_size, hidden_size, num_layers)

        # Linear layer to project LSTM output to desired output size
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input_char, hidden_state):
        """
        Define the forward pass of the LSTM model.
        Takes an input character, hidden state, and outputs the next character distribution
        along with the updated hidden state.
        """
        # Embed the input character and reshape for LSTM input
        embedded_input = self.embedding_layer(input_char).view(1, 1, -1)

        # Pass through the LSTM layer and get the output
        lstm_output, updated_hidden = self.lstm_layer(embedded_input, hidden_state)

        # Project the LSTM output to the desired output size
        output_distribution = self.output_layer(lstm_output.view(1, -1))

        return output_distribution, updated_hidden

    def init_hidden(self):
        """
        Initialize the hidden and cell states of the LSTM.
        """
        # Initialize hidden and cell states with zeros
        initial_hidden_state = torch.zeros(self.num_layers, 1, self.hidden_size)
        initial_cell_state = torch.zeros(self.num_layers, 1, self.hidden_size)

        return initial_hidden_state, initial_cell_state
