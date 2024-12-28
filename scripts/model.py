import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):

        _, (h, c) = self.lstm(x)
        return h, c


class Decoder(nn.Module):
    def __init__(self, hidden_size, input_size, num_layers, fcc_intermediate):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)


        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, fcc_intermediate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fcc_intermediate, 1)

    def forward(self, decoder_input, h, c):

        out, (h, c) = self.lstm(decoder_input, (h, c))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out, h, c


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, fcc_intermediate, prediction_horizon):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, input_size, num_layers, fcc_intermediate)  
        self.prediction_horizon = prediction_horizon

    def forward(self, x, teacher_forcing_targets=None, teacher_forcing_ratio=0.5):

        batch_size = x.size(0)

        h, c = self.encoder(x)
        num_directions = h.shape[0] // self.num_layers


        h = h.view(self.num_layers, num_directions, h.shape[1], h.shape[2]) 
        c = c.view(self.num_layers, num_directions, c.shape[1], c.shape[2])


        h = h.sum(dim=1) 
        c = c.sum(dim=1)


        decoder_input = x[:, -1, :].unsqueeze(1) 
        outputs = []

        for t in range(self.prediction_horizon):
            out, h, c = self.decoder(decoder_input, h, c)  
            outputs.append(out)

            if self.training and teacher_forcing_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = teacher_forcing_targets[:, t, :].unsqueeze(1) 
            else:
                other_features = teacher_forcing_targets[:, t, 1:].unsqueeze(1)
                predicted_power = out[:, :1].unsqueeze(1) 
                decoder_input = torch.cat([predicted_power, other_features], dim=-1)

        return torch.cat(outputs, dim=1) 
