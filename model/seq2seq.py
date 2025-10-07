import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim=256, hidden_dim=512, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers,bidirectional=True, batch_first=True,dropout=dropout if n_layers>1 else 0)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim=256, hidden_dim=512, n_layers=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, batch_first=True,dropout=dropout if n_layers>1 else 0)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.enc2dec_h = nn.Linear(encoder.hidden_dim * 2, decoder.hidden_dim)
        self.enc2dec_c = nn.Linear(encoder.hidden_dim * 2, decoder.hidden_dim)
        # Extra layers to expand from 2 → 4
        self.bridge_h = nn.Linear(2 * decoder.hidden_dim, 4 * decoder.hidden_dim)
        self.bridge_c = nn.Linear(2 * decoder.hidden_dim, 4 * decoder.hidden_dim)

    def init_decoder_state(self, hidden, cell, batch_size):
        return hidden,cell

    def forward(self, src, trg):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.embedding.num_embeddings
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        enc_outputs, hidden, cell = self.encoder(src)

        input = trg[:, 0]
        hidden, cell = self.init_decoder_state(hidden, cell, batch_size)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
   
        return outputs

