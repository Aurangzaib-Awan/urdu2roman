import os
import torch
from model.seq2seq import Encoder, Decoder, Seq2Seq
from tokenizer import urdu2idx, idx2roman, roman2idx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 50

# -----------------------
# Load model once
# -----------------------
INPUT_DIM = len(urdu2idx)
OUTPUT_DIM = len(roman2idx)

encoder = Encoder(INPUT_DIM).to(device)
decoder = Decoder(OUTPUT_DIM).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

state_dict = torch.load(os.path.join("model", "best_model.pt"), map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# -----------------------
# Translate function
# -----------------------
def translate_sentence(sentence, max_len=MAX_LEN):
    tokens = [urdu2idx.get(ch, urdu2idx['<pad>']) for ch in sentence]
    tokens = [urdu2idx['<sos>']] + tokens + [urdu2idx['<eos>']]
    tokens = tokens[:max_len] + [urdu2idx['<pad>']]*(max_len - len(tokens))
    src_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        enc_outputs, hidden, cell = model.encoder(src_tensor)
        hidden, cell = model.init_decoder_state(hidden, cell, batch_size=1)
        input_tok = torch.tensor([roman2idx['<sos>']], dtype=torch.long).to(device)
        outputs = []

        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_tok, hidden, cell)
            top1 = output.argmax(1)
            if top1.item() == roman2idx['<eos>']:
                break
            outputs.append(idx2roman[top1.item()])
            input_tok = top1

    return "".join(outputs)
