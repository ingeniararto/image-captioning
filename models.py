import torch
import torch.nn as nn
import torchvision.models as models

# Configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
MAX_SEQ_LEN = 20


# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(resnet.fc.in_features, EMBEDDING_DIM)
        self.relu = nn.ReLU()

    def forward(self, images):
        features = self.feature_extractor(images)
        features = features.view(features.size(0), -1)
        embeddings = self.relu(self.linear(features))
        return embeddings


# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.fc(hiddens)
        return outputs

    def generate_caption(self, features, vocab, max_len=MAX_SEQ_LEN, temperature=1.0):
        output_caption = []

        features_mapped = torch.nn.functional.linear(features, torch.eye(HIDDEN_DIM, EMBEDDING_DIM, device=DEVICE))
        hx = features_mapped.unsqueeze(0).unsqueeze(0)  # Shape: [num_layers, batch_size=1, hidden_dim]
        cx = torch.zeros(NUM_LAYERS, 1, HIDDEN_DIM, device=DEVICE)  # Initialize cx with zeros
        states = (hx, cx)

        # Start token as the initial input
        start_token = torch.tensor([vocab("<start>")], device=DEVICE)
        inputs = self.embedding(start_token).unsqueeze(0)  # Shape: [batch_size=1, seq_len=1, embedding_dim]

        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))  # Shape: [batch_size=1, vocab_size]

            # Apply temperature-based sampling for diversity
            probabilities = torch.softmax(outputs / temperature, dim=-1)
            predicted = torch.multinomial(probabilities, num_samples=1)

            output_caption.append(predicted.item())

            if predicted.item() == vocab.word2idx["<end>"]:
                break

            inputs = self.embedding(predicted)

        return [vocab.idx2word[idx] for idx in output_caption]
