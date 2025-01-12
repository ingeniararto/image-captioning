import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import time
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from vocabulary import build_vocab
from dataset import Flickr30kDataset
from models import Decoder, Encoder
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "image_captioning_model.pth"

# Preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Collate function for dataloader
def collate_fn(data):
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions


# Save and load model
def save_model(encoder, decoder, path):
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict()
    }, path)


def load_model(encoder, decoder, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()


# Testing function
def test_model(encoder, decoder, dataloader, vocab):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        total_bleu_score = 0
        count = 0
        for images, captions in dataloader:
            images = images.to(DEVICE)
            features = encoder(images)
            for i in range(len(captions)):
                caption = decoder.generate_caption(features[i], vocab)
                print("Generated Caption:", " ".join(caption))
                print("Original Caption:", " ".join([vocab.idx2word.get(c.item(), '<unk>') for c in captions[i]]))
                original_caption_decoded = [
                    vocab.idx2word[idx.item()] for idx in captions[i]
                    if idx.item() not in {vocab.word2idx["<pad>"], vocab.word2idx["<start>"], vocab.word2idx["<end>"]}
                ]

                bleu_score = sentence_bleu([original_caption_decoded], caption, weights=(1.0, 0, 0, 0))
                total_bleu_score += bleu_score
                count += 1

            avg_bleu_score = total_bleu_score / count
            print(f"Average BLEU Score: {avg_bleu_score:.4f}")


# Training loop
def train_model(encoder, decoder, dataloader, val_loader,  criterion, optimizer, vocab_size):
    encoder.train()
    decoder.train()

    train_losses = []
    val_losses = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        for i, (images, captions) in enumerate(dataloader):
            images, captions = images.to(DEVICE), captions.to(DEVICE)
            optimizer.zero_grad()

            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])

            outputs = outputs.view(-1, vocab_size)  # Flatten the predictions
            targets = captions.view(-1)  # Flatten the target captions

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Step {i}, Loss: {loss.item():.4f}")
        train_losses.append(epoch_train_loss / len(dataloader))
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for images, captions in val_loader:
                images, captions = images.to(DEVICE), captions.to(DEVICE)

                features = encoder(images)
                outputs = decoder(features, captions[:, :-1])

                outputs = outputs.view(-1, vocab_size)
                targets = captions.view(-1)

                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
        val_losses.append(epoch_val_loss / len(val_loader))
        # Save model after each epoch
        save_model(encoder, decoder, MODEL_SAVE_PATH)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train and Validation Loss")
    plt.savefig("loss_plot.png")
    plt.close()


# Main code
if __name__ == "__main__":
    IMG_DIR = "dataset/flickr30k_images"
    CAPTIONS_FILE = "dataset/comments.txt"

    vocab = build_vocab(CAPTIONS_FILE, threshold=5)
    VOCAB_SIZE = len(vocab)

    dataset = Flickr30kDataset(IMG_DIR, CAPTIONS_FILE, vocab, transform)

    train_size = int(0.2 * len(dataset))
    test_size = int(0.01 * len(dataset))
    val_size = int(0.01 * len(dataset))
    x_size = len(dataset) - train_size - test_size - val_size

    train_dataset, test_dataset, val_dataset, _ = random_split(dataset, [train_size, test_size, val_size, x_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder().to(DEVICE)
    decoder = Decoder(VOCAB_SIZE).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>'])
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

    train_model(encoder, decoder, train_loader, val_loader, criterion, optimizer, VOCAB_SIZE)

    # Load model for testing
    load_model(encoder, decoder, MODEL_SAVE_PATH)

    # Testing phase
    test_model(encoder, decoder, test_loader, vocab)
