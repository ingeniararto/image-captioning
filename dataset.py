import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class Flickr30kDataset(Dataset):
    def __init__(self, img_dir, captions_file, vocab, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.vocab = vocab
        self.captions = []
        with open(captions_file, "r") as f:
            for line in f:
                img, number, caption = line.strip().split(",")
                if number != "4":
                    continue
                self.captions.append((img, caption))

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name, caption = self.captions[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        tokens = caption.lower().split()
        caption = [self.vocab("<start>")] + [self.vocab(token) for token in tokens] + [self.vocab("<end>")]

        return image, torch.tensor(caption)
