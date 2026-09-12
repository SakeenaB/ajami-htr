"""
dataset.py
----------
The PyTorch Dataset that feeds line images and their transcriptions to the
models, plus the collate function that batches them for a CTC loss.

Two constraints shape the design.

First, the models need a fixed image height but must tolerate varying width,
because manuscript lines differ in length. Every image is therefore scaled to
a fixed height with its aspect ratio preserved, and batches are padded to the
width of their widest member. Distorting the aspect ratio would squash the
diacritics towards their base letters, which is precisely the information the
project is trying to preserve.

Second, CTC needs to know the true width of every image and the true length of
every label, because the padding is not real data. Those lengths are returned
alongside the tensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from alphabet import Alphabet


# Fixed input height in pixels. 64 is a common choice for CTC recognisers and
# keeps the sequence length manageable; the corpus average line is about
# 1290 x 215, so this is roughly a third-scale reduction.
DEFAULT_HEIGHT = 64

# Widths are clamped so that one unusually long line cannot blow up the memory
# of an entire batch.
MAX_WIDTH = 1600


class AjamiLineDataset(Dataset):
    """
    One item per manuscript line.

    Expects the dataset.csv produced by the Kaggle pipeline, with columns:
        ImageID  - filename stem, e.g. Infiraji_4.pdf_page_13_line_016
        text     - the ground-truth transcription
        lang     - 'Hausa' or 'Fulfulde'
        split    - 'train', 'val' or 'test'
    """

    def __init__(
        self,
        csv_path,
        images_dir,
        alphabet,
        split=None,
        height=DEFAULT_HEIGHT,
        max_width=MAX_WIDTH,
        transform=None,
        language=None,
    ):
        self.images_dir = Path(images_dir)
        self.alphabet = alphabet
        self.height = height
        self.max_width = max_width
        # transform is an albumentations pipeline, or None. It differs per
        # model - geometric for the CRNN, morphological for Kraken,
        # photometric for TrOCR - which is how error decorrelation is
        # engineered (Section 3.5.4).
        self.transform = transform

        df = pd.read_csv(csv_path)
        df["text"] = df["text"].fillna("").astype(str)

        if split is not None:
            df = df[df["split"] == split]
        if language is not None:
            df = df[df["lang"] == language]

        # A line with no transcription cannot supply a training signal, and a
        # line whose image is missing would crash mid-epoch. Both are dropped
        # here, once, with a report - never silently inside __getitem__.
        before = len(df)
        df = df[df["text"].str.len() > 0]
        no_text = before - len(df)

        exists = df["ImageID"].apply(
            lambda s: (self.images_dir / f"{s}.png").exists()
        )
        no_image = int((~exists).sum())
        df = df[exists]

        self.df = df.reset_index(drop=True)
        self.dropped = {"empty_text": no_text, "missing_image": no_image}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        # Greyscale: colour carries no information for handwriting, and one
        # channel is three times less to process.
        image = Image.open(self.images_dir / f"{row.ImageID}.png").convert("L")

        # Scale to fixed height, aspect ratio preserved.
        scale = self.height / image.height
        width = max(1, min(self.max_width, int(round(image.width * scale))))
        image = image.resize((width, self.height), Image.LANCZOS)

        array = np.array(image, dtype=np.uint8)

        if self.transform is not None:
            array = self.transform(image=array)["image"]

        # Scale to [0, 1] and add the channel dimension: (1, H, W).
        tensor = torch.from_numpy(array).float().div(255.0).unsqueeze(0)

        label = torch.tensor(self.alphabet.encode(row.text), dtype=torch.long)

        return {
            "image": tensor,
            "label": label,
            "text": row.text,
            "image_id": row.ImageID,
            "lang": row.lang,
            "width": width,
        }

    def report(self):
        lines = [
            f"lines            : {len(self)}",
            f"dropped (no text): {self.dropped['empty_text']}",
            f"dropped (no img) : {self.dropped['missing_image']}",
        ]
        if len(self):
            counts = self.df["lang"].value_counts()
            lines.append("by language      : " + ", ".join(
                f"{k} {v}" for k, v in counts.items()
            ))
            lengths = self.df["text"].str.len()
            lines.append(
                f"text length      : min {lengths.min()}, "
                f"mean {lengths.mean():.1f}, max {lengths.max()}"
            )
        return "\n".join(lines)


def collate(batch):
    """
    Pad a list of items into a single batch.

    Images are padded on the right with white (value 1.0 after scaling), which
    matches the page background and so introduces no spurious ink. Labels are
    concatenated into one flat tensor, which is the format torch's CTCLoss
    expects, together with the true lengths of both.
    """
    heights = {item["image"].shape[1] for item in batch}
    if len(heights) != 1:
        raise ValueError(f"mixed image heights in one batch: {heights}")

    height = heights.pop()
    max_width = max(item["image"].shape[2] for item in batch)

    images = torch.ones(len(batch), 1, height, max_width)
    for i, item in enumerate(batch):
        w = item["image"].shape[2]
        images[i, :, :, :w] = item["image"]

    labels = torch.cat([item["label"] for item in batch])
    label_lengths = torch.tensor([len(item["label"]) for item in batch],
                                 dtype=torch.long)
    image_widths = torch.tensor([item["image"].shape[2] for item in batch],
                                dtype=torch.long)

    return {
        "images": images,
        "labels": labels,
        "label_lengths": label_lengths,
        "image_widths": image_widths,
        "texts": [item["text"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "langs": [item["lang"] for item in batch],
    }


def make_loader(dataset, batch_size=16, shuffle=False, num_workers=2,
                weighted_by_language=False):
    """
    Wrap a dataset in a DataLoader.

    weighted_by_language addresses the 5,400 Hausa to 806 Fulfulde imbalance:
    with it enabled, each language contributes equally in expectation, so a
    batch is not almost entirely Hausa. It does not create information that
    is absent, and Fulfulde results are still reported separately.
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler

    sampler = None
    if weighted_by_language:
        counts = dataset.df["lang"].value_counts().to_dict()
        weights = dataset.df["lang"].map(lambda l: 1.0 / counts[l]).tolist()
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset),
                                        replacement=True)
        shuffle = False  # a sampler and shuffle are mutually exclusive

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the dataset.")
    parser.add_argument("--csv", default="dataset.csv")
    parser.add_argument("--images", default="lines")
    parser.add_argument("--alphabet", default="alphabet.json")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    alphabet = Alphabet.load(args.alphabet)
    ds = AjamiLineDataset(args.csv, args.images, alphabet, split=args.split)
    print(f"split: {args.split}")
    print(ds.report())

    loader = make_loader(ds, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    print()
    print("one batch")
    print("  images        :", tuple(batch["images"].shape))
    print("  labels        :", tuple(batch["labels"].shape))
    print("  label lengths :", batch["label_lengths"].tolist())
    print("  image widths  :", batch["image_widths"].tolist())
    print("  first text    :", batch["texts"][0][:50])
