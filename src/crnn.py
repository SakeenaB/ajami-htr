"""
crnn.py
-------
The convolutional-recurrent baseline (Shi, Bai & Yao, 2017) with a residual
encoder, following the configuration Al-azzawi, Barney & Liwicki (2026) used
to reach 10.66% CER on this corpus.

The architecture has three stages, matching Figure 3.4 of the report:

  1. A residual convolutional encoder turns the line image into a grid of
     feature maps. Pooling is deliberately asymmetric - height is collapsed
     much faster than width - so that what emerges is a long thin strip, one
     column per short horizontal step along the line.

  2. Column-wise max pooling flattens the remaining height to one, converting
     the two-dimensional map into a one-dimensional sequence of feature
     vectors that reads left to right along the writing direction.

  3. Three stacked bidirectional LSTMs model the sequence, and a linear
     projection produces, for every column, a distribution over the alphabet
     plus the CTC blank.

Why bidirectional matters for this script: a letter's written form depends on
its neighbours, so a model that has only seen what comes before cannot resolve
positional allography. Each column is classified with context from both sides.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Two 3x3 convolutions with a skip connection around them.

    The skip lets gradients reach the early layers directly, which is what
    makes a deeper encoder trainable (He et al., 2016). Depth matters here
    because the features that distinguish a diacritic from a speck of foxing
    are small and need several layers of context to become separable.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # If the block changes shape, the skip path needs a 1x1 convolution to
        # match, or the addition is impossible.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class CRNN(nn.Module):
    """
    Input : (batch, 1, height, width) greyscale line images
    Output: (time, batch, n_classes) log-probabilities, the layout torch's
            CTCLoss expects.
    """

    def __init__(self, n_classes, input_height=64, channels=(64, 128, 256),
                 lstm_hidden=256, lstm_layers=3, dropout=0.1):
        super().__init__()
        self.input_height = input_height

        # Stem: one 7x7 convolution to pick up broad stroke structure before
        # the residual blocks refine it.
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),          # height and width both halved once
        )

        # Residual stages. After the first, pooling is (2, 1): height halves,
        # width is preserved. Width is the time axis, and every column thrown
        # away is a column CTC can no longer align a character to. Short marks
        # are the first thing lost if the sequence is compressed too hard.
        self.stage1 = nn.Sequential(
            ResidualBlock(channels[0], channels[0]),
            ResidualBlock(channels[0], channels[0]),
            nn.MaxPool2d((2, 1)),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(channels[0], channels[1]),
            ResidualBlock(channels[1], channels[1]),
            nn.MaxPool2d((2, 1)),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(channels[1], channels[2]),
            ResidualBlock(channels[2], channels[2]),
        )

        # Column-wise max pooling: whatever height remains is reduced to one by
        # taking the strongest response in each column. This is the step that
        # turns a 2-D map into a 1-D sequence (Section 3.5.1).
        self.column_pool = nn.AdaptiveMaxPool2d((1, None))

        self.rnn = nn.LSTM(
            input_size=channels[2],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=False,
        )
        self.dropout = nn.Dropout(dropout)

        # x2 because the LSTM is bidirectional: forward and backward states
        # are concatenated at each step.
        self.classifier = nn.Linear(lstm_hidden * 2, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.column_pool(x)            # (B, C, 1, W')
        x = x.squeeze(2)                   # (B, C, W')
        x = x.permute(2, 0, 1)             # (W', B, C) = (time, batch, feature)

        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return nn.functional.log_softmax(x, dim=2)

    def output_length(self, input_width):
        """
        How many time steps the model emits for an image of this width.

        CTC needs this: it must know the true length of each sequence in a
        padded batch, and the loss is undefined if the output is shorter than
        the label. Only the stem halves the width, so the factor is 2.
        """
        return max(1, input_width // 2)


def greedy_decode(log_probs, alphabet, lengths=None):
    """
    Best-path CTC decoding: take the most likely symbol at each time step,
    collapse runs of the same symbol, then drop the blanks.

    A blank between two identical characters is what allows a genuine double
    letter to survive the collapse - without it, "dd" would become "d".

    log_probs: (time, batch, classes)
    returns  : list of strings, one per batch item
    """
    best = log_probs.argmax(dim=2).cpu().numpy()   # (time, batch)
    out = []
    for b in range(best.shape[1]):
        n = best.shape[0] if lengths is None else int(lengths[b])
        out.append(alphabet.decode_ctc(best[:n, b].tolist()))
    return out


if __name__ == "__main__":
    # Shape check with a synthetic batch: this catches the most common class
    # of bug - a mismatch between what the encoder emits and what CTC expects -
    # without needing any data.
    model = CRNN(n_classes=145)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.2f}M")

    for width in (256, 512, 1024):
        x = torch.randn(2, 1, 64, width)
        y = model(x)
        print(f"input (2, 1, 64, {width:>4})  ->  output {tuple(y.shape)}  "
              f"predicted length {model.output_length(width)}")
        assert y.shape[0] == model.output_length(width), \
            "output_length does not match the real output - CTC would break"
    print("shape checks passed")
