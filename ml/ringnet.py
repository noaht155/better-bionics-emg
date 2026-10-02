"""The general network: ring convolutions over the 8 electrodes, a grip head and a finger-angle head, and a
per-person channel transform in front.

The band's electrodes sit in a circle, electrode 8 next to 1. The convolutions treat the input as a small image
of 8 ring positions by time and look at each electrode with its two neighbours, wrapping round, with the same
weights at every position. Rotating the band by whole electrodes then only shifts the features round the ring,
and pooling over the ring (mean and max) gives the same output either way. What is left after that (half
electrodes, the band moving along the arm, a different person) is for the per-person transform: an 8x8 channel
mix and a gain per channel, starting as no change and fitted on a calibration recording with the rest frozen.
"""
from collections import Counter, deque

import torch
from torch import nn
import torch.nn.functional as F

from hand_angles import RELIABLE, JOINTS

# Filtered EMG is a few uV at rest and a few hundred in a grip. Dividing by this brings it near unit scale
INPUT_SCALE_UV = 50.0
ANGLE_JOINTS = RELIABLE
ANGLE_INDEX = [JOINTS.index(j) for j in ANGLE_JOINTS]
# Angles are trained in this unit so their loss is on a similar scale to the grip loss
ANGLE_SCALE_DEG = 60.0


class ChannelTransform(nn.Module):
    """x -> gain * (mix @ x) per window. Starts as the identity."""

    def __init__(self, channels=8):
        super().__init__()
        self.mix = nn.Parameter(torch.eye(channels))
        self.log_gain = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return torch.einsum("cd,bdt->bct", self.mix, x) * self.log_gain.exp()[None, :, None]

    def reset(self):
        with torch.no_grad():
            self.mix.copy_(torch.eye(self.mix.shape[0]))
            self.log_gain.zero_()


class RingConv(nn.Module):
    """Convolution over (ring position, time). Wraps round the ring, pads time to keep its length before the stride."""

    def __init__(self, cin, cout, time_kernel, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, (3, time_kernel), stride=(1, stride))
        self.bn = nn.BatchNorm2d(cout)
        self.time_pad = time_kernel // 2

    def forward(self, x):
        x = torch.cat([x[:, :, -1:], x, x[:, :, :1]], dim=2)
        x = F.pad(x, (self.time_pad, self.time_pad, 0, 0))
        return F.relu(self.bn(self.conv(x)))


class RingNet(nn.Module):
    def __init__(self, classes, width=32, angles=True):
        super().__init__()
        self.classes = list(classes)
        self.transform = ChannelTransform()
        self.blocks = nn.Sequential(
            RingConv(1, width // 2, 7),
            RingConv(width // 2, width, 5, stride=2),
            RingConv(width, width, 5, stride=2),
            RingConv(width, 2 * width, 3, stride=2),
        )
        features = 4 * width
        self.dropout = nn.Dropout(0.3)
        self.grip_head = nn.Linear(features, len(self.classes))
        self.angle_head = nn.Linear(features, len(ANGLE_JOINTS)) if angles else None

    def embed(self, x):
        """x: (batch, 8, samples) of filtered EMG in uV. Returns the pooled features."""
        x = self.transform(x / INPUT_SCALE_UV)
        h = self.blocks(x[:, None])            # (batch, width, 8 ring positions, time)
        h = h.mean(dim=3)                      # average over time
        return torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)   # mean and max over the ring

    def forward(self, x):
        z = self.dropout(self.embed(x))
        angles = self.angle_head(z) * ANGLE_SCALE_DEG if self.angle_head is not None else None
        return self.grip_head(z), angles

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def rotation_check(model, x):
    """Largest change in grip output when the band is rotated by whole electrodes. Near zero by design."""
    model.eval()
    with torch.no_grad():
        base = model(x)[0]
        return max(float((model(torch.roll(x, k, dims=1))[0] - base).abs().max()) for k in range(1, 8))


def save(model, info, path):
    """info: plain values only (strings, numbers, lists), so loading never has to run pickled code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "classes": model.classes, "info": info}, path)


def load(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=True)
    model = RingNet(blob["classes"]).to(device)
    model.load_state_dict(blob["state"])
    model.eval()
    return model, blob["info"]


def load_live(path):
    """A saved network in the same shape the app's predictor uses for LDA models."""
    model, info = load(path)
    return {"kind": "network", "net": model, "info": info, "classes": model.classes, "channels": info["channels"],
            "rate": info["rate"], "window": info["window"], "step_ms": info["step_ms"], "vote": info["vote"],
            "options": {"accel": False}}


class NetPredictor:
    """Live grip prediction with a saved network: same threshold and majority vote as the LDA predictor."""

    def __init__(self, model, threshold, unsure):
        self.model = model
        self.net = model["net"]
        self.win = model["window"]
        self.recent = deque(maxlen=model["vote"])
        self.threshold = threshold
        self.unsure = unsure

    def predict(self, filtered, accel=None):
        x = torch.tensor(filtered[None, :, -self.win:], dtype=torch.float32)
        with torch.no_grad():
            probs = F.softmax(self.net(x)[0], dim=1)[0].numpy()
        classes = self.model["classes"]
        raw = classes[int(probs.argmax())] if probs.max() >= self.threshold else self.unsure
        self.recent.append(raw)
        counts = Counter(self.recent)
        best = max(counts.values())
        label = next(q for q in reversed(self.recent) if counts[q] == best)
        return label, raw, dict(zip(classes, probs.round(3).tolist()))


if __name__ == "__main__":
    net = RingNet(["fist", "key", "open", "pinch", "point", "rest", "tripod"])
    x = torch.randn(16, 8, 100) * 30
    grips, angles = net(x)
    print(f"parameters: {net.parameter_count()}, grip output {tuple(grips.shape)}, angle output {tuple(angles.shape)}")
    print(f"largest grip output change for a band rotated by 1 to 7 electrodes: {rotation_check(net, x):.2e}")
    print(f"(compare: output range {float(grips.detach().abs().max()):.2f})")
