This module implements a number of different autoencoders. To import

```python
From deep.autoencoder import AutoEncoder
```

## Tied Autoencoder

As simple as it gets. A single layer autoencoder where the encoding
# and decoding layers share a weight matrix.

```python
# creates a tied autoencoder with 100 hidden sigmoid units.
ae = AutoEncoder(100)

# creates a tied autoencoder with 100 hidden tanh units
from deep.layers import TanhLayer.
ae = AutoEncoder(TanhLayer(100))
```

## Untied Autoencoder

Still pretty simple. A single layer autoencoder where encoding and decoding
layers have separate weight matrices.

```python
# creates an untied autoencoder with 100 hidden encoding sigmoid units
# and 100 hidden decoding sigmoid units.
ae = AutoEncoder(100)

# creates an untied autoencoder with 100 hidden encoding sigmoid units
# and 100 hidden decoding tanh units.
from deep.layers import TanhLayer
ae = AutoEncoder(SigmoidLayer(100), TanhLayer(100))
```

## Denoising Autoencoder

Now were talking. A single layer autoencoder where the input is corrupted
before reconstruction. Can be tied or untied.

```python
# creates a denoising autoencoder identical to a tied autoencoder but
# corrupts input with 50% salt and pepper noise before reconstruction.
dae = DenoisingAutoEncoder(100, .5)

# creates an untied denoising autoencoder which corrupts input with
# gaussian noise with a standard deviation of .5
from deep.corruption import GaussianCorruption
s = SigmoidLayer(100)
t = TanhLayer(100)
g = GaussianCorruption(.5)
dae = DenoisingAutoEncoder(s, t, g)
```