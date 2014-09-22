import theano.tensor as T


def salt_and_pepper_noise(self, inputs):
    a = self.theano_rng.binomial(size=inputs.shape, n=1,
                                 p=1-self.corruption, dtype='float32')
    b = self.theano_rng.binomial(size=inputs.shape, n=1,
                                 p=0.5, dtype='float32')
    return inputs * a + T.eq(a, 0) * b
