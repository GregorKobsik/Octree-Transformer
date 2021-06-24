import torch.nn as nn


class Autoencoder(nn.Module):
    """ Creates an autoencoder which passes received tokens from the `token_embedding` directly to the `generative_head`.

    Args:
        token_embedding: Instance of a token embedding, which embedds given sequences of tokens into an embedding space.
        generative_head: Instance of a generative head, which transforms the embedding space. into logits.
    """
    def __init__(self, token_embedding, generative_head, **_):
        super(Autoencoder, self).__init__()

        # token embedding
        self.embedding = token_embedding

        # generative head
        self.head = generative_head

    def forward(self, sequence):
        # encode input into embedding space
        z = self.embedding.source(*sequence)  # [N, T, E]
        # return logits
        return self.head(z, *sequence)  # [N, S, V]
