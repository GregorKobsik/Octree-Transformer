import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, token_embedding, generative_head, **_):
        """ Creates an instance of an autoencoder.

        It passes received tokens from the `token_embedding` directly to the `generative_head` without processing them
        by a transformer instance.

        Args:
            token_embedding: Instance of a token embedding, which embedds given sequences into an embedding space.
            generative_head: Instance of a generative head, which transforms the embedding space. into logits.
        """
        super(Autoencoder, self).__init__()

        # token embedding
        self.embedding = token_embedding

        # generative head
        self.head = generative_head

    def forward(self, sequence):
        """ Performs a pass of the input sequence through the embedding and generative head.

        Args:
            sequence: Tuple containing input sequences as (value, depth, position) sequences with the shapes ([N, S],
            [N, S], [N, S, A]).

        Return:
            Logits which describe the likelihood of the current token with shape [N, S, V].
        """
        # encode input into embedding space
        z = self.embedding.source(*sequence)  # [N, T, E]
        # return logits
        return self.head(z, *sequence)  # [N, S, V]
