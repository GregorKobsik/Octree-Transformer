import math
import torch

from utils import kdTree, TrinaryRepresentation


class AbstractSampler():
    def __init__(self, model, embedding, head, spatial_dim, device, **_):
        """ Provides an abstract implementation of the sampler.

        Args:
            model: Model which is used for sampling.
            embedding: Token embedding type used in the model.
            head: Generative head type used in the model.
            spatial_dim: The spatial dimensionality of the array of elements.
            device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
        """
        self.embedding = embedding
        self.head = head
        self.spatial_dim = spatial_dim
        self.device = device
        self.model = model

        self.tri_repr = TrinaryRepresentation(spatial_dim)

    def __call__(self, precondition, precondition_resolution, target_resolution, temperature):
        """ Encode and decode the last layer of the given precondition.

        Args:
            precondition: An array of elements (pixels/voxels) as an numpy array.
            precondition_resolution: Resolution at which the autoencoder will reconstruct the layer.
            target_resolution: unused.
            temperature: Defines the randomness of the samples.

        Return:
            An array of elements as a numpy array, which represents the sampled shape.
        """
        # preprocess the input and transform it into a token sequence
        sequence = self.preprocess(precondition, precondition_resolution)

        # encode and decode the last sequence layer
        value = self.sample(sequence, target_resolution, temperature)

        # postprocess the token value sequence and return it as an array of elements
        return self.postprocess(value, target_resolution)

    def preprocess(self, precondition, precondition_resolution):
        """ Transform input array elements into token sequences.

        Args:
            precondition: An array of elements (pixels/voxels) as an numpy array.
            precondition_resolution: Resolution, to which the input array will be downscaled and used as a precondition.

        Return:
            PyTorch tensor consisting of token sequences: (value, depth, position).
        """
        # convert input array into token sequence
        tree = kdTree(self.spatial_dim).insert_element_array(precondition)
        value, depth, pos = tree.get_token_sequence(
            depth=math.log2(precondition_resolution), return_depth=True, return_pos=True
        )

        # if neccessary, transform data into trinary representation
        if self.embedding.startswith('discrete_transformation'):
            value, depth, pos = self.tri_repr.encode_trinary(value, depth, pos)

        # convert sequence tokens to PyTorch as a long tensor
        value = torch.tensor(value, dtype=torch.long, device=self.device)
        depth = torch.tensor(depth, dtype=torch.long, device=self.device)
        pos = torch.tensor(pos, dtype=torch.long, device=self.device)

        return value, depth, pos

    def sample(self, sequences, target_resolution, temperature):
        """ Run the model once on the last layer of the input sequence and sample new values for each token.

        Args:
            sequences: Token sequences, consisting of values, depth and position sequences.
            target_resolution: unused.
            temperatur: Defines the randomness of the samples.

        Return:
            A token sequence with values, encoding the final sample.
        """
        raise ValueError("ERROR: Please override the `sample` function to implement a sampler.")

    def postprocess(self, value, target_resolution):
        """ Transform sequence of value tokens into an array of elements (voxels/pixels).

        Args:
            value: Value token sequence as a pytorch tensor.
            target_resolution: Resolution up to which an object should be sampled.

        Return:
            An array of elements as a numpy array.
        """
        # move value sequence to the cpu and convert to numpy array
        value = value.cpu().numpy()

        # if neccessary, transform data from trinary to basic representation
        if self.head.startswith('discrete_transformation'):
            value = self.tri_repr.decode_trinary_value(value)

        # insert the sequence into a kd-tree
        tree = kdTree(self.spatial_dim).insert_token_sequence(
            value,
            resolution=target_resolution,
            autorepair_errors=True,
            silent=True,
        )

        # retrive pixels/voxels from the kd-tree
        return tree.get_element_array(mode="occupancy")
