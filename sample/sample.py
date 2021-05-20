import torch
from modules import ShapeTransformer

from sample.sample_successive import preprocess_successive, sample_successive, postprocess_successive
from sample.sample_iterative import preprocess_iterative, sample_iterative, postprocess_iterative


class Sampler():
    def __init__(self, checkpoint_path: str, device="cuda"):
        """ Initializes the sampler class. Loads the correct model and sets functions and parameters according to the
            given model.

        Args:
            checkpoint_path: Relative or absolute path to a checkpoint file ("*.ckpt") containing a trained model.
            device: Selects the device on which the sampling should be performed. Either "cpu" or "cuda" (gpu-support)
                available.
        """
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device

        # load and restore model from checkpoint
        pl_module = ShapeTransformer.load_from_checkpoint(checkpoint_path)
        pl_module.freeze()
        self.model = pl_module.model.eval().to(device)

        # extract hyperparameters from the model
        hparams = pl_module.hparams

        self.max_resolution = 2**(hparams["tree_depth"] - 1)
        self.max_tokens = hparams["num_positions"]
        self.dataset = hparams["dataset"]
        self.subclass = hparams["subclass"]

        self.batch_first = False if hparams['attention'].startswith("basic") else True
        self.iterative = True if hparams['architecture'] == "encoder_decoder" else False
        self.spatial_dim = hparams['spatial_dim']

        # prepare sampling functions
        self.preprocess_function = preprocess_iterative if self.iterative else preprocess_successive
        self.sample_function = sample_iterative if self.iterative else sample_successive
        self.postprocess_function = postprocess_iterative if self.iterative else postprocess_successive

    def sample_preconditioned(self, precondition, precondition_resolution=1, target_resolution=32, temperature=1.0):
        """ Samples a single array of elements from the model.

        Args:
            precondition: Use a precondition for sampling up to a resolution in 'start_resolution' given as a numpy
                element array.
            precondition_resolution: Initial resolution used for the precondition. The precondition should have at
                least the `precondition_resolution` side length. Resolution values can be only power of 2.
            target_resolution: The target resolution for the up-sampling process. The resolution should be not bigger,
                than the maximal trained model resolution. Resolution values can be only power of 2.
            temperature: Defines the randomness of the samples. Lower temperatures make the model increasingly
                confident in its top choices, while temperatures greater than 1 decrease confidence. 0 temperature is
                equivalent to argmax/max likelihood, while infinite temperature corresponds to a uniform sampling.

        Return:
            A sampled array of elements (pixels/voxels) with the size of `target_resolution` as a numpy array.

        TODO: If `input` is None, than a single random sample will be drawn from the dataset used in the model.
        TODO: Extend with `batch_size` to allow parallel sampling.
        """

        kwargs = {
            "precondition_resolution": precondition_resolution,
            "target_resolution": target_resolution,
            "temperature": temperature,
            "spatial_dim": self.spatial_dim,
            "device": self.device,
            "max_tokens": self.max_tokens,
            "max_resolution": self.max_resolution,
            "batch_first": self.batch_first,
            "model": self.model,
        }

        # preprocess the input and transform it into a token sequence
        sequence = self.preprocess_function(precondition, **kwargs)

        # enhance the resolution of the sequence or generate a new sequence by sampling new token values
        value = self.sample_function(sequence, **kwargs)

        # postprocess the token value sequence and return it as an array of elements
        return self.postprocess_function(value, **kwargs)

    def sample_random(self, target_resolution=32, temperature=1.0):
        """ Sample a single unconditioned random array of elements from the model.

        Args:
            target_resolution: The target resolution for the up-sampling process. The resolution should be not bigger,
                than the maximal trained model resolution. Resolution values can be only power of 2.
            temperature: Defines the randomness of the samples. Lower temperatures make the model increasingly
                confident in its top choices, while temperatures greater than 1 decrease confidence. 0 temperature is
                equivalent to argmax/max likelihood, while infinite temperature corresponds to a uniform sampling.

        Return:
            A sampled array of elements (pixels/voxels) with the size of `target_resolution` as a numpy array.
        """
        # create an initial array, with all elements marked as undefined/mixed.
        initial_element_array = torch.randint(
            low=0, high=2, size=self.spatial_dim * [self.max_resolution], dtype=torch.long
        ).numpy()
        return self.sample_preconditioned(initial_element_array, 1, target_resolution, temperature)
