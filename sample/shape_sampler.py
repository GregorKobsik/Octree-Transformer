import torch
from modules import ShapeTransformer

from sample.sampler import (
    BasicEncoderDecoderSampler,
    DoubleConvolutionalEncoderDecoderSampler,
)


class ShapeSampler():
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

        self.dataset = hparams["dataset"]
        self.subclass = hparams["subclass"]
        self.spatial_dim = hparams['spatial_dim']

        kwargs = {
            "spatial_dim": hparams['spatial_dim'],
            "device": device,
            "max_tokens": hparams["num_positions"],
            "max_resolution": hparams["resolution"],
            "model": self.model,
        }

        if (
            hparams['architecture'] == "encoder_decoder" and hparams['embedding'] == "basic" and
            hparams['head'] == "generative_basic"
        ):
            self.sampler = BasicEncoderDecoderSampler(**kwargs)
        if (
            hparams['architecture'] == "encoder_decoder" and hparams['embedding'] == "double_conv" and
            hparams['head'] == "double_conv"
        ):
            self.sampler = DoubleConvolutionalEncoderDecoderSampler(**kwargs)
        else:
            print(
                "No sampler defined for the combination or parameters:" +
                f"architecture: {hparams['architecture']}, embedding: {hparams['embedding']}," +
                f"and head: {hparams['head']}."
            )
            raise ValueError

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
        if input is None:
            print("ERROR: `input` cannot be `None`.")
            raise ValueError
        return self.sampler(precondition, precondition_resolution, target_resolution, temperature)

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
        random_element_array = torch.randint(
            low=0, high=2, size=self.spatial_dim * [target_resolution], dtype=torch.long
        ).numpy()
        return self.sampler(random_element_array, 2, target_resolution, temperature)
