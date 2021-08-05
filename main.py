from argparse import ArgumentParser
from executable import train, test, sample

if __name__ == "__main__":
    """ Parses the console input and calls one of the executable functions.

    Func:
        train: Creates a new model with random or pretrained weights and trains it according to the config file.
            Allows to override the default config file arguments with command line arguments.
        test: not supported, yet.
        sample: not supported, yet.

    TODO: test - Tests the loss of the given checkpoint on the test data set.
    TODO: sample - Samples a number of sequences on the given checkpoint and creates an image with the results.
    """
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    # TRAINING
    parser_train = subparsers.add_parser("train")
    parser_train.set_defaults(func=train)
    parser_train.add_argument("config", type=str)
    parser_train.add_argument("--name", type=str, default=None)
    # data
    parser_train.add_argument("--datapath", type=str, default='datasets')
    parser_train.add_argument("--dataset", type=str, default=None)
    parser_train.add_argument("--subclass", type=str, default=None)
    parser_train.add_argument("--transform", type=str, default=None)
    parser_train.add_argument("--position_encoding", type=str, default=None)
    parser_train.add_argument("--num_vocab", type=int, default=None)
    parser_train.add_argument("--resolution", type=int, default=None)
    parser_train.add_argument("--spatial_dim", type=int, default=None)
    # architecture
    parser_train.add_argument("--embedding", type=str, default=None)
    parser_train.add_argument("--head", type=str, default=None)
    parser_train.add_argument("--architecture", type=str, default=None)
    parser_train.add_argument("--attention", type=str, default=None)
    parser_train.add_argument("--num_positions", type=int, default=None)
    parser_train.add_argument("--embed_dim", type=int, default=None)
    parser_train.add_argument("--num_layers", type=int, default=None)
    parser_train.add_argument("--num_heads", type=int, default=None)
    # training
    parser_train.add_argument("--pretrained", type=str, default=None)
    parser_train.add_argument("--loss_function", type=str, default=None)
    parser_train.add_argument("--val_loss_function", type=str, default=None)
    parser_train.add_argument("--epochs", type=int, default=None)
    parser_train.add_argument("--warmup_steps", default=None)
    parser_train.add_argument("--batch_size", type=int, default=None)
    parser_train.add_argument("--accumulate_grad_batches", type=int, default=None)
    parser_train.add_argument("--learning_rate", type=float, default=None)
    # hardware
    parser_train.add_argument("--gpus", type=int, default=None)
    parser_train.add_argument("--precision", type=int, default=None)
    # logging
    parser_train.add_argument("--log_gpu", type=str, default=None)
    parser_train.add_argument("--log_gradient", type=str, default=None)
    parser_train.add_argument("--log_weights_and_biases", type=str, default=None)
    parser_train.add_argument("--log_learning_rate", type=str, default=None)

    # TESTING
    parser_test = subparsers.add_parser("test")
    parser_test.set_defaults(func=test)

    # SAMPLING
    parser_sample = subparsers.add_parser("sample")
    parser_sample.set_defaults(func=sample)

    args = parser.parse_args()
    args.func(vars(args))
