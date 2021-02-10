from argparse import ArgumentParser
from executable import train, test, sample

if __name__ == "__main__":
    """ Parses the console input and calls one of the executable functions.

    Func:
        train: Creates a new model with random or pretrained weights and trains it according to the config file.
        test: Tests the loss of the given checkpoint on the test data set.
        sample: Samples a number of sequences on the given checkpoint and creates an image with the results.
     """
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    # TRAINING
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("config", type=str)
    parser_train.add_argument("--pretrained", type=str, default=None)
    parser_train.set_defaults(func=train)

    # TESTING
    parser_test = subparsers.add_parser("test")
    parser_test.add_argument("datadir", type=str)
    parser_test.add_argument("--checkpoint", type=str, default="checkpoints/best.ckpt")
    parser_test.set_defaults(func=test)

    # SAMPLING
    parser_sample = subparsers.add_parser("sample")
    parser_sample.add_argument("datadir", type=str)
    parser_sample.add_argument("--checkpoint", type=str, default="checkpoints/best.ckpt")
    parser_sample.add_argument("--input_depth", default=3, type=int)
    parser_sample.set_defaults(func=sample)

    args = parser.parse_args()
    args.func(args)
