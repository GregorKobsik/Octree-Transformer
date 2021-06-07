import unittest
import torch
import numpy.testing as np_test

from sample.sample_successive import append_next_layer_tokens


class TestPrepareInputForNextLayer_Spatial2(unittest.TestCase):
    """ Tests if the sequence preparation function creates correct depth and position input sequences. """
    def depth_layer_0(self, device):
        """ Test the input for the empty sequence. """
        # define input sequences
        seq = torch.tensor([], dtype=torch.long, device=device)
        depth = torch.tensor([], dtype=torch.long, device=device)
        pos = torch.tensor([], dtype=torch.long, device=device)

        # define expected output sequences
        target_seq = [1, 1, 1, 1]
        target_depth = [1, 1, 1, 1]
        target_pos = [[16, 16], [16, 48], [48, 16], [48, 48]]
        target_future_tokens = 4

        # call the tested function
        output_seq, output_depth, output_pos, future_tokens = append_next_layer_tokens(seq, depth, pos, 2, 32)

        # compare output of the function with expected output
        if device == "cpu":
            np_test.assert_array_equal(output_seq, target_seq)
            np_test.assert_array_equal(output_depth, target_depth)
            np_test.assert_array_equal(output_pos, target_pos)
            self.assertEqual(future_tokens, target_future_tokens)
        elif device == "cuda":
            np_test.assert_array_equal(output_seq.cpu(), target_seq)
            np_test.assert_array_equal(output_depth.cpu(), target_depth)
            np_test.assert_array_equal(output_pos.cpu(), target_pos)
            self.assertEqual(future_tokens.cpu(), target_future_tokens)
        else:
            self.assertTrue(False)

    def test_depth_layer_0_cpu(self):
        """ Test the input for the empty sequence on the cpu. """
        self.depth_layer_0(device="cpu")

    def test_depth_layer_0_cuda(self):
        """ Test the input for the empty sequence on the gpu. """
        self.depth_layer_0(device="cuda")

    def depth_layer_1(self, device):
        """ Test the input for the first input layer """
        # define input sequences
        seq = torch.tensor([2, 2, 2, 2], dtype=torch.long, device=device)
        depth = torch.tensor([1, 1, 1, 1], dtype=torch.long, device=device)
        pos = torch.tensor([[16, 16], [16, 48], [48, 16], [48, 48]], dtype=torch.long, device=device)

        # define expected output sequences
        target_seq = [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        target_depth = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        target_pos = [
            [16, 16], [16, 48], [48, 16], [48, 48], [8, 8], [8, 24], [24, 8], [24, 24], [8, 40], [8, 56], [24, 40],
            [24, 56], [40, 8], [40, 24], [56, 8], [56, 24], [40, 40], [40, 56], [56, 40], [56, 56]
        ]
        target_future_tokens = 16

        # call the tested function
        output_seq, output_depth, output_pos, future_tokens = append_next_layer_tokens(seq, depth, pos, 2, 32)

        # compare output of the function with expected output
        if device == "cpu":
            np_test.assert_array_equal(output_seq, target_seq)
            np_test.assert_array_equal(output_depth, target_depth)
            np_test.assert_array_equal(output_pos, target_pos)
            self.assertEqual(future_tokens, target_future_tokens)
        elif device == "cuda":
            np_test.assert_array_equal(output_seq.cpu(), target_seq)
            np_test.assert_array_equal(output_depth.cpu(), target_depth)
            np_test.assert_array_equal(output_pos.cpu(), target_pos)
            self.assertEqual(future_tokens.cpu(), target_future_tokens)
        else:
            self.assertTrue(False)

    def test_depth_layer_1_cpu(self):
        """ Test the input for the first input layer on the cpu. """
        self.depth_layer_1(device="cpu")

    def test_depth_layer_1_cuda(self):
        """ Test the input for the first input layer on the gpu. """
        self.depth_layer_1(device="cuda")

    def test_depth_layer_2(self, device):
        """ Test the input for the second input layer """
        # define input sequences
        seq = torch.tensor([2, 1, 1, 1, 2, 1, 1, 1], dtype=torch.long, device=device)
        depth = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long, device=device)
        pos = torch.tensor(
            [[16, 16], [16, 48], [48, 16], [48, 48], [8, 8], [8, 24], [24, 8], [24, 24]],
            dtype=torch.long,
            device=device
        )

        # define expected output sequences
        target_seq = [2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
        target_depth = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        target_pos = [
            [16, 16], [16, 48], [48, 16], [48, 48], [8, 8], [8, 24], [24, 8], [24, 24], [4, 4], [4, 12], [12, 4],
            [12, 12]
        ]
        target_future_tokens = 4

        # call the tested function
        output_seq, output_depth, output_pos, future_tokens = append_next_layer_tokens(seq, depth, pos, 2, 32)

        # compare output of the function with expected output
        if device == "cpu":
            np_test.assert_array_equal(output_seq, target_seq)
            np_test.assert_array_equal(output_depth, target_depth)
            np_test.assert_array_equal(output_pos, target_pos)
            self.assertEqual(future_tokens, target_future_tokens)
        elif device == "cuda":
            np_test.assert_array_equal(output_seq.cpu(), target_seq)
            np_test.assert_array_equal(output_depth.cpu(), target_depth)
            np_test.assert_array_equal(output_pos.cpu(), target_pos)
            self.assertEqual(future_tokens.cpu(), target_future_tokens)
        else:
            self.assertTrue(False)

    def test_depth_layer_2_cpu(self):
        """ Test the input for the second input layer on the cpu. """
        self.depth_layer_2(device="cpu")

    def test_depth_layer_2_cuda(self):
        """ Test the input for the second input layer on the gpu. """
        self.depth_layer_2(device="cuda")

    def correct_return_types(self, device):
        """ Test if the function returns the correct output type.

        The embedding layer allows for only tensors of type long.
        """
        # define input sequences
        seq = torch.tensor([1, 1, 1, 1], dtype=torch.long, device=device)
        depth = torch.tensor([1, 1, 1, 1], dtype=torch.long, device=device)
        pos = torch.tensor([[16, 16], [16, 48], [48, 16], [48, 48]], dtype=torch.long, device=device)

        # call the tested function
        output_seq, output_depth, output_pos, future_tokens = append_next_layer_tokens(seq, depth, pos, 2, 32)

        # compare output of the function with expected output
        if device == "cpu":
            self.assertEqual(output_seq.type(), "torch.LongTensor")
            self.assertEqual(output_depth.type(), "torch.LongTensor")
            self.assertEqual(output_pos.type(), "torch.LongTensor")
            self.assertEqual(future_tokens.type(), "torch.LongTensor")
        elif device == "cuda":
            self.assertEqual(output_seq.type(), "torch.cuda.LongTensor")
            self.assertEqual(output_depth.type(), "torch.cuda.LongTensor")
            self.assertEqual(output_pos.type(), "torch.cuda.LongTensor")
            self.assertEqual(future_tokens.type(), "torch.cuda.LongTensor")
        else:
            self.assertTrue(False)

    def test_correct_return_types_cpu(self):
        """ Test if the function returns the correct output type on the cpu. """
        self.correct_return_types(device="cpu")

    def test_correct_return_types_cuda(self):
        """ Test if the function returns the correct output type on the gpu. """
        self.correct_return_types(device="cuda")


class TestPrepareInputForNextLayer_Spatial3(unittest.TestCase):
    """ Tests if the sequence preparation function creates correct depth and position input sequences on the cpu. """
    def depth_layer_0(self, device):
        """ Test the input for the empty sequence. """
        # define input sequences
        seq = torch.tensor([], dtype=torch.long, device=device)
        depth = torch.tensor([], dtype=torch.long, device=device)
        pos = torch.tensor([], dtype=torch.long, device=device)

        # define expected output sequences
        target_seq = [1, 1, 1, 1, 1, 1, 1, 1]
        target_depth = [1, 1, 1, 1, 1, 1, 1, 1]
        target_pos = [
            [16, 16, 16],
            [16, 16, 48],
            [16, 48, 16],
            [16, 48, 48],
            [48, 16, 16],
            [48, 16, 48],
            [48, 48, 16],
            [48, 48, 48],
        ]
        target_future_tokens = 8

        # call the tested function
        output_seq, output_depth, output_pos, future_tokens = append_next_layer_tokens(seq, depth, pos, 3, 32)

        # compare output of the function with expected output
        if device == "cpu":
            np_test.assert_array_equal(output_seq, target_seq)
            np_test.assert_array_equal(output_depth, target_depth)
            np_test.assert_array_equal(output_pos, target_pos)
            self.assertEqual(future_tokens, target_future_tokens)
        elif device == "cuda":
            np_test.assert_array_equal(output_seq.cpu(), target_seq)
            np_test.assert_array_equal(output_depth.cpu(), target_depth)
            np_test.assert_array_equal(output_pos.cpu(), target_pos)
            self.assertEqual(future_tokens.cpu(), target_future_tokens)
        else:
            self.assertTrue(False)

    def test_depth_layer_0_cpu(self):
        """ Test the input for the empty sequence on the cpu. """
        self.depth_layer_0(device="cpu")

    def test_depth_layer_0_cuda(self):
        """ Test the input for the empty sequence on the gpu. """
        self.depth_layer_0(device="cuda")

    def depth_layer_1(self, device):
        """ Test the input for the first input layer """
        # define input sequences
        seq = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2], dtype=torch.long, device=device)
        depth = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long, device=device)
        pos = torch.tensor(
            [
                [16, 16, 16],
                [16, 16, 48],
                [16, 48, 16],
                [16, 48, 48],
                [48, 16, 16],
                [48, 16, 48],
                [48, 48, 16],
                [48, 48, 48],
            ],
            dtype=torch.long,
            device=device
        )
        # define expected output sequences
        target_seq = [
            2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]
        target_depth = [
            1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
        ]
        target_pos = [
            [16, 16, 16], [16, 16, 48], [16, 48, 16], [16, 48, 48], [48, 16, 16], [48, 16, 48], [48, 48, 16],
            [48, 48, 48], [8, 8, 8], [8, 8, 24], [8, 24, 8], [8, 24, 24], [24, 8, 8], [24, 8, 24], [24, 24, 8],
            [24, 24, 24], [8, 8, 40], [8, 8, 56], [8, 24, 40], [8, 24, 56], [24, 8, 40], [24, 8, 56], [24, 24, 40],
            [24, 24, 56], [8, 40, 8], [8, 40, 24], [8, 56, 8], [8, 56, 24], [24, 40, 8], [24, 40, 24], [24, 56, 8],
            [24, 56, 24], [8, 40, 40], [8, 40, 56], [8, 56, 40], [8, 56, 56], [24, 40, 40], [24, 40, 56], [24, 56, 40],
            [24, 56, 56], [40, 8, 8], [40, 8, 24], [40, 24, 8], [40, 24, 24], [56, 8, 8], [56, 8, 24], [56, 24, 8],
            [56, 24, 24], [40, 8, 40], [40, 8, 56], [40, 24, 40], [40, 24, 56], [56, 8, 40], [56, 8, 56], [56, 24, 40],
            [56, 24, 56], [40, 40, 8], [40, 40, 24], [40, 56, 8], [40, 56, 24], [56, 40, 8], [56, 40, 24], [56, 56, 8],
            [56, 56, 24], [40, 40, 40], [40, 40, 56], [40, 56, 40], [40, 56, 56], [56, 40, 40], [56, 40, 56],
            [56, 56, 40], [56, 56, 56]
        ]
        target_future_tokens = 64

        # call the tested function
        output_seq, output_depth, output_pos, future_tokens = append_next_layer_tokens(seq, depth, pos, 3, 32)

        # compare output of the function with expected output
        if device == "cpu":
            np_test.assert_array_equal(output_seq, target_seq)
            np_test.assert_array_equal(output_depth, target_depth)
            np_test.assert_array_equal(output_pos, target_pos)
            self.assertEqual(future_tokens, target_future_tokens)
        elif device == "cuda":
            np_test.assert_array_equal(output_seq.cpu(), target_seq)
            np_test.assert_array_equal(output_depth.cpu(), target_depth)
            np_test.assert_array_equal(output_pos.cpu(), target_pos)
            self.assertEqual(future_tokens.cpu(), target_future_tokens)
        else:
            self.assertTrue(False)

    def test_depth_layer_1_cpu(self):
        """ Test the input for the first input layer on cpu. """
        self.depth_layer_1(device="cpu")

    def test_depth_layer_1_cuda(self):
        """ Test the input for the first input layer on gpu. """
        self.depth_layer_1(device="cuda")

    def depth_layer_2(self, device):
        """ Test the input for the second input layer """
        # define input sequences
        seq = torch.tensor([2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long, device=device)
        depth = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], dtype=torch.long, device=device)
        pos = torch.tensor(
            [
                [16, 16, 16], [16, 16, 48], [16, 48, 16], [16, 48, 48], [48, 16, 16], [48, 16, 48], [48, 48, 16],
                [48, 48, 48], [8, 8, 8], [8, 8, 24], [8, 24, 8], [8, 24, 24], [24, 8, 8], [24, 8, 24], [24, 24, 8],
                [24, 24, 24]
            ],
            dtype=torch.long,
            device=device
        )
        # define expected output sequences
        target_seq = [2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        target_depth = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
        target_pos = [
            [16, 16, 16], [16, 16, 48], [16, 48, 16], [16, 48, 48], [48, 16, 16], [48, 16, 48], [48, 48, 16],
            [48, 48, 48], [8, 8, 8], [8, 8, 24], [8, 24, 8], [8, 24, 24], [24, 8, 8], [24, 8, 24], [24, 24, 8],
            [24, 24, 24], [4, 4, 4], [4, 4, 12], [4, 12, 4], [4, 12, 12], [12, 4, 4], [12, 4, 12], [12, 12, 4],
            [12, 12, 12]
        ]
        target_future_tokens = 8

        # call the tested function
        output_seq, output_depth, output_pos, future_tokens = append_next_layer_tokens(seq, depth, pos, 3, 32)

        # compare output of the function with expected output
        if device == "cpu":
            np_test.assert_array_equal(output_seq, target_seq)
            np_test.assert_array_equal(output_depth, target_depth)
            np_test.assert_array_equal(output_pos, target_pos)
            self.assertEqual(future_tokens, target_future_tokens)
        elif device == "cuda":
            np_test.assert_array_equal(output_seq.cpu(), target_seq)
            np_test.assert_array_equal(output_depth.cpu(), target_depth)
            np_test.assert_array_equal(output_pos.cpu(), target_pos)
            self.assertEqual(future_tokens.cpu(), target_future_tokens)
        else:
            self.assertTrue(False)

    def test_depth_layer_2_cpu(self):
        """ Test the input for the second input layer on cpu. """
        self.depth_layer_2(device="cpu")

    def test_depth_layer_2_cuda(self):
        """ Test the input for the second input layer on gpu. """
        self.depth_layer_2(device="cuda")

    def correct_return_types(self, device):
        """ Test if the function returns the correct output type.

        The embedding layer allows for only tensors of type long.
        """
        # define input sequences
        seq = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2], dtype=torch.long, device=device)
        depth = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long, device=device)
        pos = torch.tensor(
            [
                [16, 16, 16], [16, 16, 48], [16, 48, 16], [16, 48, 48], [48, 16, 16], [48, 16, 48], [48, 48, 16],
                [48, 48, 48]
            ],
            dtype=torch.long,
            device=device
        )

        # call the tested function
        output_seq, output_depth, output_pos, future_tokens = append_next_layer_tokens(seq, depth, pos, 3, 32)

        # compare output of the function with expected output
        if device == "cpu":
            self.assertEqual(output_seq.type(), "torch.LongTensor")
            self.assertEqual(output_depth.type(), "torch.LongTensor")
            self.assertEqual(output_pos.type(), "torch.LongTensor")
            self.assertEqual(future_tokens.type(), "torch.LongTensor")
        elif device == "cuda":
            self.assertEqual(output_seq.type(), "torch.cuda.LongTensor")
            self.assertEqual(output_depth.type(), "torch.cuda.LongTensor")
            self.assertEqual(output_pos.type(), "torch.cuda.LongTensor")
            self.assertEqual(future_tokens.type(), "torch.cuda.LongTensor")
        else:
            self.assertTrue(False)

    def test_correct_return_types_cpu(self):
        """ Test if the function returns the correct output type on the cpu. """
        self.correct_return_types(device="cpu")

    def test_correct_return_types_cuda(self):
        """ Test if the function returns the correct output type on the gpu. """
        self.correct_return_types(device="cuda")
