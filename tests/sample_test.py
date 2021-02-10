import unittest
import torch
import numpy.testing as np_test

from utils.sample import append_next_layer_tokens


class TestPrepareInputForNextLayer_CPU(unittest.TestCase):
    """ Tests if the sequence preparation function creates correct depth and position input sequences on the cpu. """
    def test_depth_layer_0(self):
        """ Test the input for the empty sequence. """
        # define input sequences
        seq = torch.tensor([]).long().cpu()
        depth = torch.tensor([]).long().cpu()
        pos_x = torch.tensor([]).long().cpu()
        pos_y = torch.tensor([]).long().cpu()

        # define expected output sequences
        target_seq = [0]
        target_depth = [1]
        target_pos_x = [32]
        target_pos_y = [32]
        target_future_tokens = 1

        # call the tested function
        output_seq, output_depth, output_pos_x, output_pos_y, future_tokens = append_next_layer_tokens(
            seq, depth, pos_x, pos_y
        )

        # compare output of the function with expected output
        np_test.assert_array_equal(output_seq, target_seq)
        np_test.assert_array_equal(output_depth, target_depth)
        np_test.assert_array_equal(output_pos_x, target_pos_x)
        np_test.assert_array_equal(output_pos_y, target_pos_y)
        self.assertEqual(future_tokens, target_future_tokens)

    def test_depth_layer_1(self):
        """ Test the input for the first input layer """
        # define input sequences
        seq = torch.tensor([2]).long().cpu()
        depth = torch.tensor([1]).long().cpu()
        pos_x = torch.tensor([32]).long().cpu()
        pos_y = torch.tensor([32]).long().cpu()

        # define expected output sequences
        target_seq = [2, 0, 0, 0, 0]
        target_depth = [1, 2, 2, 2, 2]
        target_pos_x = [32, 16, 48, 16, 48]
        target_pos_y = [32, 16, 16, 48, 48]
        target_future_tokens = 4

        # call the tested function
        output_seq, output_depth, output_pos_x, output_pos_y, future_tokens = append_next_layer_tokens(
            seq, depth, pos_x, pos_y
        )

        # compare output of the function with expected output
        np_test.assert_array_equal(output_seq, target_seq)
        np_test.assert_array_equal(output_depth, target_depth)
        np_test.assert_array_equal(output_pos_x, target_pos_x)
        np_test.assert_array_equal(output_pos_y, target_pos_y)
        self.assertEqual(future_tokens, target_future_tokens)

    def test_depth_layer_2(self):
        """ Tests if the sequence preparation function creates correct depth and position input sequences. """
        # define input sequences
        seq = torch.tensor([2, 2, 2, 2, 2]).long().cpu()
        depth = torch.tensor([1, 2, 2, 2, 2]).long().cpu()
        pos_x = torch.tensor([32, 16, 48, 16, 48]).long().cpu()
        pos_y = torch.tensor([32, 16, 16, 48, 48]).long().cpu()

        # define expected output sequences
        target_seq = [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        target_depth = [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        target_pos_x = [32, 16, 48, 16, 48, 8, 24, 8, 24, 40, 56, 40, 56, 8, 24, 8, 24, 40, 56, 40, 56]
        target_pos_y = [32, 16, 16, 48, 48, 8, 8, 24, 24, 8, 8, 24, 24, 40, 40, 56, 56, 40, 40, 56, 56]
        target_future_tokens = 16

        # call the tested function
        output_seq, output_depth, output_pos_x, output_pos_y, future_tokens = append_next_layer_tokens(
            seq, depth, pos_x, pos_y
        )

        # compare output of the function with expected output
        np_test.assert_array_equal(output_seq, target_seq)
        np_test.assert_array_equal(output_depth, target_depth)
        np_test.assert_array_equal(output_pos_x, target_pos_x)
        np_test.assert_array_equal(output_pos_y, target_pos_y)
        self.assertEqual(future_tokens, target_future_tokens)

    def test_correct_return_types(self):
        """ Test if the function returns the correct output type.

        The embedding layer allows for only tensors of type long.
        """
        # define input sequences
        seq = torch.tensor([2]).long().cpu()
        depth = torch.tensor([1]).long().cpu()
        pos_x = torch.tensor([32]).long().cpu()
        pos_y = torch.tensor([32]).long().cpu()

        # call the tested function
        output_seq, output_depth, output_pos_x, output_pos_y, future_tokens = append_next_layer_tokens(
            seq, depth, pos_x, pos_y
        )

        # compare output of the function with expected output
        self.assertEqual(output_seq.type(), "torch.LongTensor")
        self.assertEqual(output_depth.type(), "torch.LongTensor")
        self.assertEqual(output_pos_x.type(), "torch.LongTensor")
        self.assertEqual(output_pos_y.type(), "torch.LongTensor")


class TestPrepareInputForNextLayer_CUDA(unittest.TestCase):
    """ Tests if the sequence preparation function creates correct depth and position input sequences on CUDA. """
    def test_depth_layer_0(self):
        """ Test the input for the empty sequence. """
        # define input sequences
        seq = torch.tensor([]).long().cuda()
        depth = torch.tensor([]).long().cuda()
        pos_x = torch.tensor([]).long().cuda()
        pos_y = torch.tensor([]).long().cuda()

        # define expected output sequences
        target_seq = [0]
        target_depth = [1]
        target_pos_x = [32]
        target_pos_y = [32]
        target_future_tokens = 1

        # call the tested function
        output_seq, output_depth, output_pos_x, output_pos_y, future_tokens = append_next_layer_tokens(
            seq, depth, pos_x, pos_y
        )

        # compare output of the function with expected output
        np_test.assert_array_equal(output_seq.cpu().numpy(), target_seq)
        np_test.assert_array_equal(output_depth.cpu().numpy(), target_depth)
        np_test.assert_array_equal(output_pos_x.cpu().numpy(), target_pos_x)
        np_test.assert_array_equal(output_pos_y.cpu().numpy(), target_pos_y)
        self.assertEqual(future_tokens, target_future_tokens)

    def test_depth_layer_1(self):
        """ Test the input for the first input layer """
        # define input sequences
        seq = torch.tensor([2]).long().cuda()
        depth = torch.tensor([1]).long().cuda()
        pos_x = torch.tensor([32]).long().cuda()
        pos_y = torch.tensor([32]).long().cuda()

        # define expected output sequences
        target_seq = [2, 0, 0, 0, 0]
        target_depth = [1, 2, 2, 2, 2]
        target_pos_x = [32, 16, 48, 16, 48]
        target_pos_y = [32, 16, 16, 48, 48]
        target_future_tokens = 4

        # call the tested function
        output_seq, output_depth, output_pos_x, output_pos_y, future_tokens = append_next_layer_tokens(
            seq, depth, pos_x, pos_y
        )

        # compare output of the function with expected output
        np_test.assert_array_equal(output_seq.cpu(), target_seq)
        np_test.assert_array_equal(output_depth.cpu(), target_depth)
        np_test.assert_array_equal(output_pos_x.cpu(), target_pos_x)
        np_test.assert_array_equal(output_pos_y.cpu(), target_pos_y)
        self.assertEqual(future_tokens, target_future_tokens)

    def test_depth_layer_2(self):
        """ Tests if the sequence preparation function creates correct depth and position input sequences. """
        # define input sequences
        seq = torch.tensor([2, 2, 2, 2, 2]).long().cuda()
        depth = torch.tensor([1, 2, 2, 2, 2]).long().cuda()
        pos_x = torch.tensor([32, 16, 48, 16, 48]).long().cuda()
        pos_y = torch.tensor([32, 16, 16, 48, 48]).long().cuda()

        # define expected output sequences
        target_seq = [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        target_depth = [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        target_pos_x = [32, 16, 48, 16, 48, 8, 24, 8, 24, 40, 56, 40, 56, 8, 24, 8, 24, 40, 56, 40, 56]
        target_pos_y = [32, 16, 16, 48, 48, 8, 8, 24, 24, 8, 8, 24, 24, 40, 40, 56, 56, 40, 40, 56, 56]
        target_future_tokens = 16

        # call the tested function
        output_seq, output_depth, output_pos_x, output_pos_y, future_tokens = append_next_layer_tokens(
            seq, depth, pos_x, pos_y
        )

        # compare output of the function with expected output
        np_test.assert_array_equal(output_seq.cpu(), target_seq)
        np_test.assert_array_equal(output_depth.cpu(), target_depth)
        np_test.assert_array_equal(output_pos_x.cpu(), target_pos_x)
        np_test.assert_array_equal(output_pos_y.cpu(), target_pos_y)
        self.assertEqual(future_tokens, target_future_tokens)

    def test_correct_return_types(self):
        """ Test if the function returns the correct output type.

        The embedding layer allows for only tensors of type long.
        """
        # define input sequences
        seq = torch.tensor([2]).long().cuda()
        depth = torch.tensor([1]).long().cuda()
        pos_x = torch.tensor([32]).long().cuda()
        pos_y = torch.tensor([32]).long().cuda()

        # call the tested function
        output_seq, output_depth, output_pos_x, output_pos_y, future_tokens = append_next_layer_tokens(
            seq, depth, pos_x, pos_y
        )

        # compare output of the function with expected output
        self.assertEqual(output_seq.type(), "torch.cuda.LongTensor")
        self.assertEqual(output_depth.type(), "torch.cuda.LongTensor")
        self.assertEqual(output_pos_x.type(), "torch.cuda.LongTensor")
        self.assertEqual(output_pos_y.type(), "torch.cuda.LongTensor")
