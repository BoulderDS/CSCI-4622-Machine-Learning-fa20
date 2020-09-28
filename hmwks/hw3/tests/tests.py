import gzip
import pickle
import sys
import unittest
from functools import partial
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F


class TestNetwork(unittest.TestCase):
    def __init__(self, test, netw_ctor):
        super(TestNetwork, self).__init__(test)
        self.netw_ctor = netw_ctor

    def setUp(self):
        f = gzip.open('./data/tinyTOY.pkl.gz', 'rb')
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        self.train, self.test = u.load()

        if self.netw_ctor.__name__ == 'TorchNetwork':
            def to_torch(data):
                return [tuple(torch.from_numpy(e) for e in sample) for sample in data]
            # convert data to torch.
            self.train = to_torch(self.train)
            self.test = to_torch(self.test)

    def maybe_convert_model(self, model):
        for i, w in enumerate(model.weights):
            if isinstance(w, torch.Tensor):
                model.weights[i] = w.cpu().detach().numpy()
        for i, b in enumerate(model.biases):
            if isinstance(b, torch.Tensor):
                model.biases[i] = b.cpu().detach().numpy()

    def TestBackPropWithoutRegularization(self):
        # =====================================================
        # BackProp test without regularization
        # =====================================================
        np.random.seed(1234)
        nn_noreg = self.netw_ctor([2, 3, 2])
        nn_noreg.SGD_train(self.train, epochs=5, eta=0.25, lam=0.0, verbose=False)

        self.maybe_convert_model(nn_noreg)

        self.assertAlmostEqual(nn_noreg.weights[0][0, 0], 1.1273524638442189)
        self.assertAlmostEqual(nn_noreg.weights[0][0, 1], 0.73193288623969166)
        self.assertAlmostEqual(nn_noreg.weights[0][1, 0], -0.5195223324824767)
        self.assertAlmostEqual(nn_noreg.weights[0][1, 1], -0.16092443719267965)
        self.assertAlmostEqual(nn_noreg.weights[0][2, 0], -1.3585942451104391)
        self.assertAlmostEqual(nn_noreg.weights[0][2, 1], 1.2061099232068802)

        self.assertAlmostEqual(nn_noreg.biases[0][0][0], 0.74862652735377944)
        self.assertAlmostEqual(nn_noreg.biases[0][1][0], -1.1461180927834171)
        self.assertAlmostEqual(nn_noreg.biases[0][2][0], 0.5440188204156966)

        self.assertAlmostEqual(nn_noreg.weights[1][0, 0], 0.9903972225237162)
        self.assertAlmostEqual(nn_noreg.weights[1][1, 0], -0.092195162733213751)
        self.assertAlmostEqual(nn_noreg.weights[1][0, 1], 0.98122432833675222)
        self.assertAlmostEqual(nn_noreg.weights[1][1, 1], 0.2374002896289274)
        self.assertAlmostEqual(nn_noreg.weights[1][0, 2], -1.3028262866313183)
        self.assertAlmostEqual(nn_noreg.weights[1][1, 2], 0.49036035842372594)

        self.assertAlmostEqual(nn_noreg.biases[1][0][0], -0.0978044220637386)
        self.assertAlmostEqual(nn_noreg.biases[1][1][0], -0.25298179466851223)

    def TestBackPropWithRegularization(self):
        # =====================================================
        # BackProp test with regularization
        # =====================================================
        nn_reg = self.netw_ctor([2, 3, 2])
        nn_reg.SGD_train(self.train, epochs=5, eta=0.25, lam=0.2, verbose=False)

        self.maybe_convert_model(nn_reg)

        self.assertAlmostEqual(nn_reg.weights[0][0, 0], 0.0023322490027254854)
        self.assertAlmostEqual(nn_reg.weights[0][0, 1], 0.00094433912729247342)
        self.assertAlmostEqual(nn_reg.weights[0][1, 0], 0.0025152220763984155)
        self.assertAlmostEqual(nn_reg.weights[0][1, 1], 0.0010184838227073763)
        self.assertAlmostEqual(nn_reg.weights[0][2, 0], 0.0014913503824642196)
        self.assertAlmostEqual(nn_reg.weights[0][2, 1], 0.00060372407945523312)

        self.assertAlmostEqual(nn_reg.biases[0][0][0], 0.22180045340307644)
        self.assertAlmostEqual(nn_reg.biases[0][1][0], 0.7585097820793677)
        self.assertAlmostEqual(nn_reg.biases[0][2][0], -0.51429045271149121)

        self.assertAlmostEqual(nn_reg.weights[1][0, 0], -0.033370023956083815)
        self.assertAlmostEqual(nn_reg.weights[1][1, 0], 0.033367780844884856)
        self.assertAlmostEqual(nn_reg.weights[1][0, 1], -0.04093765557203885)
        self.assertAlmostEqual(nn_reg.weights[1][1, 1], 0.040934904514374378)
        self.assertAlmostEqual(nn_reg.weights[1][0, 2], -0.022491021177273487)
        self.assertAlmostEqual(nn_reg.weights[1][1, 2], 0.022489509526011812)

        self.assertAlmostEqual(nn_reg.biases[1][0][0], -0.055260479604691159)
        self.assertAlmostEqual(nn_reg.biases[1][1][0], 0.055274680302746619)
        
    def TestBackPropDropoutWithoutRegularization(self):
        # =====================================================
        # BackProp test without regularization
        # =====================================================
        np.random.seed(1234)
        nn_noreg = self.netw_ctor([2, 3, 2])
        nn_noreg.keep_prob = 0.5
        nn_noreg.SGD_train(self.train, epochs=50, eta=0.25, lam=0.0, verbose=False)

        self.maybe_convert_model(nn_noreg)

        self.assertAlmostEqual(nn_noreg.weights[0][0, 0], 2.98976038)
        self.assertAlmostEqual(nn_noreg.weights[0][0, 1], 2.12709538)
        self.assertAlmostEqual(nn_noreg.weights[0][1, 0], -0.17437938)
        self.assertAlmostEqual(nn_noreg.weights[0][1, 1], -0.35940241)
        self.assertAlmostEqual(nn_noreg.weights[0][2, 0], -5.93761951)
        self.assertAlmostEqual(nn_noreg.weights[0][2, 1], 3.18810481)

        self.assertAlmostEqual(nn_noreg.biases[0][0][0], -2.85175585)
        self.assertAlmostEqual(nn_noreg.biases[0][1][0], -2.15305128)
        self.assertAlmostEqual(nn_noreg.biases[0][2][0], -3.87458461)

        self.assertAlmostEqual(nn_noreg.weights[1][0, 0], -1.24989762)
        self.assertAlmostEqual(nn_noreg.weights[1][1, 0], 1.31121427)
        self.assertAlmostEqual(nn_noreg.weights[1][0, 1], -0.02962254)
        self.assertAlmostEqual(nn_noreg.weights[1][1, 1], 0.09929439)
        self.assertAlmostEqual(nn_noreg.weights[1][0, 2], -2.27559092)
        self.assertAlmostEqual(nn_noreg.weights[1][1, 2], 2.44883373)

        self.assertAlmostEqual(nn_noreg.biases[1][0][0], 0.23375274)
        self.assertAlmostEqual(nn_noreg.biases[1][1][0], -0.49388613)
        

class TestActiv(unittest.TestCase):
    def __init__(self, test, activ_fns):
        super().__init__(test)
        self.activ_fns = activ_fns

    def TestActivSingleInput(self, fn, torch_fn, _x):
        user = [fn(_xi) for _xi in _x]
        solution = torch_fn(torch.from_numpy(_x))
        np.testing.assert_array_almost_equal(user, solution)

    def TestActivMultiInput(self, fn, torch_fn, _x):
        user = fn(_x)
        solution = torch_fn(torch.from_numpy(_x))
        np.testing.assert_array_almost_equal(user, solution)

    def TestRelu(self):
        xs = np.random.rand(10)
        self.TestActivSingleInput(self.activ_fns[0], F.relu, xs)
        with patch('torch.nn.functional.relu') as mock_activ:
            self.activ_fns[0](xs[0])
        mock_activ.assert_not_called()

    def TestSigmoid(self):
        xs = np.random.rand(10)
        self.TestActivSingleInput(self.activ_fns[0], F.relu, xs)
        with patch('torch.sigmoid') as mock_activ:
            self.activ_fns[1](xs[0])
        mock_activ.assert_not_called()

    def TestSoftmax(self):
        xs = np.random.rand(10)
        self.TestActivSingleInput(self.activ_fns[0], F.relu, xs)
        with patch('torch.sigmoid') as mock_activ:
            self.activ_fns[2](xs)
        mock_activ.assert_not_called()


class TestLoss(unittest.TestCase):
    def __init__(self, test, loss_fns):
        super().__init__(test)
        self.loss_fns = loss_fns

    def TestActivSingleInput(self, fn, torch_fn, _x):
        user = [fn(_xi) for _xi in _x]
        solution = torch_fn(torch.from_numpy(_x))
        np.testing.assert_array_almost_equal(user, solution)

    def TestActivMultiInput(self, fn, torch_fn, _x):
        user = fn(_x)
        solution = torch_fn(torch.from_numpy(_x))
        np.testing.assert_array_almost_equal(user, solution)

    def TestMSE(self):
        y, y_hat = np.random.rand(10), np.random.rand(10)
        np.testing.assert_array_almost_equal(
            self.loss_fns[0](y_hat, y),
            F.mse_loss(torch.from_numpy(y_hat), torch.from_numpy(y)))
        with patch('torch.nn.functional.mse_loss') as mock_loss:
            self.loss_fns[0](y_hat, y)
        mock_loss.assert_not_called()

    def TestMAE(self):
        y, y_hat = np.random.rand(10), np.random.rand(10)
        np.testing.assert_array_almost_equal(
            self.loss_fns[1](y_hat, y),
            F.l1_loss(torch.from_numpy(y_hat), torch.from_numpy(y)))
        with patch('torch.nn.functional.l1_loss') as mock_loss:
            self.loss_fns[1](y_hat, y)
        mock_loss.assert_not_called()

    def TestHinge(self):
        # for grading - ignore
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        assert self.loss_fns[2](y_pred, y_true) == 0.25


def run_test_suite(name, ctor):
    if name == "prob 2.3":
        prob3 = unittest.TestSuite()
        for test in ["TestRelu", "TestSigmoid", "TestSoftmax"]:
            prob3.addTest(TestActiv(test, ctor))
        assert unittest.TextTestRunner(verbosity=2).run(
            prob3).wasSuccessful(), "one or more tests for prob 2.3 failed"
    elif name == "prob 2.4":
        prob3 = unittest.TestSuite()
        for test in ["TestMSE", "TestMAE", "TestHinge"]:
            prob3.addTest(TestLoss(test, ctor))
        assert unittest.TextTestRunner(verbosity=2).run(
            prob3).wasSuccessful(), "one or more tests for prob 2.4 failed"
    elif name == "prob 3":
        prob3 = unittest.TestSuite()
        for test in ["TestBackPropWithoutRegularization", "TestBackPropWithRegularization"]:
            prob3.addTest(TestNetwork(test, ctor))
        assert unittest.TextTestRunner(verbosity=2).run(
            prob3).wasSuccessful(), "one or more tests for prob 3 failed"
    elif name == "prob 4":
        prob4 = unittest.TestSuite()
        for test in [
                "TestBackPropDropoutWithoutRegularization"]:
            prob4.addTest(TestNetwork(test, ctor))
        assert unittest.TextTestRunner(verbosity=2).run(
            prob4).wasSuccessful(), "one or more tests for prob 4 failed"
    elif name == "prob 5":
        prob5 = unittest.TestSuite()
        for test in [
                "TestBackPropWithoutRegularization", "TestBackPropWithRegularization"]:
            prob5.addTest(TestNetwork(test, ctor))
        assert unittest.TextTestRunner(verbosity=2).run(
            prob5).wasSuccessful(), "one or more tests for prob 5 failed"
    else:
        raise Exception('unrecognized test suite name: {}'.format(name))
