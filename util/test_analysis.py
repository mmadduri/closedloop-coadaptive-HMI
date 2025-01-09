from analysis import *
import warnings
import unittest

class CalcTests(unittest.TestCase):

    def test_calc_rms(self):
        # sanity check
        signal = np.array([1])
        rms = calc_rms(signal)
        self.assertEqual(rms, 0)

        # check dimensions are preserved
        signal = np.array([[0.5, -0.5], [1., -1.]]).T
        rms = calc_rms(signal)
        np.testing.assert_allclose(rms, np.array([0.5, 1.]))

        # check without remove offset
        signal = np.array([1])
        rms = calc_rms(signal, remove_offset=False)
        self.assertAlmostEqual(rms, 1.)
    
    def test_calc_time_domain_error(self):
        # sanity check
        # Simple 1D Test: X = 0, Y = +/- 1
        X = np.zeros((5, 1))
        Y = np.asarray([[1], [-1], [1], [-1], [-1]])
        ans = np.ones((5, )) # The distance between these coordinates should all be equal to 1
        calc_ans = calc_time_domain_error(X, Y)
        self.assertEqual(calc_ans.shape, ans.shape)
        np.testing.assert_array_equal(calc_time_domain_error(X, Y), ans)
        
        # try this in the 2D case: X = [0, 0]; Y = [+/- 1, +/- 1]
        X = np.zeros((5, 2))
        Y = np.asarray([[1, 1], [1, -1], [-1, 1], [-1, -1], [1, 1]])
        # The distance between all these coordinates should be equal to the sqrt(2) -- sqrt(1^2 + 1^2) = sqrt(2)
        ans = np.ones((5,))*np.sqrt(2)
        calc_ans = calc_time_domain_error(X, Y)
        self.assertEqual(calc_ans.shape, ans.shape)
        np.testing.assert_array_equal(calc_time_domain_error(X, Y), ans)

        # Also confirm that this calculation = sqrt( (X[0]-Y[0]^2) + (X[1]-Y[1]^2) )
        XY = np.sqrt((X[:, 0] - Y[:, 0])**2 + (X[:, 1] - Y[:, 1])**2) 
        np.testing.assert_array_equal(calc_time_domain_error(X, Y), XY)


        # confirms that this scales with Y = [+/- 2, +/- 2], so = sqrt(2^2 + 2^2)
        Y = Y*2
        ans = np.ones((5,))*np.sqrt(8)
        calc_ans = calc_time_domain_error(X, Y)
        self.assertEqual(calc_ans.shape, ans.shape)
        np.testing.assert_array_equal(calc_time_domain_error(X, Y), ans)

        # try this in the higher-D case: X = [0, 0]; Y = [+/- 1, +/- 1]
        X = np.zeros((5, 3))
        Y = np.asarray([[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1], [1, 1, 1]])
        # The distance between all these coordinates should be equal to the sqrt(3) 
        # sqrt(1^2 + 1^2 + 1^2) = sqrt(3)
        ans = np.ones((5,))*np.sqrt(3)
        calc_ans = calc_time_domain_error(X, Y)
        self.assertEqual(calc_ans.shape, ans.shape)
        np.testing.assert_array_equal(calc_time_domain_error(X, Y), ans)

    def test_estimate_encoder_simple(self):
        # sanity check

        # set up dimensions to match the expected data dimensions
        n_dim = 9 # 8 user input features
        n_time = 1000 # large 
        n_feat = 64 # number of EMG channels 

        ## TEST 1
        ## test if the y matrix is a perfect linear relationships between weights (A) and x
        x = np.random.rand(n_dim, n_time) # the user input - dimensions x time
        A = np.random.rand(n_feat, n_dim) # the weight matrix, to be estimated
        y = A@x # the synethesize EMG matrix

        A_est, intercept, r2_avg = estimate_encoder_linear(x.T, y.T, verbose = False)
        y_est = A_est@x + intercept[:, np.newaxis]

        self.assertEqual(np.allclose(A, A_est), True)
        self.assertEqual(A_est.shape, (n_feat, n_dim))
        self.assertEqual(np.allclose(y, y_est), True)
        self.assertEqual(r2_avg, 1.0)


    def test_estimate_encoder_intercept(self):
        # Create synthetic input data
        n_time, n_dim = 100, 8
        n_ch = 64
        x_data = np.random.rand(n_time, n_dim)
        weights = np.random.rand(n_ch, n_dim)
        intercept = np.random.rand(n_ch)
        y_data = np.dot(x_data, weights.T) + intercept  # Simulate data using known weights and intercept

        # Call the function to estimate the encoder
        estimated_weights, estimated_intercept, r2_avg = estimate_encoder_linear(x_data, y_data, n_ch)

        # Check if the estimated weights and intercept are of the correct shape
        self.assertEqual(estimated_weights.shape, (n_ch, n_dim))
        self.assertEqual(estimated_intercept.shape, (n_ch,))
        
        # Check if the R^2 score is reasonable (you can define a threshold)
        self.assertTrue(r2_avg >= 1.0)  # Adjust the threshold as needed

    def test_estimate_encoder_noisy(self):

        # set up dimensions to match the expected data dimensions
        n_dim = 8 # 8 user input features
        n_time = 1000 # large 
        n_feat = 64 # number of EMG channels 
        
        x = np.random.rand(n_dim, n_time) # user inputs
        A = np.random.rand(n_feat, n_dim) # weights matrix
        y = A@x + np.random.rand(n_feat, n_time) # additional noise to the estimates

        A_est, intercept, r2_avg = estimate_encoder_linear(x.T, y.T, verbose = False)
        y_est = A_est@x + intercept[:, np.newaxis]

        self.assertEqual(len(intercept), n_feat)

        self.assertFalse(np.allclose(A, A_est))

        self.assertEqual(A_est.shape, (n_feat, n_dim))
        self.assertFalse(np.allclose(y, y_est))

        # Check if the R^2 score is reasonable (you can define a threshold)
        self.assertTrue(r2_avg >= 0.5)  # Adjust the threshold as needed


       

if __name__ == "__main__":
    unittest.main()

