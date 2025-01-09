from analysis import *
from plotting import *
import warnings
import unittest

class PlotTests(unittest.TestCase):

    def test_calcAndPlotWilcoxonBoxplots(self):
        # sanity check
        data_size = 10
        data1 = np.ones(data_size)
        data2 = data1*2
        xlabel1 = 'data1'
        xlabel2 = 'data2'
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))

        w = calcAndPlotWilcoxonBoxplot(data1, data2, data_size, xlabel1, xlabel2, color1='blue', color2='green', fig=fig, ax=axs)
        self.assert_(w.pvalue < 0.5)

        # sanity check
        data_size = 10
        data1 = np.random.rand(data_size)
        data2 = np.random.rand(data_size)
        xlabel1 = 'data1'
        xlabel2 = 'data2'
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))

        w = calcAndPlotWilcoxonBoxplot(data1, data2, data_size, xlabel1, xlabel2, color1='blue', color2='green', fig=fig, ax=axs)
        self.assert_(w.pvalue > 0.5)

if __name__ == "__main__":
    unittest.main()
