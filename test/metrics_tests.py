# Copyright (C) 2012 David Rusk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to 
# deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
# IN THE SOFTWARE.
"""
Unit tests for metrics module.

@author: drusk
"""

import unittest
import pandas as pd
from model import DataSet
import metrics

class MetricsTest(unittest.TestCase):

    def create_dataset(self, labels):
        """
        Used to create a DataSet for testing purposes where you don't really 
        care what the actual data is, just the labels.
        """
        raw_data = [[1, 2, 3] for _ in xrange(len(labels))]
        return DataSet(raw_data, labels=labels)

    def test_accuracy_integer_index(self):
        results = pd.Series(["a", "b", "c", "a"])
        dataset = self.create_dataset(["a", "b", "c", "b"])
        self.assertAlmostEqual(metrics.compute_accuracy(results, dataset), 
                               0.75, places=2)
    
    def test_accuracy_string_index(self):
        results = pd.Series(["a", "b", "c", "a"], 
                            index=["V01", "V02", "V03", "V04"])
        labels = pd.Series(["a", "b", "c", "b"], 
                           index=["V01", "V02", "V03", "V04"])
        dataset = self.create_dataset(labels)
        self.assertAlmostEqual(metrics.compute_accuracy(results, dataset), 
                               0.75, places=2)
    
    def test_accuracy_unequal_lengths(self):
        results = pd.Series(["a", "b", "c", "a"], 
                            index=["V01", "V02", "V03", "V04"])
        labels = pd.Series(["a", "b", "c"], 
                           index=["V01", "V02", "V03"])
        dataset = self.create_dataset(labels)
        self.assertRaises(ValueError, metrics.compute_accuracy, results, 
                          dataset)
    
    def test_accuracy_dataset_unlabelled(self):
        results = pd.Series(["a", "b", "c", "a"], 
                            index=["V01", "V02", "V03", "V04"])
        dataset = DataSet([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.assertRaises(ValueError, metrics.compute_accuracy, results, 
                          dataset)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    