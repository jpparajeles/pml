from pml.data import model
from pml.supervised.decision_trees import id3

__author__ = 'josep'
import numpy as np

from pml.supervised.decision_trees.id3 import build_tree
from pml.data.model import DataSet
from pandas import DataFrame, Series
from pml.supervised.decision_trees.tree_plotting import MatplotlibAnnotationTreePlotter

import pandas as pd
"""
df = DataFrame.from_csv("./test/datasets/3f_header.csv")
ds = DataSet(df, labels=["x","y","z","label"])
#print(ds)
t= build_tree(ds)
tp = MatplotlibAnnotationTreePlotter(t)
tp.plot()
"""
has_ids=True
has_header=True
has_labels=True
delimiter=","

header = 0 if has_header else None
id_col = 0 if has_ids else None

#dataframe = pd.read_csv("./test/datasets/play_tennis.data", index_col=id_col, header=header,delimiter=delimiter)
dataframe = pd.read_csv("hope.csv", index_col=id_col, header=header,delimiter=delimiter)

labels = dataframe.pop(dataframe.columns[-1]) if has_labels else None

dataset = model.DataSet(dataframe, labels=labels)
tree = id3.build_tree(dataset)

tp = MatplotlibAnnotationTreePlotter(tree)
tp.plot()