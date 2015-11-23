# Based on https://github.com/drusk/pml
import pml

__author__ = 'josep'

from pml.data import model
from pml.supervised.decision_trees import id3
import numpy as np

from pml.supervised.decision_trees.id3 import build_tree

from pml.data.model import DataSet
from pandas import DataFrame, Series
from pml.supervised.decision_trees.tree_plotting import MatplotlibAnnotationTreePlotter

def bprint(tree):
    return bprint_aux(tree.get_root_node())

def bprint_aux(raiz, index=0):
    ret = ""
    ret += " "*(index-1)+">" + raiz.get_value() + "\n"
    index+=1
    for branch in raiz.get_branches():
        ret += " "*index + branch + "\n"
        child_node = raiz.get_child(branch)
        if child_node.is_leaf():
            ret += " "*(index+1) + child_node.get_value() + "\n"
        else:
            ret += bprint_aux(child_node, index+2)
    return ret








import pandas as pd

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

printbonito = bprint(tree)
print(printbonito)
txt = open("output.txt",mode="w", encoding="UTF-8")
txt.write(printbonito)
txt.close()
tp = MatplotlibAnnotationTreePlotter(tree)
tp.plot()

