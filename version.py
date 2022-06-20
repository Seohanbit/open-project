import tensorflow as tf
import numpy as np

g= tf.compat.v1.get_default_graph()
op_list=g.get_operations()
for op in op_list:
    print(op)