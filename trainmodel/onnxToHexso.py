# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Deploy a Framework-prequantized Model with TVM - Part 3 (TFLite)
================================================================
**Author**: `Siju Samuel <https://github.com/siju-samuel>`_

Welcome to part 3 of the Deploy Framework-Prequantized Model with TVM tutorial.
In this part, we will start with a Quantized TFLite graph and then compile and execute it via TVM.


For more details on quantizing the model using TFLite, readers are encouraged to
go through `Converting Quantized Models
<https://www.tensorflow.org/lite/convert/quantization>`_.

The TFLite models can be downloaded from this `link
<https://www.tensorflow.org/lite/guide/hosted_models>`_.

To get started, Tensorflow and TFLite package needs to be installed as prerequisite.

.. code-block:: bash

    # install tensorflow and tflite
    pip install tensorflow==2.1.0
    pip install tflite==2.1.0

Now please check if TFLite package is installed successfully, ``python -c "import tflite"``

"""

###############################################################################
# Necessary imports
# -----------------
import os
import numpy as np
import onnx
import tf2onnx
import tvm
import array as arr
from tvm import relay
import tensorflow as tf
import tvm.contrib.hexagon
from onnxruntime.quantization import quantize_dynamic, QuantType
#######################################################################
# Get a real image for e2e testing
# --------------------------------
#def get_real_image(im_height, im_width):

#    import array as arr
#    a = arr.array('f')
#    a.fromfile(open("/local/mnt2/workspace/arangasa/tvm_10June/tvm/artifacts/resnet/panda.raw", "rb"), 150528)
#    data = np.reshape(a, (1, im_height, im_width, 3))
#    return data


#data = get_real_image(224, 224)

###############################################################################
# TVM compilation and inference
# -----------------------------

###############################################################################
# We use the TFLite-Relay parser to convert the TFLite pre-quantized graph into Relay IR. Note that
# frontend parser call for a pre-quantized model is exactly same as frontend parser call for a FP32
# model. We encourage you to remove the comment from print(mod) and inspect the Relay module. You
# will see many QNN operators, like, Requantize, Quantize and QNN Conv2D.
dtype_dict = {"conv2d_input": "float32"}
shape_dict = {"conv2d_input": (1,28,28,1)}

onnx_graph = tf2onnx.tfonnx.process_tf_graph(None, opset=11, tflite_path="./model.tflite")
onnx_graph = tf2onnx.optimizer.optimize_graph(onnx_graph, catch_errors=True)
onnx_model = onnx_graph.make_model("tf2onnx")
onnx.save(onnx_model, "model.onnx")
quantize_dynamic("./model.onnx", "./model.quant.onnx", weight_type=QuantType.QUInt8, op_types_to_quantize=['Conv'])
quantized_model = onnx.load("./model.quant.onnx")

mod, params = relay.frontend.from_onnx(quantized_model, shape_dict)

print("relay model\n")
print(mod)

###############################################################################
# Lets now the compile the Relay module. We use the "llvm" target here. Please replace it with the
# target platform that you are interested in.
@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

target = tvm.target.hexagon("v68", link_params=True) #"llvm"
with tvm.transform.PassContext(opt_level=3, instruments=[PrintIR()]):
    lib = relay.build(mod, target, target_host=target, params=params, mod_name="default")
lib.get_lib().save('kerasmodel.float32.so')     # save .so
with open('kerasmodel.float32.json', 'w') as f:
    f.write(lib.get_graph_json())  # save .json
