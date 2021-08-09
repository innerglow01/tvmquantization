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

dtype_dict = {"conv2d_input": "uint8"}
shape_dict = {"conv2d_input": (1,28,28,1)}

#creating a random input image array
x = np.random.randint(0,255, (28,28,1))
x = x/255.0
x = x.astype(np.float32)
x = np.expand_dims(x, axis=0)
data = x

def run_tvm(lib):
    from tvm.contrib import graph_executor

    rt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
    rt_mod.set_input("conv2d_input", data)
    rt_mod.run()
    tvm_res = rt_mod.get_output(0).asnumpy()
    return tvm_res, rt_mod

onnx_graph = tf2onnx.tfonnx.process_tf_graph(None, opset=11, tflite_path="./model.tflite")
onnx_graph = tf2onnx.optimizer.optimize_graph(onnx_graph, catch_errors=True)
onnx_model = onnx_graph.make_model("tf2onnx")
onnx.save(onnx_model, "model.onnx")
quantize_dynamic("./model.onnx", "./model.quant.onnx", weight_type=QuantType.QInt8, op_types_to_quantize=['Conv'])
quantized_model = onnx.load("./model.quant.onnx")

mod, params = relay.frontend.from_onnx(quantized_model, shape_dict)

print("relay model\n")
print(mod)


@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

###############################################################################
# Lets now the compile the Relay module.
# target platform that you are interested in.

target = tvm.target.create("llvm -mcpu=skylake-avx512") #"llvm"
with tvm.transform.PassContext(opt_level=3, instruments=[PrintIR()]):
    lib = relay.build(mod, target=target, target_host=target, params=params, mod_name="default")

tvm_res, rt_mod = run_tvm(lib)
print("Res", tvm_res.shape)
print(tvm_res)
