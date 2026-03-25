
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Load the engine
with open("../yolo.engine", "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

# Allocate GPU buffers
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding))
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    device_mem = cuda.mem_alloc(size * dtype().nbytes)
    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        inputs.append(device_mem)
    else:
        outputs.append(device_mem)

# Create a dummy input with correct shape (1x3x640x640)
dummy_input = np.random.random((1,3,640,640)).astype(np.float32)
cuda.memcpy_htod(inputs[0], dummy_input)

# Run inference
context.execute_v2(bindings)

# Retrieve output
output_shape = engine.get_binding_shape(1)
output_data = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh(output_data, outputs[0])

print("Dummy inference output shape:", output_data.shape)