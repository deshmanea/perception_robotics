import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

with open("yolo.engine", "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)

print("Engine loaded:", engine is not None)
print("Number of bindings:", engine.num_bindings)