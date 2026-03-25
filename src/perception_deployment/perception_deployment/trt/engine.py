import tensorrt as trt

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    return engine, context