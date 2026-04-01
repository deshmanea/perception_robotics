import tensorrt as trt
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        engine = builder.build_engine(network, config)

        if engine is None:
            raise RuntimeError("Engine build failed")

        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

        print("Engine built")

build_engine(
    "/robotics_ws/src/perception_deployment/perception_deployment/yolo26n.onnx",
    "/robotics_ws/src/perception_deployment/perception_deployment/yolo.engine"
)