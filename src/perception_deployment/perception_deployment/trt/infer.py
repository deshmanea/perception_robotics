import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []

    for binding in engine:
        shape = engine.get_binding_shape(binding)
        size = int(np.prod(shape))
        dtype = np.float32

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append(device_mem)
        else:
            outputs.append(device_mem)

    return inputs, outputs, bindings


def infer(context, bindings, inputs, outputs, input_data):
    cuda.memcpy_htod(inputs[0], input_data)
    context.execute_v2(bindings)

    output = np.empty((1, 300, 6), dtype=np.float32)
    cuda.memcpy_dtoh(output, outputs[0])

    return output