import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

class TensorRTInference:

    def __init__(self, engine_path, input_size=(640, 640)):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.input_size = input_size
        
        # Load Engine
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")
        
        self.stream = cuda.Stream()
        
        # Allocate Buffers (Pinned Memory)
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            
            # Handle dynamic shapes
            shape = self.engine.get_tensor_shape(name)

            if -1 in shape:
                shape = self.context.get_tensor_shape(name)

            size = trt.volume(shape)

            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem, 'name': name})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'name': name})
        return inputs, outputs, bindings

    def run(self, bgr_img, conf_threshold=0.5):
        img, ratio, (dw, dh) = self.preprocess(bgr_img)

        # Set dynamic shape
        input_name = self.engine.get_tensor_name(0)
        self.context.set_input_shape(input_name, (1, 3, self.input_size[0], self.input_size[1]))

        if self.inputs is None:
            self.inputs, self.outputs, self.bindings = self._allocate_buffers()
        
        np.copyto(self.inputs[0]['host'], img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

            # Bind memory
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        output = self.outputs[0]['host']
        print("Output size:", output.shape)

        data = output.reshape(1, -1, 6) 

        return self.postprocess(data[0], bgr_img.shape, ratio, dw, dh, conf_threshold)

    def preprocess(self, img):
        # Professional Letterboxing logic
        shape = img.shape[:2]

        r = min(self.input_size[0] / shape[0], self.input_size[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

        dw, dh = (self.input_size[1] - new_unpad[0]) / 2, (self.input_size[0] - new_unpad[1]) / 2

        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, int(round(dh - 0.1)), int(round(dh + 0.1)), 
                                 int(round(dw - 0.1)), int(round(dw + 0.1)), 
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = img.transpose((2, 0, 1))[::-1] # BGR to RGB and HWC to CHW
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        return img, r, (dw, dh)

    def postprocess(self, detections, orig_shape, ratio, dw, dh, conf_threshold):

        print("Actual raw shape:", detections.shape)
        print("Min/Max:", detections.min(), detections.max())
        print("First row:", detections[:1])

        detections = detections.reshape(-1, 6)

        print("Sample detections:\n", detections[:10])

        valid = detections[detections[:, 4] > conf_threshold]
        if len(valid) == 0:
            return []

        valid = valid.copy()

        valid[:, [0, 2]] = (valid[:, [0, 2]] - dw) / ratio
        valid[:, [1, 3]] = (valid[:, [1, 3]] - dh) / ratio

        return valid