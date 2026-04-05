import tensorrt as trt
import pycuda.driver as cuda
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
        self.context = self.engine.create_execution_context()
        
        # Allocate Buffers (Pinned Memory)
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            size = trt.volume(self.engine.get_tensor_shape(name))
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
        # 1. Preprocess (Letterbox)
        img, ratio, (dw, dh) = self.preprocess(bgr_img)
        
        # 2. Infer
        np.copyto(self.inputs[0]['host'], img.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
        
        # 3. Postprocess (NMS + Rescale)
        # Assuming YOLO output format: [batch, boxes, elements]
        data = self.outputs[0]['host'].reshape(1, -1, 6) 
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
        # Filter and Rescale
        valid = detections[detections[:, 4] > conf_threshold]
        if len(valid) == 0: return []
        
        # Simple rescaling back to original image
        valid[:, [0, 2]] = (valid[:, [0, 2]] - dw) / ratio
        valid[:, [1, 3]] = (valid[:, [1, 3]] - dh) / ratio
        return valid