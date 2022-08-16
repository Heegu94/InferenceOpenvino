from openvino.inference_engine import IENetwork, IECore, IEPlugin, IENetLayer

import os, glob
import numpy as np

class inference_openvino():
    def __init__(self, 
                 openvino_model_dir, 
                 device,
                 CAM_enabled = False,
                 num_classes = 1,
                 feature_layer_name = "activation_2/Relu",
                 weight_layer_name = "conv2d_3/BiasAdd/Add",
                ):
        self.openvino_model_dir = openvino_model_dir
        self.device             = device
        self.CAM_enabled        = CAM_enabled
        self.n_class            = num_classes
        self.feature_layer_name = feature_layer_name
        self.weight_layer_name  = weight_layer_name
    
    def load_test_images(self, ):
        ## load input images
        return 
    
    def preprocess(self, ):
        ## preprocessing input images
        return 
        
    def inference(self, imgs):
        Openvino_XML = glob.glob(self.openvino_model_dir + "/*.xml")[0]
        Openvino_BIN = glob.glob(self.openvino_model_dir + "/*.bin")[0]
        
        iec   = IECore()
        net   = iec.read_Network(model = Openvino_XML, weights = Openvino_BIN)
        
        ## CAM enabled
        if self.CAM_enabled:
            print("1. CAM Mode started ..")
            feature_layer_name = self.feature_layer_name
            weight_layer_name = self.weight_layer_name

            net.add_outputs(feature_layer_name) # add output layer
            weight_layer = net.layers[weight_layer_name] 
            layer_weights = weight_layer.blobs["weights"]
            
            in_blob   = next(iter(net.inputs))
            out_blobs = [l for l in net.outputs]
            
            model = iec.load_network(network = net, device_name = self.device)
            print("Model sucessfully loaded!")
            print("input node: ", input_node)
            print("output node: ", output_node)
            
            CAMs, predictions = [], []
            for img in imgs:
                result = model.infer(inputs={in_blob:img})
                # featuremap * weights = CAM
                featuremap        = result[feature_layer_name]
                feature_map       = np.transpose(feature_map, axes=[0,2,3,1])
                res_layer_weights = layer_weights.reshape(self.n_class, feature_map.shape[-1])
                CAM               = np.dot(feature_map[0], res_layer_weights[2])
                CAMs.append(CAM)
                
                # predictions
                out_blobs.pop(np.where(np.array(out_blobs) == feature_layer_name)[0][0])
                prediction = result[out_blobs[0]]
                predictions.append(prediction)

            return np.array(CAMS, dtype=np.float32), np.array(predictions, dtype=np.float32)
        
        ## CAM disabled - just predict the result
        else:
            print("2. CAM Mode started ..")
            in_blob  = next(iter(net.inputs))
            out_blob = next(iter(net.outputs))
            model    = iec.load_network(network = net, device_name = self.device)
            print("Model sucessfully loaded!")
            print("input node: ", input_node)
            print("output node: ", output_node)
            predictions = []
            for img in imgs:
                result     = model.infer(inputs={in_blob:img})
                prediction = result[out_blobs]
                predictions.append(prediction)
        
            return np.array(predictions, dtype=np.float32)
    
if __name__ == "__main__":
    openvino_model_dir = "C:/Users/JLK/Desktop/working/DRG_openvino/openvino_model/TB/yellow/"
    device = "CPU"
    num_class = 1
    feature_layer_name = "activation_2/Relu",
    weight_layer_name = "conv2d_3/BiasAdd/Add"
    
    InferenceOpenvino = inference_openvino(openvino_model_dir = openvino_model_dir,
                                           device=device, 
                                           feature_layer_name=feature_layer_name, 
                                           weight_layer_name=weight_layer_name)
    
    # test image's shape should be 
    images = InferenceOpenvino.load_test_images()
    preprocessed_images = InferenceOpenvino.preprocess(images)
    predictions = InferenceOpenvino.inference(preprocessed_images)
    
    print(predictions)