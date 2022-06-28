# Inference Openvino

Version : openvino_2020.3.194

___
## 22.06.28 Update
### 1. ModelFreezer_keras.py 
  - Automatic Model Freezing Code for Keras model
  
#### 1.1. Usage
 ```
 python ModelFreezer_keras.py --keras_model_path {%keras_model_path} # path where keras models located
 ```
 
 #### 1.2. Result
   - Output   : Frozen graph model
   - Save Path: {%keras_model_path}/Frozenmodels/
  
  
