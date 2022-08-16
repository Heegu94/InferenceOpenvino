# Inference Openvino

Version : openvino_2020.3.194

___
## 22.08.16 Update
### 1. ModelFreezer.py 
  - Automatic Model Freezing Code for ".h5",  SavedModel format
  
#### 1.1. Usage
 ```
 python ModelFreezer.py --model_path {%model_path} --type {%model type}
 ```
 
 - --model_path : model path
 - --type :  if model is ".h5" type "h5",  else model is SavedModel format type "dir"
 
 
### 2. ModelFreezer_keras_stable.py 
- Automatic Model Freezing Code for Keras model 
- same code of V1 model

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
  
  
