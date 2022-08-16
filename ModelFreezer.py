import os, glob
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.framework import graph_io
import argparse

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np

# 필요한 API 불러오기

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", 
                        help = "Model path",
                        type = str
                       )
    parser.add_argument("--type", 
                        help = "Select the correct model's filename extension, for example 'h5' or 'dir'",
                        type = str,
                        default = "pb")
    args = parser.parse_args()
    return args

def freeze_graph(session, output, save_dir, save_fname, save_pb_as_text=False):
    with session.graph.as_default():
        graph_def_inf = tf.graph_util.remove_training_nodes(session.graph.as_graph_def())
        graph_def_frozen = tf.graph_util.convert_variables_to_constants(session, graph_def_inf, output)
        graph_io.write_graph(graph_def_frozen, save_dir, save_fname, as_text = save_pb_as_text)
        return graph_def_frozen

def freezing_model_h5(kmodel_path):
    tf.keras.backend.set_learning_phase(0)
    model = load_model(kmodel_path, compile=False)
    session = tf.keras.backend.get_session()

    input_node = [t.op.name for t in model.inputs]
    output_node = [t.op.name for t in model.outputs]

    print(input_node, output_node)

    sfname = kmodel_path.split("\\")[-1][:-3]+"_FrozenModel.pb"
    outputs = [t.op.name for t in model.outputs]
    return session, outputs, save_dir, sfname
    
def freezing_model_tf2(frozen_out_path):  #path of the directory where you want to save your model
    # name of the .pb file
    model = load_model(frozen_out_path) # tf >= 2.x 에서 모델을 저장한 디렉토리
    # Convert Keras model to Concrete Function
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # Get frozen graph def
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, # to pb
                      logdir=frozen_out_path,
                      name="frozen_graph.pb",
                      as_text=False)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, # to pbtxt
                      logdir=frozen_out_path,
                      name=f"frozen_graph.pbtxt",
                      as_text=True)    
    
if __name__=="__main__":  
    args = get_argparse()
    model_type      = args.type
    base_model_path = args.model_path
    
    save_dir = base_model_path+"/Frozenmodels/"
    os.makedirs(save_dir, exist_ok=True)
       
    if model_type.lower() == "h5":
        print("************** Detected model **************")
        print("Type: ", model_type)
        print("Input Model Path", keras_model_list)
        print("")
        print("Model Freezing....")
    
        session, outputs, save_dir, sfname = freezing_model_h5(base_model_path)
        freeze_graph(session, outputs, save_dir, sfname)
        print("")
        del session
        tf.keras.backend.clear_session()
        
    elif model_type.lower() == "dir":
        freezing_model_tf2(base_model_path)
    else:
        print("Check the model type!")
        
        

        
    