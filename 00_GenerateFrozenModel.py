import tensorflow as tf # tensorflow version == 2.3.0 is used 
from tensorflow.keras.models import load_model
from tensorflow.python.framework import graph_io

import os

class Convert2FrozenModel:
    def __init__(self,
                 keras_model_path,
                 save_fname,
                 save_dir = "./results/frozen_models/",
                 ):
        self.keras_model_path = keras_model_path
        self.save_dir         = save_dir
        self.save_fname       = save_fname
        
    def get_session_output(self):
        tf.keras.backend.set_learning_phase(0)
        os.makedirs(self.save_dir, exist_ok=True)
        model   = load_model(self.keras_model_path, compile=False)
        session = tf.keras.backend.get_session()
        print("Model sucessfully loaded!")
    
        input_node  = [t.op.name for t in model.inputs]
        output_node = [t.op.name for t in model.outputs]
        
        print("input node: ", input_node)
        print("output node: ", output_node)
        
        return session, output_node
    
    def freeze_graph(self, session, output, save_pb_as_text=False):
        with session.graph.as_default():
            graph_def_inf = tf.graph_util.remove_training_nodes(session.graph.as_graph_def())
            graph_def_frozen = tf.graph_util.convert_variables_to_constants(session, graph_def_inf, output)
            graph_io.write_graph(graph_def_frozen, self.save_dir, self.save_fname, as_text = save_pb_as_text)
            return graph_def_frozen    
        
if __name__ == "__main__":
    # save_dir   = "./frozen_models/" # save dir
    save_fname = "sample.pb" # .pb
    keras_model_path = "C:/Users/JLK/Desktop/working/DRG_openvino/done/keras_model/Consolidation_Resize512_[V_AUC1]0.93735_[V_AUC2]0.87956_2.h5"
    
    convert2frozenmodel = Convert2FrozenModel(keras_model_path, save_fname)
    
    session, output = convert2frozenmodel.get_session_output()
    convert2frozenmodel.freeze_graph(session, output)