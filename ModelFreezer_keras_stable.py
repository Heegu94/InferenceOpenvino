import os, glob
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.framework import graph_io
import argparse

# 필요한 API 불러오기

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras_model_path", 
                        help = "Save Path where keras model located",
                        type = str,
                        default = os.getcwd()+"/")
    args = parser.parse_args()
    return args

def freeze_graph(session, output, save_dir, save_fname, save_pb_as_text=False):
    with session.graph.as_default():
        graph_def_inf = tf.graph_util.remove_training_nodes(session.graph.as_graph_def())
        graph_def_frozen = tf.graph_util.convert_variables_to_constants(session, graph_def_inf, output)
        graph_io.write_graph(graph_def_frozen, save_dir, save_fname, as_text = save_pb_as_text)
        return graph_def_frozen

def freezing_model(kmodel_path):
    tf.keras.backend.set_learning_phase(0)
    model = load_model(kmodel_path, compile=False)
    session = tf.keras.backend.get_session()

    input_node = [t.op.name for t in model.inputs]
    output_node = [t.op.name for t in model.outputs]

    print(input_node, output_node)

    sfname = kmodel_path.split("\\")[-1][:-3]+"_FrozenModel.pb"
    outputs = [t.op.name for t in model.outputs]
    return session, outputs, save_dir, sfname
    
if __name__=="__main__":  
    args = get_argparse()
    base_model_folder = args.keras_model_path
    save_dir = base_model_folder+"/Frozenmodels/"
    os.makedirs(save_dir, exist_ok=True)
    
    keras_model_list = glob.glob(base_model_folder+"/*.h5")
    print("base_model_folder: ",base_model_folder)
    print("************** Detected models **************")
    print(keras_model_list)
    
    for kmodel in keras_model_list:
        session, outputs, save_dir, sfname = freezing_model(kmodel)
        freeze_graph(session, outputs, save_dir, sfname)
        print("")
        
        del session
        tf.keras.backend.clear_session()
    