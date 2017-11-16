import tensorflow as tf
import os
import sys
import shutil

# Add FileWriter and Saver
# Check out graph
# SavedModel
# Load it and get d
# Work with placeholder so you feed
# Work with hardcoded file read (no placeholder, just write file content)
# SavedModel CLI

a = tf.constant([2], name='a')
b = tf.constant([3], name='b')
c = tf.Variable([0], name='c')
c = c.assign(b)
d = tf.multiply(a, c, name='mul_a_c')
e = tf.add(d, b, name='add_d_b')

print("***** PLEASE READ THIS")
print("This sample deletes the full directory from PATH (below)")
print("If you don't like that, then don't run this code")
print("As a safeguard, I am exiting this program now")
print("Comment that exit out once you're comfortable running this program")
if True:
    sys.exit()

PATH="/tmp2/savedmodel"
if os.path.isdir(PATH):
    assert len(PATH) > 8, "Safeguard, directory path too short"
    shutil.rmtree(PATH)



# Create the model and execute it so we get a checkpoint
sess = tf.Session()
sess.run(tf.global_variables_initializer())
fw = tf.summary.FileWriter(PATH, sess.graph)
s = tf.train.Saver()
print sess.run(d)
s.save(sess, PATH + os.sep + "model.chkp", 1)
fw.close()

def do_chkp_file():
    print("\n---------------------------------------------------------------")
    print("do_chkp_file")
    sess = tf.Session(tf.Graph())
    ckpt_state = tf.train.get_checkpoint_state(PATH)
    new_saver = tf.train.Saver()
    new_saver.restore(sess, ckpt_state.model_checkpoint_path)
    t = sess.graph.get_tensor_by_name("add_d_b:0")
    print(t.name)
    print(type(t))
    print sess.run(t)
    print(ckpt_state)

def do_meta_file():
    print("\n---------------------------------------------------------------")
    print("do_meta_file")
    sess = tf.Session(tf.Graph())
    chkp_state = tf.train.get_checkpoint_state(PATH)  # Returns <abs_path>/model.chkp-1
    new_saver = tf.train.import_meta_graph(chkp_state.model_checkpoint_path + os.sep + ".meta")
    new_saver.restore(sess, PATH + os.sep + "model.chkp-1")
    # Returns op but not any result
    #    op = sess.graph.get_operation_by_name("add_d_b")
    # In order to get result of op you need to get that tensor
    t = sess.graph.get_tensor_by_name("add_d_b:0")
    print(t.name)
    print(type(t))
    print sess.run(t)

def do_savedmodel_file(sess):
    print("\n---------------------------------------------------------------")
    print("do_savedmodel_file")

    SM_PATH = PATH + os.sep + "savedmodel"
    if os.path.isdir(SM_PATH):
        assert len(SM_PATH) > 8, "Safeguard, directory path too short"
        shutil.rmtree(SM_PATH)
    builder = tf.saved_model.builder.SavedModelBuilder(SM_PATH)

    # Build TensorInfo proto (copies dtype, shape, and name)
    model_tensor_output = tf.saved_model.utils.build_tensor_info(d)

    # Build SignatureDef proto ()
    # tf.saved_model.signature_def_utils has
    #    build_signature_def(inputs, outputs, method_name)
    #       Both inputs and outputs are { name : TensorInfo } dicts
    #       All below methods are convencience methods calling util.build_tensor_info on its tensors
    #       then calling this methods with the signature_constants
    #
    #    clasification_signature_def
    #       CLASSIFY_INPUTS, CLASSIFY_OUTPUT_CLASSES, CLASSIFY_OUTPUT_SCORES, CLASSIFY_METHOD_NAME
    #    predict_signature_def
    #       PREDICT_METHOD_NAME (no INPUT/OUTPUT exists)
    #    regression_signature_def
    #       REGRESS_INPUT, REGRESS_OUTPUT, REGRESS_METHOD_NAME
    model_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={},
        outputs={ tf.saved_model.signature_constants.PREDICT_OUTPUTS: model_tensor_output },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.global_variables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                       model_signature},
        legacy_init_op=legacy_init_op) # Initialize op after load op is finished
    builder.save()

    sess = tf.Session(graph=tf.Graph())
    print("Loading saved model")
    meta_graph_def = tf.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        SM_PATH)
    gd = meta_graph_def.graph_def
    tf.import_graph_def(gd)
    t = sess.graph.get_tensor_by_name("add_d_b:0")
    print(t.name)
    print(type(t))
    print sess.run(t)

do_chkp_file()
do_meta_file()
do_savedmodel_file(sess)
