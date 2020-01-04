import tensorflow as tf
import os

meta_path = '/home/tata/Projects/hand_detector/checkpoints/model.ckpt-53332.meta' # Your .meta file
output_node_names = ['detection_boxes:0', 'detection_scores:0', 'detection_classes:0', 'num_detections:0']    # Output nodes

with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True)) as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,'/home/tata/Projects/hand_detector/checkpoints/model.ckpt-53332')
    print([n.name for n in tf.get_default_graph().as_graph_def().node])
    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())