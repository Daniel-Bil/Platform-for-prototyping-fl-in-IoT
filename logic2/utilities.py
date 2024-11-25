import json

import tensorflow as tf
from pathlib import Path

def load_architectures(path: Path):
    print("load_architectures")
    paths = os.listdir(f"{os.getcwd()}\\..\\Backend\\architectureJsons")
    datas = []
    for idx, path in enumerate(paths):
        if idx<2:
            with open(f"{os.getcwd()}\\..\\Backend\\architectureJsons\\{path}", 'r') as file:
                data = json.load(file)[0]
                datas.append({"name": path.split(".")[0], "data": data})

    return datas

def load_methods():
    # return ["fedpaq_int", "fedpaq_float", "fedavg", "fedprox"]
    # return ["fedpaq_int", "fedavg", "fedpaq_float", "fedprox", "fedma"]
    return ["fedavg"]

def write_to_tensorboard(history_data, log_dir, start_step=0):
    # Create a TensorBoard summary writer
    writer = tf.summary.create_file_writer(log_dir)

    # Use the summary writer to log metrics, incrementing the step for each iteration
    with writer.as_default():
        for step, (acc, loss) in enumerate(zip(history_data['accuracy'], history_data['loss']), start=start_step):
            tf.summary.scalar('accuracy', acc, step=step)
            tf.summary.scalar('loss', loss, step=step)
            writer.flush()