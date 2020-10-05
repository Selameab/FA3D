import tensorflow as tf
import os


# Restricts GPU memory usage
def allow_growth():
    if tf.__version__.startswith("1"):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        tf.compat.v1.keras.backend.set_session(sess)
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)


# Creates directory and returns as string
def create_dir(d):
    os.makedirs(d, exist_ok=True)
    return d
