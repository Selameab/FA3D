import tensorflow as tf

from datasets.kitti.reader import Reader

# For distributed training
def TFDataset(subset, class_dict, batch_size, velo_encoder, target_encoder, shuffle, aug_fn=None, prefetch=0, num_parallel_calls=4):
    reader = Reader(class_dict)

    def _get_item(t):
        t = t[0].decode('UTF-8')

        # Read data
        pts = reader.get_velo_reduced(t)
        boxes = reader.get_boxes_3D(t)

        # Augment
        if aug_fn is not None:
            pts, boxes = aug_fn(pts, boxes, t)

        return [velo_encoder.encode(pts)] + target_encoder.encode(boxes)

    def _get_item_tf(t: tf.Tensor):
        result = tf.numpy_function(func=_get_item, inp=[[t]], Tout=velo_encoder.Tout + target_encoder.Tout)
        result[0] = tf.ensure_shape(result[0], velo_encoder.output_shape)
        for i in range(1, len(result)):
            result[i] = tf.ensure_shape(result[i], target_encoder.output_shapes[i - 1])
        return result[0], (*result[1:],)

    ids = reader.get_ids(subset)
    ds = tf.data.Dataset.from_tensor_slices(ids)
    if shuffle:
        ds = ds.shuffle(len(ids))
    ds = ds.map(_get_item_tf, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size)
    if prefetch > 0:
        ds = ds.prefetch(prefetch)
    return ds
