import tensorflow as tf

_EPSILON = tf.keras.backend.epsilon()

# Classification
def focal_loss(alpha, gamma):
    def focal_loss_fixed(y_true, y_pred):
        pos_mask = tf.cast(tf.equal(y_true, 1.0), tf.float32)
        neg_mask = tf.cast(tf.less(y_true, 1.0), tf.float32)

        pos_loss = -tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1.0 - _EPSILON)) * tf.pow(1 - y_pred, gamma) * pos_mask
        neg_loss = -tf.math.log(tf.clip_by_value(1 - y_pred, _EPSILON, 1.0 - _EPSILON)) * tf.pow(y_pred, gamma) * neg_mask

        num_pos = tf.reduce_sum(pos_mask)
        pos_loss = tf.reduce_sum(pos_loss) * alpha
        neg_loss = tf.reduce_sum(neg_loss) * (1 - alpha)

        return (pos_loss + neg_loss) / (num_pos + 1)

    return focal_loss_fixed


def extract_mask(y_true, y_pred):
    mask = tf.equal(y_true[..., -1], 1.0)
    pos_count = tf.reduce_sum(y_true[..., -1])
    y_true = y_true[..., 0:-1]
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return y_true, y_pred, pos_count


def smooth_L1_unmasked(sigma):
    def _smooth_L1_unmasked(y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        return tf.reduce_mean(tf.where(diff < sigma, 0.5 * tf.square(diff) / sigma, diff - 0.5 * sigma))

    return _smooth_L1_unmasked

# Regression
def smooth_L1_masked(sigma):
    _unmasked_fn = smooth_L1_unmasked(sigma)

    def _smooth_L1_masked(y_true, y_pred):
        y_true, y_pred, pos_count = extract_mask(y_true, y_pred)
        return tf.where(pos_count > 0, _unmasked_fn(y_true, y_pred), 0)

    return _smooth_L1_masked
