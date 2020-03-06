import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import truncated_normal, random_normal, glorot_normal, zeros


class SampledSoftmaxLayer(Layer):
    def __init__(self, target_song_size, target_emb_size, num_sampled=2, **kwargs):
        self.num_sampled = num_sampled
        self.target_song_size = target_song_size
        self.target_emb_size = target_emb_size
        super(SampledSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.song_embedding = self.add_weight(shape=[self.target_song_size, self.target_emb_size],
                                              initializer=truncated_normal(0, 0.1),
                                              dtype=tf.float32,
                                              name="song_embedding")
        self.zero_bias = self.add_weight(shape=[self.target_song_size],
                                         initializer=zeros,
                                         dtype=tf.float32,
                                         trainable=False,
                                         name="bias")
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_label_idx, label_idx=None, training=None, **kwargs):
        """
        The first input should be the model as it were, and the second the
        target (i.e., a repeat of the training data) to compute the labels
        argument

        """
        inputs, label_idx = inputs_with_label_idx
        # the labels input to this function is batch size by 1, where the
        # value at position (i, 1) is the index that is true (not zero)
        # e.g., (0, 0, 1) => (2) or (0, 1, 0, 0) => (1)
        return K.nn.sampled_softmax_loss(weights=self.song_embedding,
                                         biases=self.zero_bias,
                                         labels=label_idx,
                                         inputs=inputs,
                                         num_sampled=self.num_sampled,
                                         num_classes=self.target_song_size
                                         )

    def get_config(self, ):
        config = {'target_song_size': self.target_song_size, 'num_sampled': self.num_sampled}
        base_config = super(SampledSoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DotProductAttentionLayer(Layer):
    def __init__(self, shape, scope="attention", mask=True, has_W=True, pow_p=1, **kwargs):
        self.scope = scope
        self.shape = shape
        self.mask = mask
        self.has_W = has_W
        self.pow_p = pow_p
        super(DotProductAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        self.W = self.add_weight(shape=self.shape,
                                 initializer=truncated_normal(0, 0.1),
                                 dtype=tf.float32,
                                 name="W")
        super(DotProductAttentionLayer, self).build(input_shape)

    def call(self, inputs, seq_length=None, max_len=4, get_max=False, out_name="out",
             **kwargs):  # key:[B,H,E1], query:[B,E2,1]
        keys, query = inputs
        if self.mask:
            if seq_length is None or max_len is None:
                raise ValueError("seq_length and max_len must be provided if use mask")

        if self.has_W:
            weight = tf.transpose(tf.matmul(tf.tensordot(keys, self.W, axes=1), query), [0, 2, 1])
        else:
            weight = tf.transpose(tf.matmul(keys, query), [0, 2, 1])

        if self.mask:
            seq_mask = tf.sequence_mask(seq_length, max_len)
            padding = tf.ones_like(seq_mask, dtype=tf.float32) * (-2 ** 16 + 1)
            weight_tmp = tf.where(seq_mask, weight, padding, name="weight_tmp")
            weight = tf.pow(weight_tmp, self.pow_p)

        weight = tf.nn.softmax(weight, axis=-1, name="weight")  # [B,1,H]

        if get_max:
            indices = tf.argmax(weight, -1)  # [B,1]  ArgMax
            weight = tf.one_hot(indices, max_len)  # [B,1,H]

        output = tf.reshape(tf.matmul(weight, keys), [-1, keys.get_shape().as_list()[-1]])  # Reshape

        return output

    def compute_output_shape(self, input_shape):
        # todo
        return None, 8


class CapsuleLayer(Layer):
    def __init__(self, input_units, out_units, max_len, k_max, iteration=3,
                 weight_initializer=random_normal(stddev=1.0), **kwargs):
        self.input_units = input_units  # E1
        self.out_units = out_units  # E2
        self.max_len = max_len
        self.k_max = k_max
        self.iteration = iteration
        self.weight_initializer = weight_initializer
        super(CapsuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.B_matrix = self.add_weight(shape=[1, self.k_max, self.max_len], initializer=self.weight_initializer,
                                        trainable=False, name="B", dtype=tf.float32)  # [1,K,H]
        self.S_matrix = self.add_weight(shape=[self.input_units, self.out_units], initializer=self.weight_initializer,
                                        name="S", dtype=tf.float32)
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):  # seq_len:[B,1]
        low_capsule, seq_len = inputs
        B = tf.shape(low_capsule)[0]
        seq_len_tile = tf.tile(seq_len, [1, self.k_max])  # [B,K]

        for i in range(self.iteration):
            mask = tf.sequence_mask(seq_len_tile, self.max_len)  # [B,K,H]
            pad = tf.ones_like(mask, dtype=tf.float32) * (-2 ** 16 + 1)  # [B,K,H]
            B_tile = tf.tile(self.B_matrix, [B, 1, 1])  # [B,K,H]
            B_mask = tf.where(mask, B_tile, pad)
            W = tf.nn.softmax(B_mask)  # [B,K,H]
            low_capsule_new = tf.tensordot(low_capsule, self.S_matrix, axes=1)  # [B,H,E2]
            high_capsule_tmp = tf.matmul(W, low_capsule_new)  # [B,K,E2]
            high_capsule = squash(high_capsule_tmp)  # [B,K,E2]

            # ([B,K,E2], [B,H,E2]->[B,E2,H])->[B,K,H]->[1,K,H]
            B_delta = tf.reduce_sum(
                tf.matmul(high_capsule, tf.transpose(low_capsule_new, perm=[0, 2, 1])),
                axis=0, keep_dims=True
            )  # [1,K,H]
            self.B_matrix.assign_add(B_delta)

        return high_capsule


def squash(inputs):
    vec_squared_norm = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-9)
    vec_squashed = scalar_factor * inputs  # element-wise
    return vec_squashed
