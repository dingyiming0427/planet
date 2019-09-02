import keras
from keras import backend as K
import tensorflow as tf

def cross_entropy_loss(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred, dim=2)
    return tf.reduce_mean(loss)

def network_prediction(context, code_size, predict_terms):

    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(tf.layers.dense(context, units=code_size, name='z_t_{i}'.format(i=i)))

    # if len(outputs) == 1:
    #     output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    # else:
    output = tf.stack(outputs, axis=1)

    return output

def cpc_layer(preds, y_encoded):
    dot_product = preds[:, :, None, :] * y_encoded  # this should be broadcasted to N x T_pred x (negative_samples + 1) x code_size
    ret = tf.reduce_sum(dot_product, axis=-1)
    return ret

def format_cpc_data(context, embedding, predict_terms, negative_samples):
    x = tf.reshape(context, [-1] + context.shape[2:].as_list())
    y_true = tf.reshape(embedding, [-1] + embedding.shape[2:].as_list())[:, None, None, :]
    y_true = tf.tile(y_true, (1, predict_terms, negative_samples + 1, 1))

    return x, y_true


def cpc(context, embedding, predict_terms=3, negative_samples=5):
    """
    :param context: shape = (batch_size, chunk_length, context_size)
    :param embedding: shape = (batch_size, chunk_length, embedding_size)
    :return: cross entropy loss
    """
    # x, preds, y_true
    x, y_true = format_cpc_data(context, embedding, predict_terms, negative_samples)

    code_size = embedding.shape[-1].value

    preds = network_prediction(x, code_size, predict_terms)

    logits = cpc_layer(preds, y_true)

    labels_zero = tf.zeros(dtype=tf.float32, shape=(x.shape[0], predict_terms, negative_samples))
    labels_one = tf.ones(dtype=tf.float32, shape=(x.shape[0], predict_terms, 1))
    labels = tf.concat([labels_one, labels_zero], axis=-1)
    loss = cross_entropy_loss(labels, logits)

    return loss

