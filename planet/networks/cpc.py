import tensorflow as tf

def cross_entropy_loss(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred, dim=2)
    return tf.reduce_mean(loss)

def network_prediction(context, code_size, predict_terms, name='image'):

    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(tf.layers.dense(context, units=code_size, name=name + '_' + 'z_t_{i}'.format(i=i)))

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
    batch_size = context.shape[0].value
    effective_horizon = context.shape[1].value - predict_terms
    embedding_size = embedding.shape[-1].value
    context_to_use = context[:, :-predict_terms, :]
    x = tf.reshape(context_to_use, [-1] + context_to_use.shape[2:].as_list())

    positives = tf.zeros(shape=(batch_size * effective_horizon, 0, embedding_size))
    for i in range(predict_terms):
        positives = tf.concat([positives,
                              tf.reshape(embedding[:, i : i + effective_horizon], shape=(-1, 1, embedding_size))], axis=1)

    flattened_embedding = tf.reshape(embedding, shape=(-1, embedding_size))
    indexes = tf.random.uniform(shape=(batch_size * effective_horizon, predict_terms, negative_samples),
                                maxval=flattened_embedding.shape[0].value, dtype=tf.dtypes.int32)
    negatives = tf.gather(flattened_embedding, indexes)

    y_true = tf.concat([positives[:, :, None], negatives], axis = 2)

    return x, y_true

def format_cpc_action(x, y, negative_samples):
    x = tf.reshape(x, [-1] + x.shape[2:].as_list())
    positives = tf.reshape(y, (-1, 1, y.shape[-1].value))

    flattened_y = tf.reshape(y, shape=(-1, y.shape[-1].value))
    indexes = tf.random.uniform(shape=(flattened_y.shape[0].value, 1, negative_samples),
                                maxval=flattened_y.shape[0].value, dtype=tf.dtypes.int32)

    negatives = tf.gather(flattened_y, indexes)

    y_true = tf.concat([positives[:, :, None], negatives], axis=2)

    return x, y_true


def calc_acc(labels, logits):
    correct_class = tf.argmax(logits, axis=-1)
    predicted_class = tf.argmax(labels, axis=-1)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(correct_class, predicted_class), tf.int32)) / \
                    tf.size(correct_class)
    return accuracy

def cpc(context, graph, predict_terms=3, negative_samples=5):
    """
    :param context: shape = (batch_size, chunk_length, context_size)
    :param embedding: shape = (batch_size, chunk_length, embedding_size)
    :return: cross entropy loss
    """
    # x, preds, y_true
    embedding = graph.embedded
    reward = graph.data['reward'][:, :, None]
    x, y_true = format_cpc_data(context, embedding, predict_terms, negative_samples)
    _, reward_y_true = format_cpc_data(context, reward, predict_terms, negative_samples)

    code_size = embedding.shape[-1].value

    preds = network_prediction(x, code_size, predict_terms)
    reward_preds = network_prediction(x, 1, predict_terms, name='reward')

    logits = cpc_layer(preds, y_true)
    reward_logits = cpc_layer(reward_preds, reward_y_true)

    labels_zero = tf.zeros(dtype=tf.float32, shape=(x.shape[0], predict_terms, negative_samples))
    labels_one = tf.ones(dtype=tf.float32, shape=(x.shape[0], predict_terms, 1))
    labels = tf.concat([labels_one, labels_zero], axis=-1)

    loss = cross_entropy_loss(labels, logits)
    acc = calc_acc(labels, logits)

    reward_loss = cross_entropy_loss(labels, reward_logits)
    reward_acc = calc_acc(labels, reward_logits)

    return loss, acc, reward_loss, reward_acc

def inverse_model(context, graph, contrastive=True, negative_samples=10):
    embedding = graph.embedded
    actions = graph.data['action']
    x_context = context[:, :-2, :]
    x_embedding = embedding[:, 2:, :]
    x = tf.concat([x_context, x_embedding], axis=-1)
    y_action = actions[:, 1:-1, :]

    if contrastive:
        x, y_true = format_cpc_action(x, y_action, negative_samples)
        preds = network_prediction(x, y_true.shape[-1].value, 1, name="inverse_model")

        logits = cpc_layer(preds, y_true)
        labels_zero = tf.zeros(dtype=tf.float32, shape=(x.shape[0], 1, negative_samples))
        labels_one = tf.ones(dtype=tf.float32, shape=(x.shape[0], 1, 1))
        labels = tf.concat([labels_one, labels_zero], axis=-1)

        loss = cross_entropy_loss(labels, logits)
        acc = calc_acc(labels, logits)

        return loss, acc

    else:
        x_flattened = tf.reshape(x, (-1, x.shape[-1].value))
        y_flattened = tf.reshape(y_action, (-1, y_action.shape[-1].value))

        hidden = tf.layers.dense(x_flattened, units=1024, activation='relu')
        hidden = tf.layers.dense(hidden, units=1024, activation='relu')
        y_pred = tf.layers.dense(hidden, units=y_flattened.shape[-1].value)
        loss = tf.reduce_mean(tf.square(y_flattened - y_pred))
        return loss, None



