import numpy as np
from planet import tools
import tensorflow as tf

def cross_entropy_loss(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred, dim=2)
    return tf.reduce_mean(loss)

def network_prediction(context, code_size, predict_terms, name='image'):

    ''' Define the network mapping context to multiple embeddings '''
    context_shape = context.shape.as_list()
    outputs = []
    for i in range(predict_terms):
        output = tf.layers.dense(tf.reshape(context, shape=(-1, context_shape[-1])),
                                       units=code_size, name=name + '_' + 'z_t_{i}'.format(i=i))
        if len(context_shape) == 3:
            output = tf.reshape(output,shape=(context_shape[0], context_shape[1], code_size))
        outputs.append(output)

    # if len(outputs) == 1:
    #     output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    # else:
    output = tf.stack(outputs, axis=1)

    return output

def network_prediction_openloop(context, code_size, predict_terms, name='image'):

    ''' Define the network mapping context to multiple embeddings '''
    assert context.shape[1].value == predict_terms

    outputs = []
    context_shape = context[:, 0].shape.as_list()
    for i in range(predict_terms):
        context_for_curpred = context[:, i]
        output = tf.layers.dense(context_for_curpred, units=code_size, name=name + '_' + 'z_t_{i}'.format(i=i))
        if len(context_shape) == 3:
            output = tf.reshape(output, shape=(context_shape[0], context_shape[1], code_size))
        outputs.append(output)

    output = tf.stack(outputs, axis=1)

    return output

def cpc_layer(preds, y_encoded):
    dot_product = preds[:, :, None, :] * y_encoded  # this should be broadcasted to N x T_pred x (negative_samples + 1) x code_size
    ret = tf.reduce_sum(dot_product, axis=-1)
    return ret

def format_cpc_data(context_to_use, embedding, predict_terms, negative_samples, num_hard_negatives=0,
                    negative_actions=False):
    N = context_to_use.shape[0].value
    horizon = embedding.shape[1].value
    effective_horizon = horizon - predict_terms
    embedding_size = embedding.shape[-1].value
    x = context_to_use

    positives = tf.zeros(shape=(N, 0, embedding_size))
    for i in range(predict_terms):
        positives = tf.concat([positives,
                              tf.reshape(embedding[:, i + 1 : i + 1 + effective_horizon],
                                         shape=(-1, 1, embedding_size))], axis=1) # shape = N x predict_terms, embedding_size
    if negative_actions:
        return x, positives

    negatives_hard = tf.zeros(shape=(N, 0, num_hard_negatives // 2 * 2, embedding_size))
    if num_hard_negatives > 0:
        assert num_hard_negatives < negative_samples
        for i in range(predict_terms):
            negatives_hard_curi = tf.zeros(shape=(N, 0, embedding_size))
            for j in range(num_hard_negatives // 2):
                to_add = tf.reshape(tf.gather(embedding, np.mod(np.arange(i + 2 + j, i + 2 + j + effective_horizon), horizon), axis=1),
                                    shape=(-1, 1, embedding_size))
                negatives_hard_curi = tf.concat([negatives_hard_curi, to_add], axis=1)
                to_add = tf.reshape(tf.gather(embedding, np.mod(np.arange(i - 2 - j, i - 2 - j + effective_horizon), horizon), axis=1),
                                    shape=(-1, 1, embedding_size))
                negatives_hard_curi = tf.concat([negatives_hard_curi, to_add], axis=1)
            assert negatives_hard_curi.shape[1].value == num_hard_negatives // 2 * 2
            negatives_hard = tf.concat([negatives_hard, negatives_hard_curi[:, None]], axis=1)


    flattened_embedding = tf.reshape(embedding, shape=(-1, embedding_size))
    indexes = tf.random.uniform(shape=(N, predict_terms, negative_samples - num_hard_negatives // 2 * 2),
                                maxval=flattened_embedding.shape[0].value, dtype=tf.dtypes.int32)
    negatives_easy = tf.gather(flattened_embedding, indexes)
    
    if not num_hard_negatives:
        negatives = negatives_easy
    else:
        negatives = tf.concat([negatives_easy, negatives_hard], axis=2)

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

def cpc(context, graph, posterior, predict_terms=3, negative_samples=5, hard_negative_samples=0, stack_actions=False,
        negative_actions=False, cpc_openloop=False, gradient_penalty=False):
    """
    :param context: shape = (batch_size, chunk_length, context_size)
    :param embedding: shape = (batch_size, chunk_length, embedding_size)
    :return: cross entropy loss
    """
    # x, preds, y_true
    effective_horizon = context.shape[1].value - predict_terms
    embedding = graph.embedded
    actions = graph.data['action']
    if cpc_openloop:
        shape = tools.shape(actions)
        length = tf.tile(tf.constant(shape[1])[None], [shape[0]])
        context_to_use = get_overshoot_preds(graph, embedding, actions, length, predict_terms, posterior)
        context_to_use =  merge_first_two_dim(context_to_use) # shape = N x predict_terms x sample_size
        if negative_actions:
            context_to_use = context_to_use[:, :, None]
            for _ in range(negative_samples):
                random_actions = tf.random.uniform(actions.shape, minval=-1, maxval=1)
                negative_context = get_overshoot_preds(graph, embedding, random_actions, length, predict_terms, posterior)
                negative_context = merge_first_two_dim(negative_context)[:, :, None]
                context_to_use = tf.concat([context_to_use, negative_context], axis=2)

    else:
        context_to_use =  context[:, :-predict_terms, :]
        context_to_use = tf.reshape(context_to_use, [-1] + context_to_use.shape[2:].as_list())
        if stack_actions:
            future_actions = tf.stack([tf.reshape(actions[:, i:i+predict_terms], (actions.shape[0].value, -1))
                                        for i in range(effective_horizon)], axis=1)
            assert future_actions.shape[1].value == effective_horizon
            future_actions = tf.reshape(future_actions, shape=(-1, actions.shape[-1].value * predict_terms))

            if negative_actions:
                image_context = context_to_use
                context_to_use = tf.concat([image_context, future_actions], axis=-1)[:, None]
                for _ in range(negative_samples):
                    current_context = tf.concat([image_context, tf.random.uniform(future_actions.shape, minval=-1, maxval=1)], axis=-1)
                    context_to_use = tf.concat([context_to_use, current_context[:, None]], axis=1) # shape (N x negatives x action_dim)
            else:
                context_to_use = tf.concat([context_to_use, future_actions], axis=-1)

    reward = graph.data['reward'][:, :, None]
    x, y_true = format_cpc_data(context_to_use, embedding, predict_terms, negative_samples,
                                num_hard_negatives=hard_negative_samples, negative_actions=negative_actions)
    _, reward_y_true = format_cpc_data(context_to_use, reward, predict_terms, negative_samples,
                                       negative_actions=negative_actions)

    code_size = embedding.shape[-1].value

    if cpc_openloop:
        preds = network_prediction_openloop(x, code_size, predict_terms)
        reward_preds = network_prediction_openloop(x, 1, predict_terms, name='reward')
    else:
        preds = network_prediction(x, code_size, predict_terms)
        reward_preds = network_prediction(x, 1, predict_terms, name='reward')

    if negative_actions:
        logits = cpc_layer(y_true, preds)
        reward_logits = cpc_layer(reward_y_true, reward_preds)
    else:
        logits = cpc_layer(preds, y_true)
        reward_logits = cpc_layer(reward_preds, reward_y_true)

    labels_zero = tf.zeros(dtype=tf.float32, shape=(x.shape[0], predict_terms, negative_samples))
    labels_one = tf.ones(dtype=tf.float32, shape=(x.shape[0], predict_terms, 1))
    labels = tf.concat([labels_one, labels_zero], axis=-1)

    loss = cross_entropy_loss(labels, logits)
    acc = calc_acc(labels, logits)

    reward_loss = cross_entropy_loss(labels, reward_logits)
    reward_acc = calc_acc(labels, reward_logits)

    if gradient_penalty:
        gpenalty = tf.constant(0, dtype=tf.float32)

        # for i in range(predict_terms):
        #     for j in range(negative_samples + 1):
        #         grad = tf.gradients(logits[:, i, j], [x, y_true])
        #         grad_concat = tf.concat([tf.contrib.layers.flatten(grad[0]),
        #                                  tf.contrib.layers.flatten(grad[1][:, i, j])],
        #                                 axis=-1)
        #         gpenalty += tf.reduce_mean(tf.pow(tf.norm(grad_concat, axis=-1) - 1, 2))

        batch_size, horizon = graph.data['reward'].shape.as_list()
        effective_horizon = horizon - predict_terms
        f = tf.reshape(logits[:, :, 0], shape=(batch_size, effective_horizon, predict_terms))
        s_t = x
        o_tpk = graph.data['image']
        # import pdb;
        #
        # pdb.set_trace()
        counter = 0

        for k in range(predict_terms):
            current_f = f[:, :, k]
            grad0 = tf.reshape(tf.gradients(current_f, s_t, stop_gradients=[s_t])[0], shape=(batch_size, effective_horizon, -1))
            grad1 = tf.reshape(tf.gradients(current_f, o_tpk, stop_gradients=[s_t])[0][:, k + 1 : k + 1 + effective_horizon], shape=(batch_size, effective_horizon, -1))
            grad = tf.concat([grad0, grad1], axis=-1)
            gpenalty += tf.reduce_mean(tf.pow(tf.norm(grad, axis=-1) - 1, 2))
            print('grad done')
            counter += 1

        gpenalty /= counter

        return loss, acc, reward_loss, reward_acc, gpenalty

    return loss, acc, reward_loss, reward_acc, 0.

def inverse_model(context, graph, contrastive=True, negative_samples=10):
    embedding = graph.embedded
    actions = graph.data['action']
    x_context = context[:, :-1, :]
    x_embedding = embedding[:, 1:, :]
    x = tf.concat([x_context, x_embedding], axis=-1)
    y_action = actions[:, 1:, :]

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


def get_overshoot_preds(graph, embedding, actions, length, predict_terms, posterior):
    _, priors, posteriors, mask = tools.overshooting(
        graph.cell, {}, embedding, actions, length,
        predict_terms, posterior)
    posteriors, priors, mask = tools.nested.map(
        lambda x: x[:, :, 1:], (posteriors, priors, mask))


    context_to_use = priors['sample'][:, :-predict_terms]  # batch_size x effective_horizon x predict_terms x sample_size
    # TODO: is sample the right feature to use?
    
    return context_to_use

def merge_first_two_dim(tensor):
    return tf.reshape(tensor, shape=[-1] + tensor.shape[2:].as_list())

