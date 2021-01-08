import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, ortho_init
from baselines.common.mpi_running_mean_std import RunningMeanStd

mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk

def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

def build_impala_cnn(unscaled_images, depths=[16,32,32], **conv_kwargs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu, name='layer_' + get_layer_num_str())

    return out


@register("mlp")
def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h

    return network_fn


@register("cnn")
def cnn(**conv_kwargs):
    def network_fn(X):
        return nature_cnn(X, **conv_kwargs)
    return network_fn

@register("impala_cnn")
def impala_cnn(**conv_kwargs):
    def network_fn(X):
        return build_impala_cnn(X)
    return network_fn

@register("cnn_small")
def cnn_small(**conv_kwargs):
    def network_fn(X):
        h = tf.cast(X, tf.float32) / 255.

        activ = tf.nn.relu
        h = activ(conv(h, 'c1', nf=8, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
        h = activ(conv(h, 'c2', nf=16, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
        h = conv_to_fc(h)
        h = activ(fc(h, 'fc1', nh=128, init_scale=np.sqrt(2)))
        return h
    return network_fn

@register("lstm")
def lstm(nlstm=128, layer_norm=False):
    """
    Builds LSTM (Long-Short Term Memory) network to be used in a policy.
    Note that the resulting function returns not only the output of the LSTM
    (i.e. hidden state of lstm for each step in the sequence), but also a dictionary
    with auxiliary tensors to be set as policy attributes.

    Specifically,
        S is a placeholder to feed current state (LSTM state has to be managed outside policy)
        M is a placeholder for the mask (used to mask out observations after the end of the episode, but can be used for other purposes too)
        initial_state is a numpy array containing initial lstm state (usually zeros)
        state is the output LSTM state (to be fed into S at the next call)


    An example of usage of lstm-based policy can be found here: common/tests/test_doc_examples.py/test_lstm_example

    Parameters:
    ----------

    nlstm: int          LSTM hidden state size

    layer_norm: bool    if True, layer-normalized version of LSTM is used

    Returns:
    -------

    function that builds LSTM with a given input tensor / placeholder
    """

    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        h = tf.layers.flatten(X)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn


@register("cnn_lstm")
def cnn_lstm(nlstm=128, layer_norm=False, conv_fn=nature_cnn, **conv_kwargs):
    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        h = conv_fn(X, **conv_kwargs)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn

@register("impala_cnn_lstm")
def impala_cnn_lstm():
    return cnn_lstm(nlstm=256, conv_fn=build_impala_cnn)

@register("cnn_lnlstm")
def cnn_lnlstm(nlstm=128, **conv_kwargs):
    return cnn_lstm(nlstm, layer_norm=True, **conv_kwargs)


@register("conv_only")
def conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    '''
    convolutions-only net

    Parameters:
    ----------

    conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer.

    Returns:

    function that takes tensorflow tensor as input and returns the output of the last convolutional layer

    '''

    def network_fn(X):
        out = tf.cast(X, tf.float32) / 255.
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = tf.contrib.layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu,
                                           **conv_kwargs)

        return out
    return network_fn


@register("neural_map")
def neural_map(**nm_kwargs):
    def neural_map_fn(X, nenv=1):

        def global_read(nm, c_dim):
            # input: neural map (nm)
            # output: c-dimensional global read vector (r)

            # 2 conv layer 1 fc layer
            activ = tf.nn.relu

            hidden_0 = activ(conv(nm, 'gr_c1', nf=8, rf=3, stride=1, init_scale=np.sqrt(2)))
            #hidden_1 = activ(conv(hidden_0, 'gr_c2', nf=8, rf=3, stride=2, init_scale=np.sqrt(2)))

            hidden_2 = tf.layers.flatten(hidden_0)

            r = activ(fc(hidden_2, 'gr_fc1', nh=c_dim, init_scale=np.sqrt(2)))

            return r

        def context_read(nm, s_flat, r, c_dim):
            # input: neural map (nm), flattened state (s_flat), r
            # output: c-dimensional context read vector (c)

            scope = 'cr'

            with tf.variable_scope(scope):
                input = tf.concat([s_flat, r], 1)

                no_rows_W = input.get_shape()[1].value
                W = tf.get_variable("W", [no_rows_W, c_dim], initializer=ortho_init(1.))

                batch_size = s_flat.shape[0]
                nm_reshaped = tf.reshape(nm, [batch_size, -1, c_dim])

                q = tf.matmul(input, W)
                a = tf.keras.backend.batch_dot(nm_reshaped, q, (2, 1))
                a_exp = tf.math.exp(a)
                norm_fac = tf.reduce_sum(a_exp, 1)
                norm_fac_expanded = tf.expand_dims(norm_fac, -1)
                alpha = tf.math.divide(a_exp, norm_fac_expanded)
                alpha_expanded = tf.expand_dims(alpha, -1)
                nm_scored = tf.math.multiply(alpha_expanded, nm_reshaped)
                c = tf.reduce_sum(nm_scored, 1)

                return c

        def local_write(s_flat, r, c, nm_xy, c_dim):
            # input: flattened state (s_flat), r, c, neural map's feature vector(s) at position(s) x_i,y_i (nm_xy)
            # output: c-dimensional local write candidate vector w

            # 2 fc layer
            activ = tf.nn.relu

            input = tf.concat([s_flat, r, c, nm_xy], 1)

            hidden_0 = activ(fc(input, 'lw_fc1', nh=64, init_scale=np.sqrt(2)))
            w = activ(fc(hidden_0, 'lw_fc2', nh=c_dim, init_scale=np.sqrt(2)))

            return w

        def final_nn(r, c, w, no_actions):
            # input: r, c, w
            # output: no_actions-dimensional vector

            # 2 fc layer
            activ = tf.nn.relu
            softmax = tf.nn.softmax

            input = tf.concat([r, c, w], 1)

            hidden_0 = activ(fc(input, 'fnn_fc1', nh=64, init_scale=np.sqrt(2)))
            hidden_1 = activ(fc(hidden_0, 'fnn_fc2', nh=no_actions, init_scale=np.sqrt(2)))
            output = softmax(hidden_1)

            return output

        nm_h_dim = 5
        nm_v_dim = 5
        c_dim = 8

        no_actions = 3

        batch_size = X.shape[0]

        M = tf.placeholder(tf.float32, [batch_size, c_dim])
        S = tf.placeholder(tf.float32, [batch_size, nm_v_dim, nm_h_dim, c_dim])

        s_flat = tf.layers.flatten(X)

        r = global_read(S, c_dim)
        c = context_read(S, s_flat, r, c_dim)
        w = local_write(s_flat, r, c, M, c_dim)
        output = final_nn(r, c, w, no_actions)

        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return output, {'S':S, 'M':M, 'state':w, 'initial_state': initial_state}
    return neural_map_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))