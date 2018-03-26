import numpy as np
import tensorflow as tf  # pylint: ignore-module
import builtins
import functools
import copy
import os
import collections


# ================================================================
# Make consistent with numpy
# ================================================================

clip = tf.clip_by_value


def sum(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_sum(x, axis=axis, keep_dims=keepdims)


def mean(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_mean(x, axis=axis, keep_dims=keepdims)


def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdims=keepdims)
    return mean(tf.square(x - meanx), axis=axis, keepdims=keepdims)


def std(x, axis=None, keepdims=False):
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))


def max(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_max(x, axis=axis, keep_dims=keepdims)


def min(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_min(x, axis=axis, keep_dims=keepdims)


def concatenate(arrs, axis=0):
    return tf.concat(axis=axis, values=arrs)


def argmax(x, axis=None):
    return tf.argmax(x, axis=axis)


def switch(condition, then_expression, else_expression):
    """Switches between two operations depending on a scalar value (int or bool).
    Note that both `then_expression` and `else_expression` should be symbolic
    tensors of the *same shape*.

    # Arguments
        condition: scalar tensor,
        then_expression: TensorFlow operation,
        else_expression: TensorFlow operation.
    """
    x_shape = copy.copy(then_expression.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: then_expression,
                lambda: else_expression)
    x.set_shape(x_shape)

    return x


# ================================================================
# Extras
# ================================================================


def l2reg(params):  # weight decay
    if len(params) == 0:
        return tf.constant(0.0)
    else:
        # add_n: adds all input tensors (inputs) element-wise
        # Returns a tensor of same shape and type as the elements of inputs
        return tf.add_n([sum(tf.square(p)) for p in params])


def leaky_relu(x, leak=0.2):
    # Best way to implement Leaky ReLU
    # Relies on the identity: max(0, x) = (x + abs(x))/2
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)

    return f1 * x + f2 * abs(x)


def categorical_samples(X):
    # https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/
    # Relies on the Gumbel Max trick to get a sample out of a categorical distribution
    # without needing to construct such distribution and sample from it
    # Since we just want a sample (policies are not involved, therefore goodness is not relevant)
    # z Gumbel noise, the sample s verifies: s = argmax(z)
    # From Gumbel distribution CDF, we get z = x - log(-log(P(Z >= z)))
    # Since we just want a categorical sample, we can take P(Z >= z) as samples from a uniform U
    U = tf.random_uniform(tf.shape(X))
    samples = argmax(X - tf.log(-tf.log(U)), axis=1)

    return samples


# ================================================================
# Inputs
# ================================================================


def is_placeholder(x):
    return type(x) is tf.Tensor and len(x.op.inputs) == 0


class TfInput(object):
    """Is there any reason for a class declaration to inherit from `object`?
    Yes, this is a 'new style' object. It was a feature introduced in python 2.2.
    New style objects have a different object model to classic objects,
    and some things won't work properly with old style objects, for instance,
    super(), @property and descriptors.
    """
    def __init__(self, name="(unnamed)"):
        """Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """Returns the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplemented()

    def make_feed_dict(self, data):
        """Given data, input (feed) it to the placeholder(s)."""
        raise NotImplemented()


class PlaceholderTfInput(TfInput):
    def __init__(self, placeholder):
        """Wrapper for regular tensorflow placeholer(s)."""
        super().__init__(placeholder.name)
        self._placeholder = placeholder  # '_' means private attribute

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: data}


class BatchInput(PlaceholderTfInput):
    def __init__(self, shape, dtype=tf.float32, name=None):
        """Creates a placeholder for a batch of tensors of a given shape and dtype

        # Parameters:
            - shape: [int]
                shape of a single element of the batch
            - dtype: tf.dtype
                number representation used for tensor contents
            - name: str
                name of the underlying placeholder
        """
        super().__init__(tf.placeholder(dtype, [None] + list(shape), name=name))
        self._shape = shape


class Uint8Input(PlaceholderTfInput):
    def __init__(self, shape, name=None):
        """Takes input in uint8 format (1-byte (8-bit) unsigned integers:
        integer between 0 and 2^8 - 1 -> 255) which is cast to float32 and divided
        by 255 before passing it to the model.
        (By working with integers, a division by 255 will yield 0 for 0-254, that is
        why we first cast the uint8 to float32, then divide. The goal is to get a float
        between 0 and 1, since on GPUs it ensures lower data transfer times.)


        # Parameters:
            - shape: [int]
                shape of the tensor
            - name: str
                name of the underlying placeholder
        """
        super().__init__(tf.placehodler(tf.uint8, [None] + list(shape), name=name))
        self._shape = shape
        # Retrieve the placeholder via super().get() defined in parent
        self._unit_placeholder = tf.cast(super().get(), tf.float32) / 255.0

    def get(self):
        return self._unit_placeholder


def ensure_tf_input(candidate):
    """Takes either tf.placeholder of TfInput and outputs equivalent TfInput"""
    if isinstance(candidate, TfInput):
        return candidate
    elif is_placeholder(candidate):
        return PlaceholderTfInput(candidate)
    else:
        raise ValueError("Must be a placeholder or TfInput")


# ================================================================
# Mathematical utils
# ================================================================


def huber_loss(x, delta=1.0):
    """Less sensistive to outliers than the l2 loss
    Reference: https://en.wikipedia.org/wiki/Huber_loss
    """
    return tf.where(
        tf.abs(x) < delta,
        0.5 * tf.square(x),
        delta * (tf.abs(x) - 0.5 * delta)
    )


# ================================================================
# Optimizer utils
# ================================================================


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimizes `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensuring the norm of the gradients for each variable
    is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    # `gradients` is a list of (gradient, variable) pairs.
    # Variable is always present, but gradient can be None.
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            # Edit the gradient for the i-th variable
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)

    return optimizer.apply_gradients(gradients)


# ================================================================
# Global session
# ================================================================


def get_session():
    """Returns the default session being used in the current thread"""
    return tf.get_default_session()


def make_session(num_cpu):
    """Returns a session which will use `num_cpu` CPUs only"""
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    return tf.Session(config=tf_config)


def single_threaded_session():
    """Returns a session which will only use a single CPU"""
    return make_session(1)


ALREADY_INITIALIZED = set()  # python set: unordered collection unique elements


def initialize():
    """Initialize all the uninitialized variables in the global scope"""
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


def eval(expr, feed_dict=None):
    if feed_dict is None:
        feed_dict = {}
    return get_session().run(expr, feed_dict=feed_dict)


VALUE_SETTERS = collections.OrderedDict()


def set_value(v, val):
    global VALUE_SETTERS
    if v in VALUE_SETTERS:
        # if the setter is already registred
        set_op, set_endpoint = VALUE_SETTERS[v]
    else:
        # if not, create it and add it to the collection
        set_endpoint = tf.placeholder(v.dtype)
        set_op = v.assign(set_endpoint)
        VALUE_SETTERS[v] = (set_op, set_endpoint)

    get_session().run(set_op, feed_dict={set_endpoint: val})


# ================================================================
# Loading/Saving variables
# ================================================================


def load_state(fname):
    # Restore variables from disk
    saver = tf.train.Saver()
    saver.restore(get_session(), fname)


def save_state(fname):
    # Save the variables to disk
    os.makedirs(os.path.dirname(fname), exist_ok=True)  # no exception if already exists
    saver = tf.train.Saver()
    saver.save(get_session(), fname)


# ================================================================
# Model components
# ================================================================


def normc_initializer(std=1.0):
    """Initializes a parameter from a normal distribution, normalized by column"""

    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        # randn returns a sample (or samples) from the standard normal distribution (N(0,1))
        # it takes as parameters the dimensions of the returned array
        # `*shape` enables us to unpack the list `shape` containing those dimensions
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        # we normalize by column, therefore usable when layers are defines with xW
        # in order to be usable with Wx, initialize by row instead (not provided) -> xW
        return tf.constant(out)

    # returns a function
    # Example: my_init_func = normc_initializer(0.2)
    # my_init_func is a function that takes `shape` (and optinal) as argument
    # it is not used explicitely in tf -> we just create the init function with the desired
    # std and tf itself will use it with the shape defined in the placeholder
    return _initializer


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME",
           dtype=tf.float32, collections=None, summary_tag=None):
    # Vanilla tf.nn.conv2d:
    # shape of input tensor: [batch, in_height, in_width, in_channels]
    # shape of filter tensor: [filter_height, filter_width, in_channels, out_channels]
    # shape of strides tensor: stride of the sliding window for each dimension of input
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num imput feature maps * filter height * filter width"
        # inputs to each hidden unit (defined as the "fan in")
        # "num input feature maps" is `in_channels`
        fan_in = intprod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width / pooling size"
        # "num output feature maps" is `out_channels`
        fan_out = intprod(filter_shape[:2]) * num_filters

        # glorot initialization:
        # w ~ gaussian distribution -> variance(w) = 2.0 / (fan_in + fan_out)
        # w ~ uniform distribution -> weights w in [-w_bound, w_bound] with
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype,
                            tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters],
                            initializer=tf.zeros_initializer(),
                            collections=collections)

        if summary_tag is not None:
            tf.summary.image(summary_tag,
                             tf.transpose(tf.reshape(w, [filter_size[0], filter_size[1], -1, 1]),
                                          [2, 0, 1, 3]),
                             max_images=10)

        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def dense(x, size, name, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [int(x.get_shape()[1]), size], initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
        return ret + b
    else:
        return ret


def wndense(x, size, name, init_scale=1.0):
    # "wn" stands for "weight normalization"
    v = tf.get_variable(name + "/V", [int(x.get_shape()[1]), size],
                        initializer=tf.random_normal_initializer(0, 0.05))
    g = tf.get_variable(name + "/g", [size], initializer=tf.constant_initializer(init_scale))
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(0.0))

    # use weight normalization
    # Salimans & Kingma, 2016: https://arxiv.org/pdf/1602.07868.pdf
    x = tf.matmul(x, v)
    scaler = g / tf.sqrt(sum(tf.square(v), axis=0, keepdims=True))  # normalize columns

    # element-wise product of the scaler and x (both row vectors) + bias in row vector shape
    return tf.reshape(scaler, [1, size]) * x + tf.reshape(b)


def densenobias(x, size, name, weight_init=None):
    return dense(x, size, name, weight_init=weight_init, bias=False)


def dropout(x, pkeep, phase=None, mask=None):
    # `pkeep` is a scalar tensor with the same type as `x` (broadcastable to the shape of `x`)
    # and contains the probability that each element is kept
    mask = tf.floor(pkeep + tf.random_uniform(tf.shape(x))) if mask is None else mask
    if phase is None:
        return mask * x  # by default, if no phase is indicated, it is counted as a True
    else:
        return switch(phase, mask * x, pkeep * x)


def batchnorm(x, name, phase, updates, gamma=0.96):
    # For this to even make sense, x has to be a mini-batch (dim 0 of the shape)
    # and the running statistics are taken over the mini-batch
    k = x.get_shape()[1]  # [0] would be the number of vectors in the mini-batch
    running_mean = tf.get_variable(name + "/mean", shape=[1, k],
                                   initializer=tf.constant_initializer(0.0),
                                   trainable=False)
    running_var = tf.get_variable(name + "/var", shape=[1, k],
                                  initializer=tf.constant_initializer(1e-4),
                                  trainable=False)
    test_y = (x - running_mean) / tf.sqrt(running_var)

    # axis=0 means the statistics are computed over the mini-batch
    mean_ = mean(x, axis=0, keepdims=True)
    var_ = mean(tf.square(x), axis=0, keepdims=True)
    std = tf.sqrt(var_)
    train_y = (x - mean_) / std

    updates.extend([
        tf.assign(running_mean, running_mean * gamma + mean_ * (1 - gamma)),
        tf.assign(running_var, running_var * gamma + var_ * (1 - gamma))
    ])

    y = switch(phase, train_y, test_y)

    scaling = tf.get_variable(name + "/scaling", shape=[1, k],
                              initializer=tf.constant_initializer(1.0),
                              trainable=True)
    translation = tf.get_variable(name + "/translation", shape=[1, k],
                                  initializer=tf.constant_initializer(0.0),
                                  trainable=True)

    return y * scaling + translation


# ================================================================
# Theano-like function
# ================================================================


def function(inputs, outputs, updates=None, givens=None):
    """Just like Theano function. Take a bunch of tensorflow placeholders and expressions
    computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
    values to be fed to the input's placeholders and produces the values of the expressions
    in outputs.

    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name (passed to constructor or accessible via placeholder.op.name).

    Example:
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})

        with single_threaded_session():
            initialize()

            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12

    # Parameters:
        - inputs: [tf.placeholder or TfInput]
            list of input arguments
        - outputs: [tf.Variable] or tf.Variable
            list of outputs or a single output to be returned from the function.
            Returned value will also have the same shape.
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs.values(), updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, inputs, outputs, updates, givens, check_nan=False):
        for inpt in inputs:
            if not issubclass(type(inpt), TfInput):
                assert len(inpt.op.inputs) == 0, \
                    "inputs should all be placeholders of baselines.common.TfInput"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens
        self.check_nan = check_nan

    def _feed_input(self, feed_dict, inpt, value):
        if issubclass(type(inpt), TfInput):
            feed_dict.update(inpt.make_feed_dict(value))
        elif is_placeholder(inpt):
            feed_dict[inpt] = value

    def __call__(self, *args, **kwargs):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # Update the kwargs
        kwargs_passed_inpt_names = set()
        for inpt in self.inputs[len(args):]:
            inpt_name = inpt.name.split(':')[0]
            inpt_name = inpt_name.split('/')[-1]
            assert inpt_name not in kwargs_passed_inpt_names, \
                """this function has two arguments with the same name \"{}\",
                so kwargs cannot be used.""".format(inpt_name)
            if inpt_name in kwargs:
                kwargs_passed_inpt_names.add(inpt_name)
                self._feed_input(feed_dict, inpt, kwargs.pop(inpt_name))
            else:
                assert inpt in self.givens, "Missing argument " + inpt_name
        assert len(kwargs) == 0, "Function got extra arguments " + str(list(kwargs.keys()))
        # Update feed dict with givens
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        if self.check_nan:
            if any(np.isnan(r).any() for r in results):
                raise RuntimeError("Nan detected")

        return results


def mem_friendly_function(nondata_inputs, data_inputs, outputs, batch_size):
    if isinstance(outputs, list):
        return _MemFriendlyFunction(nondata_inputs, data_inputs, outputs, batch_size)
    else:
        f = _MemFriendlyFunction(nondata_inputs, data_inputs, [outputs], batch_size)
        return lambda *inputs: f(*inputs)[0]


class _MemFriendlyFunction(object):
    def __init__(self, nondata_inputs, data_inputs, outputs, batch_size):
        self.nondata_inputs = nondata_inputs
        self.data_inputs = data_inputs
        self.outputs = list(outputs)
        self.batch_size = batch_size

    def __call__(self, *inputvals):
        assert len(inputvals) == len(self.nondata_inputs) + len(self.data_inputs)
        nondata_vals = inputvals[0:len(self.nondata_inputs)]
        data_vals = inputvals[len(self.nondata_inputs):]
        feed_dict = dict(zip(self.nondata_inputs, nondata_vals))
        n = data_vals[0].shape[0]
        for v in data_vals[1:]:
            assert v.shape[0] == n
        for i_start in range(0, n, self.batch_size):
            slice_vals = [v[i_start:builtins.min(i_start + self.batch_size, n)] for v in data_vals]
            for (var, val) in zip(self.data_inputs, slice_vals):
                feed_dict[var] = val
            results = tf.get_default_session().run(self.outputs, feed_dict=feed_dict)
            if i_start == 0:
                sum_results = results
            else:
                for i in range(len(results)):
                    sum_results[i] = sum_results[i] + results[i]
        for i in range(len(results)):
            sum_results[i] = sum_results[i] / n

        return sum_results


# ================================================================
# Modules
# ================================================================


class Module(object):
    def __init__(self, name):
        self.name = name
        self.first_time = True
        self.scope = None
        self.cache = {}

    def __call__(self, *args):
        if args in self.cache:
            print("(%s) retrieving value from cache" % (self.name,))
            return self.cache[args]
        with tf.variable_scope(self.name, reuse=not self.first_time):
            scope = tf.get_variable_scope().name
            if self.first_time:
                self.scope = scope
                print("(%s) running function for the first time" % (self.name,))
            else:
                assert self.scope == scope, "Tried calling function with a different scope"
                print("(%s) running function on new inputs" % (self.name,))
            self.first_time = False
            out = self._call(*args)
        self.cache[args] = out
        return out

    def _call(self, *args):
        raise NotImplementedError

    @property
    def trainable_variables(self):
        assert self.scope is not None, "need to call module once before getting variables"
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    @property
    def variables(self):
        assert self.scope is not None, "need to call module once before getting variables"
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)


def module(name):
    @functools.wraps
    def wrapper(f):
        class WrapperModule(Module):
            def _call(self, *args):
                return f(*args)
        return WrapperModule(name)
    return wrapper


# ================================================================
# Graph traversal
# ================================================================


VARIABLES = {}


def get_parents(node):
    return node.op.inputs


def topsorted(outputs):
    """Topological sort via non-recursive depth-first search"""
    assert isinstance(outputs, (list, tuple))
    marks = {}
    out = []
    stack = []  # pylint: disable=W0621
    # i: node
    # jidx = number of children visited so far from that node
    # marks: state of each node, which is one of
    #   0: haven't visited
    #   1: have visited, but not done visiting children
    #   2: done visiting children
    for x in outputs:
        stack.append((x, 0))
        while stack:
            (i, jidx) = stack.pop()
            if jidx == 0:  # 0 children visited so far from that node
                m = marks.get(i, 0)
                if m == 0:
                    marks[i] = 1
                elif m == 1:
                    raise ValueError("not a dag")
                else:
                    continue
            ps = get_parents(i)
            if jidx == len(ps):
                marks[i] = 2
                out.append(i)
            else:
                stack.append((i, jidx + 1))
                j = ps[jidx]
                stack.append((j, 0))

    return out


# ================================================================
# Flat vectors
# ================================================================


def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):  # stands for "number of elements"
    return intprod(var_shape(x))


def intprod(x):
    return int(np.prod(x))


def flatgrad(loss, var_list, clip_norm=None):
    grads = tf.gradients(loss, var_list)
    # returns a list of sum(dy/dx) for each x in xs
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]

    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
        ])


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)  # creates an op that groups multiple operations

    def __call__(self, theta):
        get_session().run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return get_session().run(self.op)


# ================================================================
# Scopes
# ================================================================


def scope_vars(scope, trainable_only=False):
    """Get variables inside a scope, which can be specified as a string

    # Parameters:
        - scope: str or VariableScope
            scope in which the vaiable reside
        - trainable_only: bool
            whether or not to return only the variables that were marked as trainable

    # Returns
        - vars: [tf.Variable]
            list of variables in `scope`
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def scope_name():
    """Returns the name of the current scope as a string, e.g. deepq/q_func"""
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    return scope_name() + "/" + relative_scope_name


def lengths_to_mask(lengths_b, max_length):
    """Turns a vector of lengths into a boolean mask

    # Parameters:
        - lengths_b: an integer vector of lengths
        - max_length: maximum length to fill the mask

    # Returns:
        - a boolean array of shape (batch_size, max_length)
        row[i] consists of True repeated lengths_b[i] times, followed by False
    """
    lengths_b = tf.convert_to_tensor(lengths_b)
    assert lengths_b.get_shape().ndims == 1
    mask_bt = tf.expand_dims(tf.range(max_length), 0) < tf.expand_dims(lengths_b, 1)

    return mask_bt


def in_session(f):
    @functools.wraps(f)
    def newfunc(*args, **kwargs):
        with tf.Session():
            f(*args, **kwargs)
    return newfunc


_PLACEHOLDER_CACHE = {}  # name -> (placeholder, dtype, shape)


def get_placeholder(name, dtype, shape):
    if name in _PLACEHOLDER_CACHE:
        placeholder_, dtype_, shape_ = _PLACEHOLDER_CACHE[name]
        assert dtype_ == dtype and shape_ == shape
        return placeholder_
    else:
        placeholder_ = tf.placeholder(dtype=dtype, shape=shape, name=name)
        _PLACEHOLDER_CACHE[name] = (placeholder_, dtype, shape)
        return placeholder_


def get_placeholder_cached(name):
    return _PLACEHOLDER_CACHE[name][0]


def flatten_all_but_0(x):
    return tf.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])


def reset():
    # Reset variables and placeholders
    global _PLACEHOLDER_CACHE
    global VARIABLES
    _PLACEHOLDER_CACHE = {}
    VARIABLES = {}
    # Reset graph
    tf.reset_default_graph()
