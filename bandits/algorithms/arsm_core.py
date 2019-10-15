import tensorflow as tf
from tensorflow import distributions as tfd
from keras import layers, initializers, models


def swap(x, i, j):
    ans = x.copy()
    ans[i], ans[j] = ans[j], ans[i]
    return ans

@tf.custom_gradient
def reinforce_loss(inputs, action, reward):
    probs = tf.math.sigmoid(inputs)
    indicator = tf.one_hot(action, depth=inputs.shape[1])
    loss = reward * tf.boolean_mask(probs, indicator)
    def grad(dy):
        g = tf.expand_dims(dy * reward, 1) * (indicator - probs)
        return g, 0, 0  # action and reward are considered constant
    return loss, grad

@tf.custom_gradient
def ar_loss(inputs, action, reward):
    pass

@tf.custom_gradient
def ars_loss(inputs, action, reward):
    pass

@tf.custom_gradient
def arsm_loss(inputs, action, reward):
    pass


class PolicyGradientModel(models.Model):
    def __init__(self, num_actions):
        super(PolicyGradientModel, self).__init__()
        self.num_actions = num_actions
        self.sampler = tfd.Dirichlet(tf.ones(num_actions))

    def action(self, inputs):
        omega = self.sampler.sample(inputs.shape[0])
        log_omega = tf.math.log(omega)
        a = tf.math.argmin(log_omega - inputs, axis=-1)
        return a

    def call(self, inputs, action):
        raise NotImplementedError


class Reinforce(PolicyGradientModel):
    def __init__(self, num_actions):
        super(Reinforce, self).__init__(num_actions)

    def call(self, inputs, action, reward):
        return reinforce_loss(inputs, action, reward)


class AR(PolicyGradientModel):
    def __init__(self, num_actions):
        super(AR, self).__init__(num_actions)

    def call(self, inputs):
        pass


class ARS(PolicyGradientModel):
    def __init__(self, num_actions):
        pass

    def call(self, inputs):
        pass


class ARM(PolicyGradientModel):
    def __init__(self, num_actions):
        pass

    def call(self, inputs):
        pass


tf.enable_eager_execution()

reward = tf.constant([3.0, 1.0])
action = tf.constant([1,0])
num_actions = 2
inputs = tf.constant([[3.0, 1.0, -0.5], [0.0, 0.0, 0.1]])
dense_layer = layers.Dense(units=3)
with tf.GradientTape() as tape:
    x = dense_layer(inputs)
    loss = reinforce_loss(x, action, reward)
    dy = tape.gradient(loss, dense_layer.trainable_weights)
0