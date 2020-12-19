from abc import ABC

from MooEnv import *

import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
from typing import List


for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
del gpu

max_steps = 10000
elite_threshold = 200
step_reward = 0.
avg_reward = 0.

gamma = 0.99
num_hidden_layer_units = 128
num_hidden_layers = 2

env = env_sch

Global_Seed = 42
np.random.seed(Global_Seed)
tf.random.set_seed(Global_Seed)

eps = np.finfo(np.float32).eps
dim_action_space = 2 * SCH_Test_Setting.x_bin_num
action_shape = (SCH_Test_Setting.x_bin_num, 2)

optimizer = tf.keras.optimizers.Adam()


class NetClass(tf.keras.Model, ABC):
    def __init__(self, unit_output, unit_hidden, layer_hidden):
        super(NetClass, self).__init__()
        self.num_layers = layer_hidden + 1
        self.net_layers = []
        for i in range(layer_hidden):
            self.net_layers.append(
                layers.Dense(unit_hidden, 'relu')
            )
        self.net_layers.append(
            layers.Dense(unit_output)
        )

    def call(self, input, **kwargs):
        x = self.net_layers[0](input)
        for i in range(1, self.num_layers ):
            x = self.net_layers[i](x)
        return x


actor = NetClass(dim_action_space, num_hidden_layer_units, num_hidden_layers)
critic = NetClass(1, num_hidden_layer_units, num_hidden_layers)


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(
        env.step,
        [action],
        [tf.float32, tf.int32]
    )


# @tf.function
def train_iteration(
        initial_state: tf.Tensor,
        gamma: float,
        max_iter_step: tf.int32,
        actor_model: tf.keras.Model,
        critic_model: tf.keras.Model,
        optimizer: tf.keras.optimizers
) -> List[tf.Tensor]:
    state_curr = initial_state
    state_shape = initial_state.shape

    step_reward = 0
    avg_reward = 0.
    # state_curr = tf.expand_dims(state_curr, 0)
    for curr_step in tf.range(max_iter_step):
        with tf.GradientTape(persistent=True) as tape:

            action_net_out = actor_model(state_curr)
            value_curr_out = critic_model(state_curr)

            action_net_out = tf.reshape(action_net_out, action_shape)
            action_prob_softmax = tf.nn.softmax(action_net_out)
            action = tf.random.categorical(
                tf.math.log(action_prob_softmax), 1
            )
            action = tf.squeeze(action)

            state_next, step_reward = tf_env_step(action)
            state_next.set_shape(state_shape)
            state_next = tf.expand_dims(state_next, 0)

            value_next_out = critic_model(state_next)
            avg_reward = 0.99 * avg_reward + \
                0.01 * tf.cast(step_reward, tf.float32)

            TD_err = tf.cast(step_reward, dtype=tf.float32) - avg_reward + \
                gamma * value_next_out - value_curr_out
            loss_actor = - TD_err * tf.math.reduce_sum(
                [tf.math.log(action_prob_softmax[i, j])
                 for i, j in enumerate(action)]
            )
            loss_critic = tf.square(TD_err)

        grads_actor = tape.gradient(loss_actor, actor_model.trainable_variables)
        grads_critic = tape.gradient(
            loss_critic, critic_model.trainable_variables
        )
        optimizer.apply_gradients(
            zip(grads_actor, actor_model.trainable_variables)
        )
        optimizer.apply_gradients(
            zip(grads_critic, critic_model.trainable_variables)
        )

        state_curr = state_next

        if curr_step % 100 == 1:
            print(
                f'Step: {curr_step}, average rewards = {avg_reward}'
                f' elite num = {len(env.elite_list)}\n'
            )

    return [avg_reward, len(env.elite_list)]


env.reset()
initial_state = env.state
train_iteration(
    initial_state,
    gamma,
    max_steps,
    actor,
    critic,
    optimizer
)
