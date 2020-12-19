import numpy as np


class MooSCH:
    """
    the description of MOO SCH
    decision space dim, objective space dim, decision spacial range,
    objective function definition
    """
    def __init__(self, x_range=(-30., 30.), parameters=(0., 2.)):
        self.x_low, self.x_up = x_range
        self.a, self.b = parameters
        self.x_dim = 1
        self.obj_dim = 2

    def obj(self, u):
        x = u[0] * (self.x_up - self.x_low) + self.x_low
        f1 = (x - self.a) ** 2
        f2 = (x - self.b) ** 2
        return np.array([f1, f2], dtype=np.float32)


class MooTestSettingA:
    """
    test setting for moo RL
    state encode, decode
    reward function definition
    moo problem initialize
    """
    def __init__(self, moo_pro, x_bin_num=16):
        self.moo_pro = moo_pro
        self.x_bin_num = x_bin_num
        self.bit_dec_list = list(reversed([2 ** i for i in range(x_bin_num)]))

    def _state_to_x(self, state):
        x = []
        n = self.moo_pro.x_dim
        for x_i in range(n):
            temp = 0.
            for bit_i in range(self.x_bin_num):
                temp += state[x_i, bit_i] * self.bit_dec_list[bit_i]
            temp = temp / (2 ** self.x_bin_num - 1)
            x.append(temp)
        return np.array(x, dtype=np.float32)

    def obj_value(self, state):
        x = self._state_to_x(state)
        y = self.moo_pro.obj(x)
        return y

    def reward_value(self, elite_list, obj_value):
        m = self.moo_pro.obj_dim
        if_dominated = False
        if_dominate_others = False
        if_exist = False
        num_dominated = 0
        reward = 1

        for i, cur_elite in enumerate(elite_list):
            if (cur_elite == obj_value).all():
                reward = 0
                if_exist = True
                break
            temp = cur_elite - obj_value
            temp = np.sum(np.sign(temp), dtype=np.int8)
            if temp == m:
                if_dominate_others = True
                elite_list[i] = obj_value
                num_dominated += 1
                reward = 1
            elif temp == -m:
                reward = 0
                if_dominated = True
                break

        if if_exist:
            # print('obj exist, reward ', reward)
            return reward
        if if_dominate_others:
            for i in range(num_dominated - 1):
                elite_list.remove(obj_value)
            # print('obj dominate, reward ', reward)
            return reward
        elif if_dominated:
            # print('obj dominated, reward ', reward)
            return reward
        else:
            elite_list.append(obj_value)
            # print('obj others, reward ', reward)
            return reward

    @staticmethod
    def state_update(state, action):
        return np.bitwise_xor(state, action)

    def initial_state(self):
        return np.random.randint(0, 2, (self.moo_pro.x_dim, self.x_bin_num))


class MooEnv:
    """
    this class is used to describe a multi-objective optimization problem
    for reinforcement learning algorithm, simulate a game environment,
    state is defined as the decision variable, action is ..., reward is ...
    and ...
    """
    env_sch = None

    def __init__(self, moo_test, *args):
        self.obj_func = moo_test.obj_value
        self.red_func = moo_test.reward_value
        self.sta_func = moo_test.state_update
        self.initial_state_func = moo_test.initial_state
        self.args = args
        self.elite_list = []
        self.state = None
        self.object = None

    def reset(self):
        initial_state = self.initial_state_func()
        self.state = initial_state
        obj_fun_values = self.obj_func(initial_state)
        self.object = obj_fun_values
        self.elite_list = [obj_fun_values]
        return self.elite_list

    def _cal_reward(self):
        reward = self.red_func(self.elite_list, self.object)
        return reward

    def _next_state(self, action):
        return self.sta_func(self.state, action)

    def step(self, action):
        state_next = self._next_state(action)
        self.object = self.obj_func(state_next)
        reward = self._cal_reward()
        self.state = state_next
        return self.state, reward


SCH_Pro_Setting = MooSCH()
SCH_Test_Setting = MooTestSettingA(SCH_Pro_Setting)
env_sch = MooEnv(SCH_Test_Setting)
