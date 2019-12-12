#!/usr/bin/env python
from __future__ import print_function

import argparse
import time

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys

sys.path.append("game/")

import snake as game
import random
import numpy as np
from collections import deque

import json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
import tensorflow as tf
import matplotlib.pyplot as plt

path = "model_DQN5.h5"
lenth = 100


def procImg(img):
    proc_img = skimage.color.rgb2gray(img)
    proc_img = skimage.transform.resize(proc_img, (80, 80))
    proc_img = skimage.exposure.rescale_intensity(proc_img, out_range=(0, 255))
    proc_img = proc_img / 255.0
    return proc_img


class DQN():
    def __init__(self):
        self.Dead = deque()
        self.D = deque()
        self.Dfood = deque()
        self.model = Sequential()
        self.startTime = time.time()
        self.GAME = 'snake'  # the name of the game being played for log files
        self.CONFIG = 'nothreshold'
        self.ACTIONS = 3  # number of valid actions
        self.GAMMA = 0.99  # decay rate of past observations
        self.OBSERVATION = 3200.  # timesteps to observe before training
        self.EXPLORE = 3000000.  # frames over which to anneal epsilon
        self.FINAL_EPSILON = 0.0001  # final value of epsilon
        self.INITIAL_EPSILON = 0.1  # starting value of epsilon
        self.REPLAY_MEMORY = 30000  # number of previous transitions to remember
        self.BATCH = 128  # size of minibatch
        self.FRAME_PER_ACTION = 1
        self.LEARNING_RATE = 1e-4

        self.img_rows = 80
        self.img_cols = 80
        self.img_channels = 4  # We stack 4 frames

        self.input_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.topScore = 0
        self.scorelist = []
        self.timelist = []


    def buildDQN(self):
        print("Now we build the model")
        self.model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',
                                     input_shape=self.input_shape))  # 80*80*4
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.ACTIONS))

        adam = Adam(lr=self.LEARNING_RATE)
        self.model.compile(loss='mse', optimizer=adam)
        print("We finish building the model")
        return self.model

    def trainNetwork(self, model, args):
        # open up a game state to communicate with emulator
        game_state = game.GameState()

        # store the previous observations in replay memory

        # get the first state by doing nothing and preprocess the image to 80x80x4
        # 输入为a_t（(1, 0,0,0)代表上下左右）
        do_nothing = np.zeros(self.ACTIONS)
        do_nothing[0] = 1
        x_t, r_0, terminal, score = game_state.frame_step(do_nothing)

        x_t = procImg(x_t)

        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        # print (s_t.shape)

        # In Keras, need to reshape
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4

        if args['mode'] == 'Run':
            OBSERVE = 999999999  # We keep observe, never train
            self.epsilon = self.FINAL_EPSILON
            print("Now we load weight")
            self.model.load_weights(path)
            adam = Adam(lr=self.LEARNING_RATE)
            self.model.compile(loss='mse', optimizer=adam)
            print("Weight load successfully")
        elif args['mode'] == 'Train':  # We go to training mode

            OBSERVE = self.OBSERVATION
            self.epsilon = self.INITIAL_EPSILON
        elif args['mode'] == 'reTrain':  # contiune train with  self.epsilon = 0.1
            print("Now we load weight")
            self.model.load_weights(path)
            adam = Adam(lr=self.LEARNING_RATE)
            self.model.compile(loss='mse', optimizer=adam)
            print("Weight load successfully")
            OBSERVE = self.OBSERVATION
            self.epsilon = 0.1

        else:  # contiune train
            print("Now we load weight")
            self.model.load_weights(path)
            adam = Adam(lr=self.LEARNING_RATE)
            self.model.compile(loss='mse', optimizer=adam)
            print("Weight load successfully")
            OBSERVE = self.OBSERVATION
            self.epsilon = np.load("e.npy")
            self.epsilon = float(self.epsilon)

        t = 0
        step = 0
        score_pre = 0
        while (True):
            step += 1
            # print(step)
            loss = 0
            Q_sa = 0
            action_index = 0
            r_t = 0
            # 输入为a_t（(1, 0)代表不跳，(0,1)代表跳）。
            a_t = np.zeros([self.ACTIONS])
            # choose an action epsilon greedy
            # 每一帧都处理，做一步动作，并储存记忆
            if t % self.FRAME_PER_ACTION == 0:
                if random.random() <= self.epsilon:
                    # 小于epsilon  随机动作
                    print("----------Random Action----------")
                    action_index = random.randrange(self.ACTIONS)
                    a_t[action_index] = 1
                else:
                    # 大于等于epsilon  通过预测的Q_table，选取最优动作
                    q = model.predict(s_t)  # input a stack of 4 images, get the prediction
                    max_Q = np.argmax(q)
                    action_index = max_Q
                    a_t[max_Q] = 1

            # We reduced the epsilon gradually
            # 随着训练次数的增加，逐渐减小epsilon，即减少随机动作（探索），更倾向于预测值
            if self.epsilon > self.FINAL_EPSILON and t > OBSERVE:
                self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE

            # run the selected action and observed next state and reward
            x_t1_colored, r_t, terminal, score = game_state.frame_step(a_t)

            if terminal:
                self.plot_M(score_pre, step, t),
                step = 0

            if score > self.topScore:
                self.topScore = score
            score_pre = score
            x_t1 = procImg(x_t1_colored)

            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

            # store the transition in D
            self.D.append((s_t, action_index, r_t, s_t1, terminal))
            if r_t == 1:
                self.Dfood.append((s_t, action_index, r_t, s_t1, terminal))
            if r_t == -1:
                self.Dead.append((s_t, action_index, r_t, s_t1, terminal))

            # 控制D（记忆的大小），忘记以前的事情
            if len(self.D) > self.REPLAY_MEMORY:
                self.D.popleft()
            if len(self.Dfood) > self.REPLAY_MEMORY//10:
                self.Dfood.popleft()
            if len(self.Dead) > self.REPLAY_MEMORY//10:
                self.Dead.popleft()

            # only train if done observing
            # 从记忆库中选取片段训练，即经验回放
            if t > OBSERVE:
                loss, Q_sa = self.learn()
            # 迭代，准备下一步
            s_t = s_t1
            t = t + 1

            # save progress every 10000 iterations
            # 每1000次，保存模型
            if t % 1000 == 0:
                self.saveModel()
            if t % 10 == 0:
                self.printIofo(t, OBSERVE, action_index, r_t, Q_sa, loss, score)
        print("Episode finished!")
        print("************************")

    def learn(self):
        # sample a minibatch to train on
        minibatch = random.sample(self.D, int(self.BATCH / 2))
        try:
            minibatch += random.sample(self.Dead, int(self.BATCH / 4))
        except:
            minibatch += random.sample(self.Dead * 100, int(self.BATCH / 4))
        try:
            minibatch += random.sample(self.Dfood, int(self.BATCH / 4))
        except:
            minibatch += random.sample(self.Dfood * 100, int(self.BATCH / 4))

        # Now we do the experience replay
        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)
        targets = self.model.predict(state_t)
        Q_sa = self.model.predict(state_t1)
        targets[range(self.BATCH), action_t] = reward_t + self.GAMMA * np.max(Q_sa, axis=1) * np.invert(
            terminal)

        loss = self.model.train_on_batch(state_t, targets)
        return loss, Q_sa

    def printIofo(self, t, OBSERVE, action_index, r_t, Q_sa, loss, score):
        # print info
        # 打印状态
        state = ""
        if t <= OBSERVE:
            state = "observe"
        # 探索期
        elif t > OBSERVE and t <= OBSERVE + self.EXPLORE:
            state = "explore"
        else:
            state = "train"
        timeCost = int(time.time() - self.startTime)
        print("TIMESTEP", t, "/TIMECOST", timeCost, "/ STATE", state, \
              "/ EPSILON", self.epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
              "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss, "/ \nTop score", self.topScore, "\nScore", score, \
              "\ncount", len(self.scorelist)
              )

    def saveModel(self):
        print("Now we save model")
        self.model.save_weights(path, overwrite=True)
        # with open("model_DQN4.json", "w") as outfile:
        #     json.dump(self.model.to_json(), outfile)

        np.save("e.npy", np.array(self.epsilon))

    def saveHistory(self, timecost):
        np.save("history/score%d.npy" % timecost, np.array(self.scorelist))
        np.save("history/step%d.npy" % timecost, np.array(self.timelist))

    def playGame(self, args):
        # 创建DQN模型
        model = self.buildDQN()
        # 传入模型，训练
        self.trainNetwork(model, args)

    def plot_M(self, score, time, t):

        self.scorelist.append(score)
        self.timelist.append(time)
        if len(self.scorelist) == lenth:
            print(self.timelist)
            self.plot(self.scorelist, self.timelist, t)

    def plot(self, scorelist, timelist, timecost):
        # plt.plot([[x for x in range(10)],[x for x in range(10)],[x for x in range(10)]],[np.array(scorelist),np.array(timelist),np.array(scorelist)/np.array(timelist)])
        # print(timelist)
        plt.figure(1)
        plt.subplot(211)
        plt.plot([x for x in range(lenth)][::1], timelist[::1])
        plt.title(np.average(timelist))
        plt.subplot(212)
        plt.plot([x for x in range(lenth)][::1], scorelist[::1])
        plt.title(np.average(scorelist))
        plt.savefig('pic/step_score%d.png' % timecost)
        plt.show()
        # 保存分数和步数
        self.saveHistory(timecost)

        self.scorelist = []
        self.timelist = []


def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    # 实例化，训练
    dqn = DQN()
    dqn.playGame(args)


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K

    K.set_session(sess)

    # K.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu': 0})))

    main()
