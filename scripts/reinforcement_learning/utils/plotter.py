import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# call before loop
def initCanvas(subPlotRow_=1, subPlotCol_=1, len=400, plot_interval_=5):
    if subPlotCol_ < 1 or subPlotRow_ < 1:
        raise ValueError('subPlotCol_ and subPlotRow_ should at least be positive')
    global ax, plot_interval, window_len, subPlotRow, subPlotCol
    subPlotRow = subPlotRow_
    subPlotCol = subPlotCol_
    plot_interval = plot_interval_
    window_len = len
    fig, ax = plt.subplots(subPlotRow, subPlotCol)


class Plotter:
    def __init__(self, plotNo_=0, title_='No title', ylabel='Value'):
        global subPlotRow, subPlotCol
        if plotNo_ >= subPlotRow*subPlotCol:
            raise ValueError('plotNo should be less than subPlotRow*subPlotCol')
        self.plotNo = plotNo_
        self.title = title_
        self.yAxisDeque1 = deque(maxlen=window_len)
        self.yAxisDeque2 = deque(maxlen=window_len)
        self.dequeArrays = []
        self.timestamp = 0
        self.lines = []  # 新增：存储曲线对象
        self.ylabel = ylabel  # 新增Y轴标签

    # call in loop
    def plotLine(self, *args, labels=None, colors=None):
        if labels is not None:
            if len(args) != len(labels):
                raise ValueError('args and label must have the same length')
        global subPlotRow, subPlotCol
        self.timestamp += 1

        for _ in range(len(args)):
            self.dequeArrays.append(deque(maxlen=window_len))

        plt.ion()
        for i, arg in enumerate(args):
            self.dequeArrays[i].append(arg)

        psoRow = int(self.plotNo / subPlotCol)
        posCol = self.plotNo % subPlotCol
        if self.timestamp % plot_interval == 0 and self.timestamp > window_len:
            x = torch.linspace(0, len(self.dequeArrays[0]), len(self.dequeArrays[0]))
            if subPlotRow == 1 or subPlotCol == 1:
                ax[self.plotNo].cla()
                if labels is None:
                    for i in range(len(args)):
                        yAxis = np.array(self.dequeArrays[i])
                        ax[self.plotNo].plot(x, yAxis)
                else:
                    for i in range(len(args)):
                        yAxis = np.array(self.dequeArrays[i])
                        ax[self.plotNo].plot(x, yAxis, label=labels[i])
                        ax[self.plotNo].legend()
                ax[self.plotNo].set_title(self.title)
            else:
                ax[psoRow, posCol].cla()
                if labels is None:
                    for i in range(len(args)):
                        yAxis = np.array(self.dequeArrays[i])
                        ax[psoRow, posCol].plot(x, yAxis)
                else:
                    for i in range(len(args)):
                        yAxis = np.array(self.dequeArrays[i])
                        ax[psoRow, posCol].plot(x, yAxis, label=labels[i])
                        ax[psoRow, posCol].legend()
                ax[psoRow, posCol].set_title(self.title)
            # if self.plotNo == 0:
            #     print("Paused ",self.plotNo)
            #     plt.pause(0.001)
