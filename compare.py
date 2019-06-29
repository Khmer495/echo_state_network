# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick

def fig(u, x, Wout, b, output_folder, kind):
    u = u.reshape(-1)
    y = np.dot(Wout, x).reshape(-1) + b
    t = list(range(len(y)))

    label_size = 14
    #title_size = 16
    scale_size = 12
    legend_size = 12
    suptitle_size = 16
    ax_offset_size = 10
    #list_name/epoch(1~100)/[time,epoch,iteration,main_loss,test_loss]
    fig = plt.figure()
    #plt.subplots_adjust(wspace=0.4, hspace=0.6)
    ax = fig.add_subplot(1,1,1)
    ax.plot(t, u, color='r', marker='', linewidth=0.5, label='original')
    ax.plot(t, y, color='b', marker='', linewidth=0.5, label='inference')

    #軸目盛文字
    plt.xticks(fontsize=scale_size)
    plt.yticks(fontsize=scale_size)

    #軸範囲
    #ax1.set_ylim(-1e-5,max(id_max)*1.1)

    # x軸に補助目盛線を設定
    ax.grid(which = "major", axis = "x", color = "black", alpha = 0.1, linestyle = ":", linewidth = 1)
    # y軸に補助目盛線を設定
    ax.grid(which = "major", axis = "y", color = "black", alpha = 0.1, linestyle = ":", linewidth = 1)

    #label
    ax.set_xlabel("time", fontsize=label_size)
    ax.set_ylabel("value", fontsize=label_size)

    plt.legend(loc='upper left', fontsize=legend_size)

    #title
    #plt.title("eigenvalues", fontsize=title_size)

    #指数表記
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.yaxis.offsetText.set_fontsize(ax_offset_size)
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))

    #タイトル
    #plt.tight_layout()
    #fig.suptitle("eigenvalues", fontsize=suptitle_size)

    a=0.02
    plt.subplots_adjust(top=0.9+a,left=0.15-a,right=0.95+a,bottom=0.15-a)

    plt.savefig("{}/compare_{}.svg".format(output_folder,kind))
    #plt.show()
