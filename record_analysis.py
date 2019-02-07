import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RECORD = "records/ddqn-record2.txt"

df = pd.read_csv(RECORD, names=['model', 'result', 'time', 'tot_reward',
    'steps', 'avg_q_max', 'avg_loss'])
df['time'] = pd.to_datetime(df['time'],unit='s')
df['result'] = df['result'].str.strip()
print(df.head())

# df.plot(y='avg_q_max', use_index=True)
dfwin = df[df['result'] == 'Result.Victory']
dfloss = df[df['result'] == 'Result.Defeat']



xaxis = ['index']
yaxis = ['tot_reward', 'steps', 'avg_q_max', 'avg_loss']
for x in xaxis:
    for y in yaxis:
        fig, ax = plt.subplots(nrows=3, ncols=1)
        fig.suptitle(y)
        ax[0].set_ylabel("tot")
        ax[1].set_ylabel("win")
        ax[2].set_ylabel("loss")
        df.reset_index().plot.scatter(x=x, y=y, ax=ax[0])
        dfwin.reset_index().plot.scatter(x=x, y=y, ax=ax[1], c='g')
        dfloss.reset_index().plot.scatter(x=x, y=y, ax=ax[2], c='r')
        plt.show()

xaxis = ['time']
yaxis = ['tot_reward', 'steps', 'avg_q_max', 'avg_loss']
for x in xaxis:
    for y in yaxis:
        fig, ax = plt.subplots(nrows=3, ncols=1)
        fig.suptitle(y)
        ax[0].set_ylabel("tot")
        ax[1].set_ylabel("win")
        ax[2].set_ylabel("loss")
        df.reset_index().plot(x=x, y=y, ax=ax[0])
        dfwin.reset_index().plot(x=x, y=y, ax=ax[1], c='g')
        dfloss.reset_index().plot(x=x, y=y, ax=ax[2], c='r')
        plt.show()

yaxis = ['tot_reward', 'steps', 'avg_q_max', 'avg_loss']
for x in yaxis:
    for y in yaxis:
        if x != y:
            fig, ax = plt.subplots(nrows=3, ncols=1)
            fig.suptitle(y)
            ax[0].set_ylabel("tot")
            ax[1].set_ylabel("win")
            ax[2].set_ylabel("loss")
            df.reset_index().plot.scatter(x=x, y=y, ax=ax[0])
            dfwin.reset_index().plot.scatter(x=x, y=y, ax=ax[1], c='g')
            dfloss.reset_index().plot.scatter(x=x, y=y, ax=ax[2], c='r')
            plt.show()
