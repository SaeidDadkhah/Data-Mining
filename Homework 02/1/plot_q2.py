import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv', header=None, names=['1', '2', '3', '4', '5', '6', '7'])


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

sns.regplot(x='1', y='7', data=train, ax=ax1)
sns.regplot(x='2', y='7', data=train, ax=ax2)
sns.regplot(x='3', y='7', data=train, ax=ax3)
sns.regplot(x='4', y='7', data=train, ax=ax4)
sns.regplot(x='5', y='7', data=train, ax=ax5)
sns.regplot(x='6', y='7', data=train, ax=ax6)

ax1.set_xlabel('First Grade', fontweight='bold')
ax1.set_ylabel('Seventh Grade', fontweight='bold')
ax2.set_xlabel('Second Grade', fontweight='bold')
ax2.set_ylabel('')
ax3.set_xlabel('Third Grade', fontweight='bold')
ax3.set_ylabel('')

ax4.set_xlabel('Forth Grade', fontweight='bold')
ax4.set_ylabel('Seventh Grade', fontweight='bold')
ax5.set_xlabel('Fifth Grade', fontweight='bold')
ax5.set_ylabel('')
ax6.set_xlabel('Sixth Grade', fontweight='bold')
ax6.set_ylabel('')

fig.suptitle('Seventh grade vs other grades (with missing values)', fontsize=18, fontweight='bold')
plt.show(fig)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

sns.regplot(x='1', y='7', data=train.loc[train['1'].nonzero()], ax=ax1)
sns.regplot(x='2', y='7', data=train.loc[train['2'].nonzero()], ax=ax2)
sns.regplot(x='3', y='7', data=train.loc[train['3'].nonzero()], ax=ax3)
sns.regplot(x='4', y='7', data=train.loc[train['4'].nonzero()], ax=ax4)
sns.regplot(x='5', y='7', data=train.loc[train['5'].nonzero()], ax=ax5)
sns.regplot(x='6', y='7', data=train.loc[train['6'].nonzero()], ax=ax6)

ax1.set_xlabel('First Grade', fontweight='bold')
ax1.set_ylabel('Seventh Grade', fontweight='bold')
ax2.set_xlabel('Second Grade', fontweight='bold')
ax2.set_ylabel('')
ax3.set_xlabel('Third Grade', fontweight='bold')
ax3.set_ylabel('')

ax4.set_xlabel('Forth Grade', fontweight='bold')
ax4.set_ylabel('Seventh Grade', fontweight='bold')
ax5.set_xlabel('Fifth Grade', fontweight='bold')
ax5.set_ylabel('')
ax6.set_xlabel('Sixth Grade', fontweight='bold')
ax6.set_ylabel('')

fig.suptitle('Seventh grade vs other grades (without missing values)', fontsize=18, fontweight='bold')
plt.show(fig)
