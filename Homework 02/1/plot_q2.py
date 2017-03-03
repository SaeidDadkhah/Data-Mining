import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv', header=None)

plt.subplot(231)
plt.suptitle('Seventh grade vs other grades', fontsize=14, fontweight='bold')
plt.plot(train[0], train[6], 'o')
plt.xlabel('First Grade')
plt.ylabel('Seventh Grade')

plt.subplot(232)
plt.plot(train[1], train[6], 'o')
plt.xlabel('Second Grade')

plt.subplot(233)
plt.plot(train[2], train[6], 'o')
plt.xlabel('Third Grade')

plt.subplot(234)
plt.plot(train[3], train[6], 'o')
plt.xlabel('Forth Grade')
plt.ylabel('Seventh Grade')

plt.subplot(235)
plt.plot(train[4], train[6], 'o')
plt.xlabel('Fifth Grade')

plt.subplot(236)
plt.plot(train[5], train[6], 'o')
plt.xlabel('Sixth Grade')

plt.show()
