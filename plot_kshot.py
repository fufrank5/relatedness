import matplotlib.pyplot as plt

def plot_kshot():
  plt.clf()
  plt.xlabel('N')
  plt.ylabel('F1')
  plt.xlim([10, 50])
  plt.ylim([30.0, 60.0])
  x = [10, 20, 30, 40, 50]
  plt.xticks(x)
  multi_task = [37.99, 43.67, 47.54, 48.29, 49.82]
  relatedness = [42.7, 47.88, 49.26, 50.15, 51.66]
  plt.plot(x, multi_task, color='blue', linestyle='dashed', marker='o', lw=2, label='multi-task')
  plt.plot(x, relatedness, color='red', linestyle='dashed', marker='X', lw=2, label='relatedness')
  plt.legend(loc='lower right')
  plt.savefig('kshot.svg', format='svg')

plot_kshot()
