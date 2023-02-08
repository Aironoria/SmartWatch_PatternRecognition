import itertools
import matplotlib.pyplot as plt
import numpy as np
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*100,fmt)  + "%",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout(pad=2)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


matrix = np.array([
  [82.4,11.8,2.0,2,2,0],
    [3.6,84.8,0.9,2.7,3.6,4.5],
    [0,1.7,96.6,0,0,1.7],
    [1.2,7.4,3.7,87.7,0,0],
    [0,5.1,0,2,92.9,0],
    [0, 1.3,3.9,0,0,94.7]

])

label =  ["Burger", "Edamame","Egg Fried Rice", "Noodle","Peanuts","Unknown"]


for i in range(matrix.shape[0]):
    # if matrix[i].sum()!= 100.0:
    print(f"line {i} sum to {matrix[i].sum()}")
plot_confusion_matrix(matrix, classes=label, normalize=True, title='Normalized confusion matrix')
