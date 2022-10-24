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
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


matrix = np.array([
    [0.918, 0, 0.01925,0.02275, 0.02875,0.01125,0],
    [0,0.9545, 0.0455,0,0,0,0],
    [0,0,0.952,0.00725,0.04075,0,0],
    [0.0105,0.031,0.01125,0.91225,0.0245,0.0105,0],
    [0,0.027, 0.05425,0.0125,0.893,0,0.014],
    [0.01,0,0.043,0, 0,0.94725,0],
    [0.01866,0.015,0,0.015,0,0,0.961]
])

label =  [ 'Apple', "Burger",'Edamame',"Noodle","Nugget", "Peanut", "Rice"]

plot_confusion_matrix(matrix, classes=label, normalize=False, title='Normalized confusion matrix')
