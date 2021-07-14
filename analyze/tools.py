#! -*- coding:utf-8 -*-
import itertools
import numpy as np
from scipy import optimize
import scipy.stats as stats
from matplotlib import pyplot as plt

# 画 3D 图
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# #ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图
# ax1.scatter3D(xd,yd,zd, label='a', c='r', cmap='Blues')  #绘制散点图
# ax1.plot3D(x,y,z,'gray')    #绘制空间曲线
# plt.show()

def pearson( x, y ):
    if len(x)==0:
        return np.nan
    else:
        return stats.pearsonr( np.array(x)/np.max(x), np.array(y)/np.max(y) )[0]

                                
# # Python实现正态分布
# # 绘制正态分布概率密度函数
# u = 55   # 均值μ
# sig = math.sqrt(5)  # 标准差δ
# x = np.linspace(u - 3*sig, u + 3*sig, 50)
# y_sig = np.exp(-(x - u) ** 2 /(2* sig **2)) / (math.sqrt(2*math.pi)*sig)
# plt.plot(x, y_sig, "r-", linewidth=5)
# # plt.plot(x, y, 'r-', x, y, 'go', linewidth=2,markersize=8)
# plt.grid(False)
# plt.savefig('a_55.png')
# plt.close()
# plt.clf()


# fig,ax=plt.subplots() # 生成一张图和一张子图
# ax.plot(xs,ys,'k-') # x横坐标 y纵坐标 ‘k-’线性为黑色
# ax.grid()# 添加网格线
# ax.axis('equal')
# plt.show()


# def plot_history(history, save_dir):
#     # plot acc, loss, val_acc, val_loss curves
#     for key, value in history.history.items():
#         plot_curves({'epoch:%d'%len(value):value}, '%s_curve'%key,
#                     'epoch', key, os.path.join(save_dir,'%s.png'%key), True)
#         savelist(list_data=value, savepath=os.path.join(save_dir,'%s.txt'%key), mode='w')


def fitting_curve(dict_x, dict_y, dict_label, colors, title, xlabel, ylabel, mapfilepath, times=0):
    def f_1(x, A, B):   #直线方程函数
        return A*x + B
    def f_2(x, A, B, C):    #二次曲线方程
        return A*x*x + B*x + C
    def f_3(x, A, B, C, D): #三次曲线方程
        return A*x*x*x + B*x*x + C*x + D
    # plt.figure(figsize=(5, 4))
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in range(len(dict_x)):
        assert dict_x.has_key(i)
        plt.scatter(dict_x[i], dict_y[i], 2, "red")
        if times==1:
            A1, B1 = optimize.curve_fit(f_1, dict_x[i], dict_y[i])[0]
            dict_x[i] = np.arange(0, 1, 0.01)
            dict_y[i] = A1*dict_x[i] + B1
        elif times==2:
            A2, B2, C2 = optimize.curve_fit(f_2, dict_x[i], dict_y[i])[0]  
            dict_x[i] = np.arange(0, 1, 0.01)
            dict_y[i] = A2*dict_x[i]*dict_x[i] + B2*dict_x[i] + C2
        elif times==3:
            A3, B3, C3, D3= optimize.curve_fit(f_3, dict_x[i], dict_y[i])[0]
            dict_x[i] = np.arange(0, 1, 0.01)
            dict_y[i] = A3*dict_x[i]*dict_x[i]*dict_x[i] + B3*dict_x[i]*dict_x[i] + C3*dict_x[i] + D3
        plt.plot(dict_x[i], dict_y[i], color=colors[i], linewidth=1, linestyle="-", label=dict_label[i])
    plt.legend(loc="lower right")
    # plt.xlim( 0, 1 )
    # plt.ylim( 0, 1 )
    plt.savefig(mapfilepath)
    plt.clf()


def plot_confusion_matrix(cm, save_path='confusion_matrix.png', normalize=False, classes=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
    plt.title('confusion_matrix', fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, classes or tick_marks)
    plt.yticks(tick_marks, classes or tick_marks)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "%.1f%%"%(cm[i,j]*100) if normalize else "%d"%cm[i,j],
                 horizontalalignment="center", fontsize=12,
                 color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig.set_size_inches(10.2, 8.5)
    plt.savefig( save_path, dpi=200)
    plt.close()
    np.save( save_path, cm)

