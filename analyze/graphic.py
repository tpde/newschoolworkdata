#! -*- coding:utf-8 -*-
import pylab
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from ..tools.date import cropDate

is_grid = False
is_reset = False #True  False
if is_reset:
    sns.set()
    
# from xpinyin import Pinyin
# ''.join( Pinyin().get_pinyin(mystr).split('-') ).capitalize()

# Cs=['b','r','g','m','c', 'y', 'k', '#7e7e7e','#cd919e','#FF9900',
#         '#660033','#560033','#460033','#360033', '#260033', '#160033']
# Cs=['b','m','g','r','y','c',]

Cs=['g','r','b','m','c', 'y', 'k', '#7e7e7e','#cd919e','#FF9900',
        '#660033','#560033','#460033','#360033', '#260033', '#160033']

Ss = np.ones((15,))*70 # np.ones((15,))*70
Ms = ['.', 'o', '^', '*', 'p', '+', 'x', 'h']
# Ms = ['.' for _ in range(15)]
# Ms = ['^', ]


colorMAP = [['#99FF99','#CCFFCC'],  #   '#FF851B'
            ['#FF99FF','#FFCCFF'],
            ['#99CCFF','#CCECFF'],
            ['#FF9999','#FFCCCC'],
            ['#66FFFF','#CCFFFF'],
            ['#FFFF66','#FFFFCC'],]

def whichIpt(ipts, idx, key, offer):
    if ipts is None:
        ipt = offer[idx]
    elif isinstance(ipts, int):
        ipt = ipts
    elif isinstance(ipts, list):
        ipt = ipts[idx]
    elif isinstance(ipts, dict):
        ipt = ipts[key]
    return ipt

def configFig(dict_font, title, xlabel, ylabel, savepath, xrange=None, yrange=None, lgloc='upper right'):
    # lgloc: ["lower left", "upper right", "best"] or {'anchor':(0.98, 1), 'ncol':4}
    pylab.grid(is_grid)
    if xrange:
        pylab.xlim(xrange[0], xrange[1])
    if yrange:
        pylab.ylim(yrange[0], yrange[1])
    # 设置文字大小    # plt.text(17, 277, 'loss curve')
    plt.setp(pylab.title(title), size='large', color='k', fontsize=dict_font['title'])
    plt.setp(pylab.xlabel(xlabel, fontsize=dict_font['xylabel']), weight='light', color='k')
    plt.setp(pylab.ylabel(ylabel, fontsize=dict_font['xylabel']), weight='light',color='k')
    plt.xticks(fontsize=dict_font['xytick'])
    plt.yticks(fontsize=dict_font['xytick'])
    
    if isinstance(lgloc, str):
        plt.legend(fontsize=dict_font['legend'], loc=lgloc)
    elif isinstance(lgloc, dict):
        plt.legend(fontsize=dict_font['legend'], loc='upper center',
                   bbox_to_anchor=lgloc['anchor'], ncol=lgloc['ncol'])
    else:
        raise( TypeError, 'lgloc must be str or dict')
    plt.savefig(savepath)   # plt.show()
    plt.close()
    plt.clf()

def plot_curves( dict_curves, title, xlabel, ylabel, savepath, is_scatter=False, fsize=(10,10),
                 xrange=None, yrange=None, colors=None, sizes=None, markers=None, lgloc = "upper right",
                 dict_font={'title':30, 'xylabel':28, 'xytick':24, 'legend':26}):
#     pylab.xlim(115.6, 117.0)
#     pylab.ylim(39.1, 40.5)
    plt.figure(figsize=fsize) if fsize else plt.figure()
    idx = 0
    for key,curve in dict_curves.items():
        color = whichIpt(colors, idx, key, Cs)
        size = whichIpt(sizes, idx, key, Ss)
        marker = whichIpt(markers, idx, key, Ms)
        if isinstance(curve, dict):
            if is_scatter:
                if False:
                    plt.scatter(curve['x'], curve['y'], marker=marker, color=color, label=key, s=size)
                else:
                    plt.scatter(curve['x'], curve['y'], marker=marker, color=color, s=size)
                    poly = np.polyfit(curve['x'], curve['y'], deg = 1)
                    plt.plot(np.array([xrange[0]-0.1, xrange[1]+0.1]), np.polyval(poly, np.array(xrange)),
                             marker=marker, color=color, label=key)
                if 'txt' in curve:
                    for i,txt in enumerate(curve['txt']):
                        x_txt = curve['x'][i] + (curve['txtOffset'][0] if 'txtOffset' in curve else 0)
                        y_txt = curve['y'][i] + (curve['txtOffset'][1] if 'txtOffset' in curve else 0)
                        plt.annotate(txt, (x_txt, y_txt), fontsize=dict_font['legend']-4, color='k', ) # 'k' or color
            else:
                # plt.step(curve['x'], curve['y'], colors, where='post', label=key, linewidth=1.5)
                # plt.plot(curve['x'], curve['y'], '-', marker='.', color=color, label=key, linewidth=1 )
                plt.plot(curve['x'], curve['y'], color=color, label=key)
#                 plt.xlim(min(curve['x']), max(curve['x']))
            if 'err' in curve:
                if isinstance(curve['err'], dict):
                    plt.errorbar(curve['err']['x'], curve['err']['y'], yerr=curve['err']['err'], fmt='r.', ecolor=color, elinewidth=2, capsize=4 )
                else: # isinstance(curve['err'], list):
                    plt.errorbar(curve['x'], curve['y'], yerr=curve['err'], fmt='r.', ecolor=color, elinewidth=2, capsize=4 )
                    
        elif isinstance(curve, list):
            curve_x = np.array(range(len(curve)))+1
            if is_scatter:
                plt.scatter(curve_x, curve, marker=marker, color=color, s=size)
            else:
                plt.plot(curve_x, curve, color, label=key, linewidth=1.5)
        else:
            raise(TypeError)
        idx = idx+1
    configFig(dict_font, title, xlabel, ylabel, savepath, xrange, yrange, lgloc=lgloc)

def plot_datecurve( dict_curves, savepath, title, xlabel, ylabel, struct='%Y-%m',
                    fsize=(300,10), trange=None, yrange=None,
                    colors=None, markers=None,
                    dict_font={'title':30, 'xylabel':28, 'xytick':24, 'legend':26},
                    lgloc = "upper right" ):
    # struct: datetime.strptime('2018-12-31 00:00:00', '%Y-%m-%d %H:%M:%S').strftime("%Y%m%d%H%M%S")  'YYYY-mm-dd HH:MM:SS'
    pylab.figure(figsize=fsize)
    idx = 0
    for key,curve in dict_curves.items():
        color = whichIpt(colors, idx, key, Cs)
        marker = whichIpt(markers, idx, key, Ms)
        curve_date = [datetime.strptime(date, struct) for date in curve['date']] if struct != '' else curve['date']
        if trange:
            idxDn = 0
            num = len( curve_date )
            while( idxDn<num and curve_date[idxDn] < trange[0]):
                idxDn += 1
            idxUp = idxDn
            while( idxUp<num and curve_date[idxUp] < trange[1] ):
                idxUp += 1
            curve_date = curve_date[idxDn:idxUp]
            curve['data'] = curve['data'][idxDn:idxUp]
            # print(curve_date)
        if len(curve_date)>0:
            pylab.plot_date( pylab.date2num(curve_date), curve['data'], color,
                            marker=marker, label=key, linestyle='-')
            if 'err' in curve:
                pylab.errorbar(pylab.date2num(curve_date), curve['data'], yerr=curve['err'],
                                fmt='-o', ecolor=color, color=color, elinewidth=2, capsize=4 )
            if 'txt' in curve:
                for i,txt in enumerate(curve['txt']):
                    x_txt = pylab.date2num(curve_date[i]) + (curve['txtOffset'][0] if 'txtOffset' in curve else 0)
                    y_txt = curve['data'][i] + (curve['txtOffset'][1] if 'txtOffset' in curve else 0)
                    pylab.annotate(txt, (x_txt, y_txt), fontsize=dict_font['legend']-4, color='k')
        idx += 1
    configFig(dict_font, title, xlabel, ylabel, savepath, yrange=yrange, lgloc=lgloc)
    
    
def plot_pairdatecurve(dict_curves_, savepath, title, xlabel, ylabel, struct='%Y-%m',
                       fsize=(30,10), trange=None, yrange=None, colors=None, markers=None,
                       dict_font={'title':30, 'xylabel':28, 'xytick':24, 'legend':24},
                       lgloc = "upper right",      # {'anchor':(0.98, 1), 'ncol':4}
                       list_skey0=None,     ):
    # struct: datetime.strptime('2018-12-31 00:00:00', '%Y-%m-%d %H:%M:%S').strftime("%Y%m%d%H%M%S")  'YYYY-mm-dd HH:MM:SS'
    # markers if is dict, key of markers musk be curve.keys() not be dict_curves.keys()
#     dict_curves = dict_curves.copy()
    dict_curves ={}
    for key,value in dict_curves_.items():
        dict_curves[key] = {skey:sv for skey,sv in value.items()}
    list_style = [ '-', ':', '--', '-.',]
    pylab.figure(figsize=fsize)
    idx = 0
    for key,curve in dict_curves.items():
        color = whichIpt(colors, idx, key, Cs)
        curve['date'] = [datetime.strptime(date, struct) for date in curve['date']] if struct != '' else curve['date']
        if trange:
            idxDn, idxUp = cropDate( curve['date'], trange )
            for skey in curve:
                curve[skey] = curve[skey][idxDn:idxUp]
        if len(curve['date'])>0:
            if list_skey0 is not None:
                list_skey = list_skey0
            else:
                list_skey = [skey for skey in curve if skey!='date']
#             pylab.plot_date(pylab.date2num(curve_date), curve[list_skey[0]], colors[i],
#                             marker='o', label=key+'(%s)'%(list_skey[0]), linestyle='-')
#             pylab.plot_date(pylab.date2num(curve_date), curve[list_skey[1]], colors[i],
#                             marker='o', label=key+'(%s)'%(list_skey[1]), linestyle='--')
            for j,skey in enumerate(list_skey):
                marker = whichIpt(markers, j, skey, Ms)
                pylab.plot_date(pylab.date2num(curve['date']), curve[skey], color,
                            marker=marker, label=key+'(%s)'%skey, linestyle=list_style[j], linewidth=2 )
        idx += 1
        
    configFig(dict_font, title, xlabel, ylabel, savepath, yrange=yrange, lgloc=lgloc)


def plot_groupbar( list_xticks, dict_bars, title, xlabel, ylabel, savepath,
                   bar_w=0.3, fsize=(10,10), xrange=None, yrange=None,
                   colors=['lightgreen', 'cornflowerblue', 'r', 'y'],
                   dict_font={'title':30, 'xylabel':28, 'xytick':24, 'legend':26},
                   lgloc = "upper right", list_group=None):

    if list_group==None:
        list_group = list(dict_bars.keys())
    else:
        assert( len(list_group)==len(dict_bars) )
    n_group = len(dict_bars)
    
    n_bar = len(list_xticks)
    inds = np.arange(n_bar)
    
    plt.figure(figsize=fsize)
    for idx,key in enumerate(list_group):
        color = whichIpt(colors, idx, key, Cs)
        bar = dict_bars[key]
        if isinstance( bar, list ):
            assert( len(bar)==n_bar )
            plt.bar(inds+bar_w*(idx+1/2), bar, bar_w, label=key, color=color)
            
        elif isinstance( bar, dict ):
            assert( len(bar['data'])==n_bar )
            plt.bar(inds+bar_w*(idx+1/2), bar['data'], bar_w, label=key, color=color)
            if 'err' in dict_bars[key]:
                plt.errorbar(inds+bar_w*(idx+1/2), bar['data'], yerr=bar['err'],
                                fmt='r.', ecolor='r', elinewidth=2, capsize=4 )
            if 'txt' in bar:
                for i,txt in enumerate(bar['txt']):
                    if 'txt_x' in bar:
                        txt_x = bar['txt_x'][i] if isinstance(bar['txt_x'], list) else bar['txt_x']
                    else:
                        txt_x = 0
                    txt_x += inds[i]+bar_w*(idx+1/2)
                    
                    if 'txt_y' in bar:
                        txt_y = bar['txt_y'][i] if isinstance(bar['txt_y'], list) else bar['txt_y']
                    else:
                        txt_y = 0
                    txt_y += bar['data'][i]
                    pylab.annotate(txt, (txt_x, txt_y), fontsize=dict_font['legend']-6, color='k')
        else:
            raise( TypeError )
        
    if xrange is None:
        xrange = [-bar_w, np.max(inds+bar_w*(idx+1/2))+bar_w/2]
    plt.xticks(inds+bar_w*n_group/2, list_xticks, fontsize=dict_font['xytick'])
    configFig(dict_font, title, xlabel, ylabel, savepath, xrange, yrange, lgloc)
    

def plot_groupBar2( dict_bars, savepath, title, xlabel, ylabel,
                    padding=0.2, fsize=(10,10), xrange=None, yrange=None, list_group=None,
                    dict_font={'title':30, 'xylabel':28, 'xytick':24, 'legend':26},
                    lgloc = "upper right"):
    """
    dict_bars ={'Net':{'P0':0.75, 'P1':0.70, ...},
                'WRF':{'P0':0.72, 'P1':0.65, ...},    }
    or
    dict_bars ={ 'GC':{'P0':0.75, 'P1':0.70, ...},
                'GKD':{'P0':0.72, 'P1':0.65, ...},    }
    list_group = ['P0', 'P1', ... ]    can get from modifing print(list_group)
    """
    list_group = list_group or sorted(list( list(dict_bars.values())[0].keys() ))
#     print(list_group)     # can using for getting from modifing print(list_group)
    n_group = len(list_group)
    n_eleme = len(dict_bars)
    bar_w = (1-padding) / n_eleme
    for bars in dict_bars.values():
        assert( sorted(list_group)==sorted(list(bars.keys())) )
    
    plt.figure(figsize=fsize) if fsize else plt.figure()
    inds = np.arange(n_group)
    idx = 0
    for key,bars in dict_bars.items():
        list_bar = [bars[group] for group in list_group]
        plt.bar(inds+ bar_w*idx, list_bar, width=bar_w, label=key, fc=Cs[idx])
        idx = idx+1
        
    plt.xticks(inds+bar_w*(len(dict_bars)-1)/2, list_group, fontsize=dict_font['xytick'])
    configFig(dict_font, title, xlabel, ylabel, savepath, xrange, yrange, lgloc)
 

def scatterwithcolorbar(dict_group, title, xlabel, ylabel, savepath,
                        fsize=(8, 6), sizes=None, markers=None, xrange=None, yrange=None,
                        dict_font={'title':30, 'xylabel':28, 'xytick':24, 'legend':26},
                        lgloc = "upper right",):
    Ss = np.ones((15,))*5
#     Ms = ['.', 'o', '^', '*', 'p', '+', 'x', ]
    Ms = ['o' for i in range(15)]
    plt.figure(figsize=fsize, dpi=120) if fsize else plt.figure(dpi=120)
    is_samerange = False
    if len( set([group['min'] for group in dict_group.values() if 'min' in group]) )==1:
        if len( set([group['max'] for group in dict_group.values() if 'min' in group]) )==1:
            is_samerange = True
    idx = 0
    for key,group in dict_group.items():
        if 'c' in group:
            size = whichIpt(sizes, idx, key, Ss)
            marker = whichIpt(markers, idx, key, Ms)
            plt.scatter(group['x'], group['y'], c=group['c'], label=key, s=size, marker=marker,
                        cmap='jet', vmin=group['min'], vmax=group['max'])
        else:
            plt.plot(group['x'], group['y'], '-', marker='.', color='k', linewidth=1 )
        
        
        if 'txt' in group:
#             xoff, yoff = 0, 0.015
            if idx == 0:
                xoff, yoff = -0.07, +0.035
            else:
                xoff, yoff = -0.1, +0.03        
            for i,txt in enumerate(group['txt']):
                plt.annotate(txt, (group['x'][i]+xoff, group['y'][i]+yoff), fontsize=dict_font['legend']-4, color='k')
#         if (not is_samerange) or (is_samerange and idx==0):
            plt.colorbar()
        idx += 1
    configFig(dict_font, title, xlabel, ylabel, savepath, xrange, yrange, lgloc)
    

def plot_windy( dict_curves, title, xlabel, ylabel, savepath, is_scatter=False, fsize=(10,10), 
                colors=None, sizes=None, markers=None, xrange=None, yrange=None,
                dict_font={'title':30, 'xylabel':28, 'xytick':24, 'legend':26},
                lgloc = "upper right"):
    
    plt.figure(figsize=fsize) if fsize else plt.figure()
    idx = 0
    for key,curve in dict_curves.items():
        if isinstance(curve, dict):
            color = whichIpt(colors, idx, key, Cs)
            size = whichIpt(sizes, idx, key, Ss)
            marker = whichIpt(markers, idx, key, Ms)
            if is_scatter:
                plt.scatter(curve['x'], curve['y'], marker=marker, color=colors[idx], label=key, s=size)
                if 'txt' in curve:
                    for i,txt in enumerate(curve['txt']):
                        plt.annotate(txt, (curve['x'][i]-0.025, curve['y'][i]+0.015), fontsize=20, color=colors[idx])
                        # plt.annotate(txt, (curve['x'][i], curve['y'][i]), fontsize=20, color=colors[idx])
            else:
                # plt.step(curve['x'], curve['y'], colors[idx], where='post', label=key, linewidth=1.5)
                plt.plot(curve['x'], curve['y'], '-', marker='.', color=color, label=key, linewidth=1 )
#                 plt.xlim(min(curve['x']), max(curve['x']))
        elif isinstance(curve, list):
            curve_x = np.array(range(len(curve)))+1
            plt.step(curve_x, curve, colors[idx], where='post', label=key, linewidth=1.5)
            if is_scatter:
                plt.scatter(curve_x, curve, marker='o', color='m', s=30)
        else:
            raise(TypeError)
        idx = idx+1
#     pylab.xlim(115.6, 117.0)
#     pylab.ylim(39.1, 40.5)
    configFig(dict_font, title, xlabel, ylabel, savepath, xrange, yrange, lgloc)
    
    
def plot_groupIncreaseBar( list_dict_bars, savepath, title, xlabel, ylabel,
                    padding=0.2, fsize=(10,10), xrange=None, yrange=None,
                    list_group=None, list_xtick=None,
                    dict_font={'title':30, 'xylabel':28, 'xytick':24, 'legend':26},
                    lgloc = "upper right"):

    """
    list_dict_bars = [dict_bars1, dict_bars2, dict_bars3, ...]    # response ['WRF','Net', ...]
    = [{'GC':{'P0':0.75, 'P1':0.70, ...},    # this dict_bar is generated by WRF
       'GKD':{'P0':0.72, 'P1':0.65, ...},
        'NC':{'P0':0.72, 'P1':0.65, ...},
       'QKY':{'P0':0.72, 'P1':0.65, ...},},    
       {'GC':{'P0':0.75, 'P1':0.70, ...},    # this dict_bar is generated by Net
       'GKD':{'P0':0.72, 'P1':0.65, ...},
        'NC':{'P0':0.72, 'P1':0.65, ...},
       'QKY':{'P0':0.72, 'P1':0.65, ...},},]    
        
    list_group = ['P0', 'P1', ... ]    can get from modifing print(list_group)
    """
    dict_bars = list_dict_bars[0]
    list_group = list_group or sorted(list( list(dict_bars.values())[0].keys() ))
    # print(list_group)     # can using for getting from modifing print(list_group)
    n_group = len(list_group)
    n_eleme = len(dict_bars)
    bar_w = (1-padding) / n_eleme
    for dict_bars in list_dict_bars:
        for bars in dict_bars.values():
            assert( sorted(list_group)==sorted(list(bars.keys())) )
    
    plt.figure(figsize=fsize) if fsize else plt.figure()
    ind = np.arange(n_group)
    for ith,dict_bars in enumerate( list_dict_bars ):
        idx = 0
        for key,bars in dict_bars.items():
            if ith==0:
                list_bar = [bars[group] for group in list_group]
                plt.bar(ind+ bar_w*idx, list_bar, width=bar_w, label=key, fc=colorMAP[idx][ith])
            else:
                list_bar0 = [list_dict_bars[ith-1][key][group] for group in list_group]
                list_bar = [bars[group]-list_bar0[i] for i,group in enumerate(list_group)]
                plt.bar(ind+ bar_w*idx, list_bar, width=bar_w, bottom=list_bar0, fc=colorMAP[idx][ith])
            idx = idx+1
        
    if list_xtick:
        plt.xticks(ind+bar_w*(len(dict_bars)-1)/2, list_xtick, fontsize=dict_font['xytick'])
    else:
        plt.xticks(ind+bar_w*(len(dict_bars)-1)/2, list_group, fontsize=dict_font['xytick'])
    configFig(dict_font, title, xlabel, ylabel, savepath, xrange, yrange, lgloc)


def plot_histgram( dict_data, savepath, title, xlabel, ylabel, fsize=(12,10),
                 xrange=None, yrange=None, colors=None, lgloc = "upper right",
                 dict_font={'title':30, 'xylabel':26, 'xytick':24, 'legend':24}, n_bins=20):
    # https://www.cnblogs.com/feffery/p/11128113.html
#     pylab.xlim(115.6, 117.0)
#     pylab.ylim(39.1, 40.5)
    plt.figure(figsize=fsize) if fsize else plt.figure()
    idx = 0
    for key,data in dict_data.items():
        color = whichIpt(colors, idx, key, Cs)
        # hist, bin_edges = np.histogram( arr_diff )
        # plt.hist(x=arr_diff, bins = 20, color = 'steelblue', edgecolor = 'black')
        # sns.distplot(data, color="g",)  # 默认
        # sns.distplot(data, kde=False) # 隐藏数据趋势线kde
        # sns.distplot(data, hist=False, rug=True)  # 隐藏直方图hist
        # sns.distplot(data, kde=False, fit=stats.gamma)  # 显示数据紧密度fit
        if False:
            sns.distplot(data, bins=n_bins, kde=False, color=color, fit=stats.gamma, fit_kws={'color':color}, label=key)
        else:   
            sns.distplot(data, bins=n_bins, hist=True, color=color, vertical=True, norm_hist=False, label=key) #核函数拟合 = 直方图 + 拟合曲线
#             sns.distplot(data, bins=n_bins, hist=False, kde=True, color=color, vertical=True, norm_hist=False) #只画拟合曲线，不画直方图
#             sns.distplot(data, bins=n_bins, kde=False, color=color, vertical=True, norm_hist=False, label=key) #只画直方图频数，不画拟合曲线
            
        idx = idx+1
    configFig(dict_font, title, xlabel, ylabel, savepath, xrange, yrange, lgloc=lgloc)

# from Ailib.io import readexcel
# 
# xlsNet = '/home/tpde/project/DataBase/WorkCov5_2019/depend/AvgPred_4Station/eval_Net.xls'
# dict_Net = readexcel(xlsNet, struct='dictdict', axis='v')
# 
# xlsWRF = '/home/tpde/project/DataBase/WorkCov5_2019/depend/AvgPred_4Station/eval_WRF.xls'
# dict_WRF = readexcel(xlsWRF, struct='dictdict', axis='v')
# 
# site = 'GC'
# dict_bars = {'WRF':dict_WRF[site],
#              'Net':dict_Net[site], }
# plot_groupBar2( dict_bars, savepath='%s.png'%site,
#                title='WRF vs Net (4-fold)', xlabel='Attitudes', ylabel='Pearson',
#                padding=0.2, fsize=None, yrange=None)
# 
# list_dict_bars = [dict_WRF, dict_Net]
# plot_groupIncreaseBar( list_dict_bars, savepath='all.png',
#                title='WRF vs Net (4-fold)', xlabel='Attitudes', ylabel='Pearson',
#                padding=0.2, fsize=(20,7), k=2, yrange=None,
#                list_group = ['P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_Avg', 'P_3D'])