#! -*- coding:utf-8 -*-
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys,os
import inspect
from os.path import join

DEBUG = 0

def sprint(*args):
    # used at start of function
    if DEBUG == 1:
        print( sys._getframe().f_back.f_code.co_name, sys._getframe().f_back.f_lineno,':' )
    print( args )

def mprint(*args):
    # used at middle of function
    print('\t',args )

def debug(*args):
    # inspect.stack()[1][3]                 caller name
    # sys._getframe().f_lineno              current lineno
    # sys._getframe().f_code.co_name        myprint
    # sys._getframe().f_code.co_filename    
    if DEBUG == 1:
        # caller name, the caller lineno
        print(sys._getframe().f_back.f_code.co_name, sys._getframe().f_back.f_lineno,':' )
        print('\t',args )

def crashes(*args):
    if DEBUG == 1:
        return sys._getframe().f_back.f_code.co_name, sys._getframe().f_back.f_lineno,':\t',args
    return 'crashes:',args

class MyDict(dict):
    """
    Dictionary that allows to access elements with dot notation.
    ex:
        >> d = MyDict({'key': 'val'})
        >> d.key
        'val'
        >> d.key2 = 'val2'
        >> d
        {'key2': 'val2', 'key': 'val'}
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    
def readdir( path ):
    files = os.listdir(path)
    if len(files)==0:
        raise( OSError, 'directory of "%s" empty!!!'%path )
    else:
        return files

def mkdir(path):
    if (path is not '') and (not os.path.exists(path)):
        # os.system('mkdir -p ' +path)
        os.makedirs(path)
        
def rmdir(path):
    if os.path.exists(path):
        os.system('rm -rf ' +path)

def rename(root_dir, dict_replace):
    for file in os.listdir(root_dir):
        srcpath = join(root_dir, file)
        for str1,str2 in dict_replace.items():
            file.replace(str1,str2)
        os.system(  'mv %s %s'%(srcpath, file)  )
        
def rename2( foldir, replace='' ):
    os.system("rename 's/\ /%s/' %s/*"%(replace, foldir) )