#! -*- coding:utf-8 -*-
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from ...basic import *
from ...tools import listKit

def transvertgif2png(dict_img):
    def getframes(img):
        mypalette = img.getpalette()
        try:
            while 1:
                img.putpalette(mypalette)
                dst = Image.new(img.mode, img.size)
                dst.paste(img)
                img.seek(img.tell()+1)
        except EOFError:
            pass
        return dst

    dict_dst = dict()
    for name,img in dict_img.items():
        dict_dst[name] = getframes(img)
    return dict_dst

def transvert2gray(dict_img):
    for name,img in dict_img.items():
        dict_img[name] = img.convert('L')
    return dict_img
    
def thresh_binary( dict_img, thresh ):
    table = []
    for i in range(256):
        if i < thresh:
            table.append(0)
        else :
            table.append(1)
    
    dict_dst = dict()
    for name,img in dict_img.items():
        gray = img.convert('L')
        dst = gray.point(table, "1")
        dict_dst[name] = dst
    return dict_dst


def unite_size(dict_img, size_w, list_hs=None, is_stretch=False, is_binary=False, is_padding=False, is_recover=False):
    dict_dst = dict()
    dict_resize = dict()
    for name,img in dict_img.items():
        img_w, img_h = img.size
        if is_padding:
            dst = Image.new( img.mode, (size_w, list_hs))
            start_x = int((size_w-img_w)/2)
            start_y = int((list_hs-img_h)/2)
            dst.paste(img, box=(start_x, start_y, start_x+img_w, start_y+img_h) )
            dict_resize[name] = ['crop', [start_x, start_y, start_x+img_w, start_y+img_h]]
        else:
            if list_hs is None:
                size_h = int(size_w*img_h/img_w)
                dst = img.resize( (size_w, size_h) )
                dict_resize[name] = ['resize', [img_w, img_h]]
            else:
                target_h = int(size_w*img_h/img_w)
                size_h = listKit.find_nearest(list_hs, target_h)
                if is_stretch:
                    dst = img.resize( (size_w, size_h) )
                    dict_resize[name] = ['resize', [img_w, img_h]]
                else:
                    h = img_w*size_h/(size_w+0.)
                    crop_y1 = int(img_h/2.-h/2.)
                    crop_y2 = int(img_h/2.+h/2.)
                    dst = img.crop( box = (0, crop_y1, img_w, crop_y2) )
                    dst = dst.resize( (size_w, size_h) )
                    dict_resize[name] = ['resize_crop', [img_w, crop_y2-crop_y1],
                                         [0, -crop_y1, img_w, img_h-crop_y1] ]
        dict_dst[name] = dst
    if is_binary:
        dict_dst = thresh_binary( dict_dst, 127 )
    if is_recover:
        return dict_dst, dict_resize
    else:
        return dict_dst


def deresize(dict_img, dict_rsiz):
    for name, img in dict_img.items():
        rsiz = dict_rsiz[name]
        if rsiz[0] == 'resize':
            img = img.resize(rsiz[1])
        elif rsiz[0] == 'crop':
            img = img.crop( box=rsiz[1] )
        else:
            img = img.resize(rsiz[1])
            img = img.crop( box=rsiz[2] )
        dict_img[name] = img
    return dict_img
            
    
def combineimages( list_dict_img, combinein='width'):
    assert( all( [len(dict_img) == len(list_dict_img[0]) for dict_img in list_dict_img] ) )
    dict_combine = dict()
    for key in list_dict_img[0]:
        img1 = list_dict_img[0].get(key)
        if combinein == 'width':
            assert( all( [img1.size[1] == dict_img.get(key).size[1] for dict_img in list_dict_img] ) )
            list_ws = [dict_img.get(key).size[0] for dict_img in list_dict_img]
            dst = Image.new( img1.mode, (sum(list_ws), img1.size[1]))
            for idx,dict_img in enumerate(list_dict_img):
                img = dict_img.get(key)
                dst.paste(img, box=(sum(list_ws[:idx]), 0, sum(list_ws[:idx+1]), img.size[1]) )
        else:
            assert( all( [img1.size[0] == dict_img.get(key).size[0] for dict_img in list_dict_img] ) )
            list_hs = [dict_img.get(key).size[1] for dict_img in list_dict_img]
            dst = Image.new( img1.mode, (img1.size[0],sum(list_hs)))
            for idx,dict_img in enumerate(list_dict_img):
                img = dict_img.get(key)
                dst.paste(img, box=(0,sum(list_hs[:idx]), img.size[0],sum(list_hs[:idx+1])) )
        dict_combine[key] = dst
    return dict_combine

def combine2mask(dict_gro1, dict_gro2):
    dict_dst = {}
    for key in dict_gro1:
        gro1 = np.array( dict_gro1[key] )
        gro2 = np.array( dict_gro2[key] )
        gro = (gro1>0)|(gro2>0)
        dict_dst[key] = Image.fromarray( gro.astype(np.uint8)*255 )
    return dict_dst
    
def split_img(dict_combine, axis=0, num=2):
    # img.crop( (left, upper, right, lower) )
    list_dict = list()
    for i in range(num):
        list_dict.append(dict())
        
    for key, img in dict_combine.items():
        img_w, img_h = img.size
        if axis == 0: # width split
            for idx in range(num):
                list_dict[idx][key] = img.crop( (idx*img_w/num,0,(idx+1)*img_w/num,img_h) )
        else:
            for idx in range(num):
                list_dict[idx][key] = img.crop( (0,idx*img_h/num,img_w,(idx+1)*img_h/num) )
    return list_dict


def pastemask(dict_ori, dict_gro, color):
    dict_dst = dict()
    for key,ori in dict_ori.items():
        assert dict_gro.has_key(key)
        gro = dict_gro.get(key)
        gro = gro.resize(ori.size)
        img = Image.new("RGB",ori.size, color)
        dst = Image.composite(img, ori, gro)
        dict_dst[key] = dst
    return dict_dst

def same_size(dict_src, dict_target):
    dict_dst = dict()
    for key,src in dict_src.items():
        assert dict_target.has_key(key)
        target = dict_target.get(key)
        dst = src.resize(target.size)
        dict_dst[key] = dst
    return dict_dst



def puzzel_patches(patches_dir, cols, padding, savepath):
    files = os.listdir(patches_dir)
    assert len(files)%cols==0
    tmp = Image.open(os.path.join(patches_dir,files[0]))
    patch_w, patch_h = tmp.size
    dst = Image.fromarray((np.ones(((patch_h+padding)*len(files)/cols-padding, (patch_w+padding)*cols-padding, 3))*255).astype(np.uint8))
    for row in range(len(files)/cols):
        for col in range(cols):
            patch = Image.open( os.path.join(patches_dir,files[row*cols+col]) )
            dst.paste( patch, box=(col*(patch_w+padding), row*(patch_h+padding), col*(patch_w+padding)+patch_w, row*(patch_h+padding)+patch_h) )
    dst.save(savepath)

# puzzel_img('ori', 4, 5, 'dst.png')
        
# def getGroundTruthConnectionMax(dir_src, dir_dst, orderbit_start, orderbit_len):
#     def getConnectionMax(imgArray):
#         debug(np.unique(imgArray))
#         imgNotZero = (imgArray>0).astype(int)
#         dstImage = np.zeros(imgArray.shape)
#         connections = measure.label(imgNotZero, background=0, connectivity=2)
#         label = 1
#         while (label <= connections.max()):
#             connection = (connections==label).astype(int)
#             dstImage += ( connection*imgArray == np.max(connection*imgArray) ).astype(int)
#             label += 1
#         debug(np.unique(dstImage))
#         return dstImage
# 
#     files = readdir(dir_src)
#     mkdir(dir_dst)
#     files = sorted(files, key=(lambda file:int(file[orderbit_start:orderbit_start+orderbit_len]))) 
# 
#     for i in range(len(files)):
#         file = files[i]
#         src = cv2.imread( os.path.join(dir_src,file), 0)
#         # src = cv2.resize(src, (128, 128))
#         imgArray = np.asarray(src)
#         debug(imgArray.shape)
#         dstImage = getConnectionMax(imgArray)*255
#         dstImg = Image.fromarray(dstImage.astype(np.uint8))
#         if( dir_src == dir_dst ):
#             os.remove(os.path.join(dir_src,file))
#         dstImg.save( os.path.join(dir_dst,file) )

# def combineTwoGroundTruth( imgs_dir_bv, imgs_dir_ex, imgs_dir_dst ):
#     if not os.path.exists(imgs_dir_dst):
#         os.system('mkdir -p ' +imgs_dir_dst)
# 
#     imgsArray_bv = getImgsArray(imgs_dir_bv, 1)
#     print '---',np.max(imgsArray_bv), np.min(imgsArray_bv)
#     imgsArray_ex = getImgsArray(imgs_dir_ex, 1)
#     print '---',np.max(imgsArray_ex), np.min(imgsArray_ex)
# 
#     imgsArray = imgsArray_bv + imgsArray_ex
#     overArray = (imgsArray>255).astype(int)
#     imgsArray = imgsArray*(1-overArray)+255*overArray
# 
#     for i in range(imgsArray.shape[0]):
#         imgArray = imgsArray[i]
#         # groupVisualize(imgsArray, numsPerRow, filepath)
#         visualize( group_images( np.reshape(imgArray, (1,imgArray.shape[0],imgArray.shape[1],imgArray.shape[2])), 1), imgs_dir_dst+'%03d.png'%(i+1))

