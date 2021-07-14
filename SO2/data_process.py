import cv2
import os
import xlwt
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import morphology,color,measure

def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return dst

def make_CLAHE(img):
    img=img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=20,tileGridSize=(8,8))
    img = clahe.apply(img)
    return img

def skel2vessel(vessel_ori, vessel_skel):
#     skels, distance = morphology.medial_axis(vessel_ori, return_distance=True)
#     img = Image.fromarray((vessel_ori*255).astype(np.uint8) )
#     img.save('vessel_ori.png')
    skel_label = np.unique(vessel_skel)
    kernal = np.ones((13,13), np.uint8)
    img_new = np.zeros(vessel_ori.shape, np.uint8)
    for label in skel_label[1:]:
        vessel_single_skel = ((vessel_skel==label).astype(np.uint8))
#         img = Image.fromarray((vessel_single_skel*255).astype(np.uint8) )
#         img.save('vessel_single_skel.png')
        dil_vessel = cv2.dilate(vessel_single_skel,kernal,iterations=1)
        dil_vessel = dil_vessel*vessel_ori
        crows, cols = np.where(dil_vessel>0)
        img_new[crows, cols] = label
#     img = color.label2rgb(img_new)
#     img = Image.fromarray((img*255).astype(np.uint8) )
#     img.save('dil_vessel.png')
    img_new.reshape(vessel_ori.shape)
    return img_new

def digui(arr_vessel, mask_vessel,new_row, new_col):
    row_dst, col_dst = np.where(mask_vessel>0)
    row_branchs = []
    col_branchs = []
    for i in range(len(row_dst)):
        row_branch = []
        col_branch = []
        rows = row_dst[i]
        cols = col_dst[i]
        for j in range(8):
            mask = np.zeros(mask_vessel.shape, np.uint8)
            row_branch.append(rows)
            col_branch.append(cols)
            arr_vessel[rows,cols] = 0
            mask = cv2.rectangle(mask,(cols-1,rows-1),(cols+1, rows+1),(1,1),-1 )
            mask_vessel = arr_vessel*mask
            if np.sum(mask_vessel>0)>1:
                digui(arr_vessel, mask_vessel, new_row, new_col)
            rows, cols = np.where(mask_vessel>0)
            if len(rows)==0:
                break
            arr_vessel[rows,cols] = 0
            rows = rows[0]
            cols = cols[0]
        row_branchs.append(row_branch)
        col_branchs.append(col_branch)
    if len(new_row)>7:
        v_ori = [new_row[-1]-new_row[-8], new_col[-1]-new_col[-8]]
    else:v_ori = [new_row[-1]-new_row[0], new_col[-1]-new_col[0]]
    cos = []
    for j in range(len(row_branchs)):
        if (len(row_branchs[j]))>1 :
            v_branch = [row_branchs[j][-1]-row_branchs[j][0],col_branchs[j][-1]-col_branchs[j][0] ]
            cos.append((v_ori[0]*v_branch[0]+v_ori[1]*v_branch[1])/(np.linalg.norm([v_ori[0], v_ori[1]]) * np.linalg.norm([v_branch[0], v_branch[1]])) )
        else:cos.append(-1)
    ind = cos.index(max(cos) )
    return np.array(row_branchs[ind]), np.array(col_branchs[ind])
    
def find_vessel(arr_vessel, arr_OD):
    arr_OD = arr_OD.astype(np.uint8)
    arr_vessel = arr_vessel.astype(np.uint8)
    row_OD, col_OD = np.where(arr_OD>0)
    row_OD = int(np.mean(row_OD) )
    col_OD = int(np.mean(col_OD) )
    
    OD_kernal = np.ones((13,13), np.uint8)
    dil_OD = cv2.dilate(arr_OD,OD_kernal,iterations=1)
    arr_vessel = arr_vessel*(1-dil_OD)
    T_filter = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    img_conv1 = ndimage.convolve(arr_vessel, T_filter, cval=1)
    rows_T, cols_T = np.where(img_conv1==4)
    arr_vessel[rows_T,cols_T] = 0
    
    neighb_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    img_conv = ndimage.convolve(arr_vessel, neighb_filter, cval=1)
    img_cross = ((img_conv==2).astype(np.uint8))
    rows, cols = np.where(img_cross>0)
    pixel_value = arr_vessel[rows,cols]
    rows = rows[np.where(pixel_value>0)]
    cols = cols[np.where(pixel_value>0)]
    new_image = np.zeros(arr_vessel.shape, np.uint8)
    mask = np.zeros(arr_vessel.shape, np.uint8)
    for ind in range(len(rows)):
        d_OD = ( (rows[ind]-row_OD)**2+(cols[ind]-col_OD)**2 )**0.5
        new_row = []
        new_col = []
        if d_OD > 100:
            new_row.append(rows[ind])
            new_col.append(cols[ind])
            arr_vessel[rows[ind],cols[ind]] = 0
            while True:
                mask = np.zeros(arr_vessel.shape, np.uint8)
                mask = cv2.rectangle(mask,(cols[ind]-1,rows[ind]-1),(cols[ind]+1, rows[ind]+1),(1,1),-1 )
                mask_vessel = arr_vessel*mask
                if np.sum(mask_vessel>0) ==1 :
                    list_row, list_col = np.where(mask_vessel>0)
                if np.sum(mask_vessel>0) >1 :
                    list_row, list_col = digui(arr_vessel, mask_vessel,new_row, new_col)
                else: 
                    break
                new_row.extend(list_row)
                new_col.extend(list_col)
                arr_vessel[list_row,list_col] = 0
        new_image[new_row,new_col] = ind+1
    dst = color.label2rgb(new_image)      
    img = Image.fromarray( (dst*255).astype(np.uint8) )
    img.save('skel_rgb.png')
    return new_image
    
def find_single_vessel(arr_vessel, arr_OD, arr_RGB):
    _,arr_OD = cv2.threshold(arr_OD,20,1,cv2.THRESH_BINARY)
    _,arr_vessel = cv2.threshold(arr_vessel,20,1,cv2.THRESH_BINARY)
    arr_OD = arr_OD.astype(np.uint8)
    arr_vessel = arr_vessel.astype(np.uint8)
    OD_kernal = np.ones((13,13), np.uint8)
    dil_OD = cv2.dilate(arr_OD,OD_kernal,iterations=1)
    arr_vessel = arr_vessel*(1-dil_OD)
    vessel_label = measure.label(arr_vessel,connectivity=2)
    labels = np.unique(vessel_label)
    
    dst = color.label2rgb(vessel_label)
    img = Image.fromarray( (dst*255).astype(np.uint8) )
    img.save('label.png')

    img_new_cross = np.zeros(arr_vessel.shape, np.uint8)
    img_vessel_breake = np.zeros(arr_vessel.shape, np.uint8)
    img_skel_breake = np.zeros(arr_vessel.shape, np.uint8)
    for ind in labels[1:]:
        vessel_single = ((vessel_label==ind).astype(np.uint8))
        vessel_single_skel = (morphology.thin(vessel_single) ).astype(np.uint8)
        
        #************find each single vessel cross point****************#
        neighb_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        img_conv = ndimage.convolve(vessel_single_skel, neighb_filter, cval=1)
        img_cross = (img_conv>3).astype(np.uint8)
        #************delete each single vessel T type cross point****************#
        rows, cols = np.where(img_cross>0)
        delete_id = []
        for ide in range(len(rows)):
            center = ( cols[ide],rows[ide])
            img_new = np.zeros(img_cross.shape, np.uint8)
            img_circle = cv2.circle(img_new, center, 4, (1,1),1)
            img_dst = vessel_single_skel*img_circle
            if np.sum(img_dst)<=3:
                delete_id.append(ide)
        rows = np.delete(rows, delete_id)
        cols = np.delete(cols, delete_id)
        img_new_cross[rows,cols] = 1
        
        #************break skel vessel and original vessel****************#
        img_cross_dia = cv2.dilate(img_new_cross,np.ones((16,16), np.uint8),iterations=1)
        img_vessel_breake = img_vessel_breake + vessel_single*(1-img_cross_dia)
        img_skel_breake = img_skel_breake + vessel_single_skel*(1-img_new_cross)
    img = Image.fromarray( (img_vessel_breake.astype(np.uint8))*255 )
    img.save('break.png')
    img = Image.fromarray( (img_skel_breake.astype(np.uint8))*255 )
    img.save('skel_break.png')
    
    #************distinguish vessels****************#
    
    vessel_breake_label = measure.label(img_vessel_breake,connectivity=2)
    skel_breake_label = measure.label(img_skel_breake,connectivity=2)
    
    img_new_vessel = np.zeros(arr_vessel.shape, np.uint8)
    rows, cols = np.where(img_new_cross>0)
    for ide in range(len(rows)):
        center = ( cols[ide],rows[ide])
        img_new = np.zeros(img_new_cross.shape, np.uint8)
        img_circle = cv2.circle(img_new, center, 6, (1,1),1)
        img_dst = img_skel_breake*img_circle
        row_dst, col_dst = np.where(img_dst>0)
        print(len(row_dst))
        if len(row_dst)<=3:
            label_dst = []
            for i in range(len(row_dst)):
                label_dst.append(skel_breake_label[row_dst[i],col_dst[i]])
            for label in label_dst:
                skel_breake_label[skel_breake_label==label]= min(label_dst)
        else:
            row_vector = row_dst-rows[ide]
            col_vector = col_dst-cols[ide]
            label_dst1 = []
            cos = []
            for i in range(1,len(row_vector)):
                cos.append((row_vector[0]*row_vector[i]+col_vector[0]*col_vector[i])/(np.linalg.norm([row_vector[0], col_vector[0]]) * np.linalg.norm([row_vector[i], col_vector[i]])) )
            label_dst1.append(skel_breake_label[row_dst[0],col_dst[0]])
            ind_cos = cos.index( np.min(cos) )
            label_dst1.append(skel_breake_label[row_dst[ind_cos+1],col_dst[ind_cos+1]])
            row_dst = np.delete(row_dst,[0,ind_cos+1])
            col_dst = np.delete(col_dst,[0,ind_cos+1])
            label_dst2 = list( skel_breake_label[row_dst,col_dst] )
        
            for label in label_dst1:
                skel_breake_label[skel_breake_label==label]= np.min(label_dst1)
            for label in label_dst2:
                skel_breake_label[skel_breake_label==label]= np.min(label_dst2)  
    dst = color.label2rgb(skel_breake_label)      
    img = Image.fromarray( (dst*255).astype(np.uint8) )
    img.save('skel_rgb.png')
    
    for label in np.unique(vessel_breake_label)[1:]:
        vessel_breake = (vessel_breake_label==label).astype(np.uint8)
        img_vessel_label = vessel_breake*skel_breake_label
        rows, cols = np.where(vessel_breake>0)
        img_new_vessel[rows,cols] = np.max(np.unique(img_vessel_label) )
    
    label = np.unique(img_new_vessel)
    for k in range(1,len(label)):   
        dst = (img_new_vessel==label[k] ).astype(np.uint8) 
        img = Image.fromarray( (dst*255).astype(np.uint8) )
        img.save('new_vessel_%d.png'%k)
    return img_new_vessel
        

def combineimages( list_dict_img):
    assert( all( [len(dict_img) == len(list_dict_img[0]) for dict_img in list_dict_img] ) )
    dict_combine = dict()
    for key in list_dict_img[0]:
        img1 = list_dict_img[0].get(key)
        assert( all( [img1.size[1] == dict_img.get(key).size[1] for dict_img in list_dict_img] ) )
        list_hs = [dict_img.get(key).size[0] for dict_img in list_dict_img]
        dst = Image.new( img1.mode, (sum(list_hs), img1.size[1]))
        for idx,dict_img in enumerate(list_dict_img):
            img = dict_img.get(key)
            dst.paste(img, box=(sum(list_hs[:idx]), 0, sum(list_hs[:idx+1]), img.size[1]) )
        dict_combine[key] = dst
    return dict_combine

def write_images(dict_img, save_dir, offset=1, keylen=3, prefix='', tail='.png'):
    """ if you want add prefix or tail on filename, please call tools.sortdict.dict_keystandard
    """
    rightkey = lambda keylen,index: '%02d'%(index) if keylen==2     \
                        else '%03d'%(index) if keylen==3        \
                        else '%04d'%(index) if keylen==4        \
                        else '%05d'%(index)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for key, img in dict_img.items():
        if type(key)==type(1):
            newkey = prefix + rightkey(keylen,key+offset) + tail
            if type(img)==type(np.array([])):
                if np.max(img)<=1:
                    img = img*255
                if len(img.shape)==3 and img.shape[2]==1:
                    img = img.reshape((img.shape[0],img.shape[1]))                
                img = Image.fromarray(img.astype(np.uint8))
            img.save( os.path.join(save_dir,newkey) )
        else:
            if type(img)==type(np.array([])):
                if np.max(img)<=1:
                    img = img*255
                if len(img.shape)==3 and img.shape[2]==1:
                    img = img.reshape((img.shape[0],img.shape[1]))
                img = Image.fromarray(img.astype(np.uint8))
            img.save( os.path.join(save_dir,key) )
            # img.save( os.path.join(save_dir,os.path.splitext(key)[0]+'.png') )
            
def save2excel(data, filepath):
#     if not os.path.exists(os.path.split(filepath)[0]):
#         os.makedirs(os.path.split(filepath)[0])
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('data', cell_overwrite_ok=True)
    
    if isinstance(data, list):
        if isinstance(data[0], (int, str)):
            for row in range(len(data)):
                sheet.write(row, 0, data[row])
        elif isinstance(data[0], list):
            for row in range(len(data)):
                for col in range(len( data[row] )):
                    sheet.write(row, col, data[row][col])
        elif isinstance(data[0], dict):
            list_key = list(data[0].keys())
            for key in list_key:
                sheet.write(0, list_key.index(key), '{}'.format(key))
            for row in range(len(data)):
                for key in list_key:
                    sheet.write(row+1, list_key.index(key), '{}'.format(data[row][key]))
        else:
            raise TypeError
    elif isinstance(data, dict):
        list_key = list(data.keys())
        if isinstance(data[list_key[0]], (int, str, float)):
            for key in list_key:
                sheet.write(list_key.index(key),0, key)
                sheet.write(list_key.index(key),1, str(data[key]))         
        elif isinstance(data[list_key[0]], list):
            for key in list_key:
                sheet.write(0,list_key.index(key), key)
                for row,value in enumerate( data[key] ):
                    value = '%.4f'%value if isinstance(value, float) else str(value)
                    sheet.write(row+1, list_key.index(key), value)
        elif isinstance(data[list_key[0]], dict):
            columname = list(data[list_key[0]].keys())
            for col,name in enumerate(columname):
                sheet.write(0, col+1, name )
            for row,key in enumerate(list_key):
                sheet.write((row+1), 0, key)
                for col,value in data[key].items():
                    sheet.write((row+1), (columname.index(col)+1), str(value))
        else:
            raise TypeError
        

