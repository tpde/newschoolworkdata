# from scipy.misc import imsave, imread
# import numpy as np
# import os
# from ...basic import mkdir
# 
# def histmatch(refpath, srcdir, dstdir):
#     if not os.path.exists(dstdir):
#         mkdir(dstdir)
#         
#     src_dir = lambda file:os.path.join(srcdir,file)
#     dst_dir = lambda file:os.path.join(dstdir,file)
#     
#     for file in os.listdir(src_dir('')):
#         imsrc = imread(src_dir(file))
#         imtint = imread(refpath)
#         nbr_bins=255
#         if len(imsrc.shape) < 3:
#             imsrc = imsrc[:,:,np.newaxis]
#             imtint = imtint[:,:,np.newaxis]
#         
#         imres = imsrc.copy()
#         for d in range(imsrc.shape[2]):
#             imhist,bins = np.histogram(imsrc[:,:,d].flatten(),nbr_bins,normed=True)
#             tinthist,bins = np.histogram(imtint[:,:,d].flatten(),nbr_bins,normed=True)
#             cdfsrc = imhist.cumsum() #cumulative distribution function
#             cdfsrc = (255 * cdfsrc / cdfsrc[-1]).astype(np.uint8) #normalize
#             cdftint = tinthist.cumsum() #cumulative distribution function
#             cdftint = (255 * cdftint / cdftint[-1]).astype(np.uint8) #normalize
#             im2 = np.interp(imsrc[:,:,d].flatten(),bins[:-1],cdfsrc)
#             im3 = np.interp(im2,cdftint, bins[:-1])
#             imres[:,:,d] = im3.reshape((imsrc.shape[0],imsrc.shape[1] ))
#             
#         try:
#             imsave(dst_dir(file), imres)
#         except:
#             imsave(dst_dir(file), imres.reshape((imsrc.shape[0],imsrc.shape[1] )))