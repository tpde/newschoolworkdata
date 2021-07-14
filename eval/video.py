import numpy as np

def evaluate(arr_label, arr_pred):
    mae = np.absolute(arr_label-arr_pred).mean()
    mse = np.square(arr_label-arr_pred).mean()
    
    psnr = np.square( (arr_label*255).astype(np.uint8) - (arr_pred*255).astype(np.uint8) ).mean()
    psnr = 20 * np.log10(255) - 10 * np.log10(psnr)
    return mae, mse, psnr