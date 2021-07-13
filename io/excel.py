import os
import numpy as np
import xlwt, xlrd
from ..basic import mkdir

def save2excel(data, filepath):
    mkdir(os.path.split(filepath)[0])
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('data', cell_overwrite_ok=True)
    
    if isinstance(data, np.ndarray):
        if len(data.shape)==1:
            data = data.tolist()
        elif len(data.shape)==2:
            rows, cols = data.shape
            data = [[float(data[row,col]) for col in range(cols)] for row in range(rows)]
        else:
            raise( TypeError )
        
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
        if isinstance(data[list_key[0]], (int, str, float, np.ndarray)):
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
    book.save(filepath)
    


def readexcel(filepath, struct, axis='h', dtype='float', list_sheet=[0]):
    if dtype=='float':
        tsf = lambda x:float(x)
    elif dtype=='int':
        tsf = lambda x:int(x)
    elif dtype=='str':
        tsf = lambda x:str(x)
    else:
        raise(TypeError)
    
    fd = xlrd.open_workbook(filepath)
    dict_all = {}
    for sheet in list_sheet:
        if isinstance(sheet, int):
            table = fd.sheets()[sheet]
        else:
            table = fd.sheet_by_name(sheet)
        nrows = table.nrows
        ncols = table.ncols
        
        list_data = []
        dict_data = {}
        if struct=='list':
            if axis=='h':
                for col in range(ncols):
                    list_data.append( tsf(table.cell(0,col).value) )
            else:
                for row in range(nrows):
                    list_data.append( tsf(table.cell(row, 0).value) )
                    
        elif struct=='listlist':
            if axis=='h':            
                for col in range(ncols):
                    list_data.append([])
                    for row in range(nrows):
                        list_data[col].append( tsf(table.cell(row, col).value) )
            else:
                for row in range(nrows):
                    list_data.append([])
                    for col in range(ncols):
                        list_data[row].append( tsf(table.cell(row, col).value) )
    
        elif struct=='listdict':
            if axis=='h':
                rownames = table.col_values(0)
                for col in range(1,ncols):
                    list_data.append({})
                    for row in range(len(rownames)):
                        list_data[col-1][rownames[row]] = tsf(table.cell(row, col).value)
            else:
                colnames = table.row_values(0)
                for row in range(1,nrows):
                    list_data.append({})
                    for col in range(len(colnames)):
                        list_data[row-1][colnames[col]] = tsf(table.cell(row, col).value)
    
        elif struct=='dict':
            if axis=='h':
                colnames = table.row_values(0)
                for col in range(len(colnames)):
                    dict_data[colnames[col]] = tsf(table.cell(1, col).value)
            else:
                rownames = table.col_values(0)
                for row in range(len(rownames)):
                    dict_data[rownames[row]] = tsf(table.cell(row, 1).value)
            
        elif struct=='dictlist':
            if axis=='h':
                colnames = table.row_values(0)
                for col in range(len(colnames)):
                    dict_data[colnames[col]] = []
                    for row in range(1,nrows):
                        v = table.cell(row, col).value
                        if v=='':
                            break
                        else:
                            dict_data[colnames[col]].append( tsf(v) )
            else:
                rownames = table.col_values(0)
                for row in range(len(rownames)):
                    dict_data[rownames[row]] = []
                    for col in range(1,ncols):
                        v = table.cell(row, col).value
                        if v=='':
                            break
                        else:
                            dict_data[rownames[row]].append( tsf(v) )
                        
        elif struct=='dictdict':
            colnames = table.row_values(0)[1:]
            rownames = table.col_values(0)[1:]
            if axis=='h':
                for col,colname in enumerate(colnames):
                    dict_data[colname] = dict()
                    for row,rowname in enumerate(rownames):
                        dict_data[colname][rowname] = tsf(table.cell(row+1,col+1).value)
            else:
                for row,rowname in enumerate(rownames):
                    dict_data[rowname] = dict()
                    for col,colname in enumerate(colnames):
                        dict_data[rowname][colname] = tsf(table.cell(row+1,col+1).value)
        else:
            raise(TypeError)
        if struct[:4]=='list':
            dict_all[sheet] = list_data
        else:
            dict_all[sheet] = dict_data
    return dict_all[list_sheet[0]] if len(list_sheet)==1 else dict_all

# if __name__ == '__main__':
#     dict_cmp = readexcel('../cmp.xls', 'dictlist', axis='horizontal', dtype='str')
#     for key in dict_cmp:
#         if key == 'Time':
#             dict_cmp[key] = np.array( dict_cmp[key] )
#         else:
#             dict_cmp[key] = np.array([float(v) for v in dict_cmp[key]])