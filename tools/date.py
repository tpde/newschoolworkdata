
from datetime import datetime

def cropDate( dates, trange, struct_date=None, struct_trange=None):
    # struct enum( [None, '%Y%m%d%H%M%S', '%Y-%m-%d %H:%M:%S', '%Y.%m.%d %H%M%S'] )
    t1 = datetime.strptime(trange[0], struct_trange) if struct_trange else trange[0]
    t2 = datetime.strptime(trange[1], struct_trange) if struct_trange else trange[1]
    if struct_date:
        dates = [datetime.strptime(date, struct_date) for date in dates]
        
    num = len(dates)
    idxDn = 0
    while( idxDn<num and dates[idxDn] < t1 ):
        idxDn += 1
    idxUp = idxDn
    while( idxUp<num and dates[idxUp] < t2 ):
        idxUp += 1
    return idxDn, idxUp

def get_months(tstart, tend):
    # tstart ~ tend: 20161022  ~  20190324
    list_date = [ ]
    
    months = ( int(tend[:4])*12 + int(tend[4:6]) )   -   ( int(tstart[:4])*12 + int(tstart[4:6]) )
    for m in range( months+2 ):
        count = (int(tstart[:4])*12 + int(tstart[4:6])-1) + m
        year = count//12
        month = count%12 + 1
        list_date.append( '%d.%02d.01'%(year, month) )

    return list_date

