from math import pi, acos, sin, cos

def huduzhi(x):
    return x * pi/180

def jiaoduzhi(x):
    return x * 180/pi

def relative(PA, PB, angle=0):  # p1经纬度，p2经纬度
    Ax, Ay = PA
    Bx, By = PB
    x = Bx - Ax
    y = By - Ay
    d = (x ** 2 + y ** 2) ** 0.5
    fw = jiaoduzhi( acos(y/d) ) if d>0 else 0
#     if Bx < Ax-1E-7:
#         fw = - fw
    assert(-180<fw<=180)
    
    assert(-180<angle<=180)
    fw = fw - angle
    if fw<=-180:
        fw = 360+fw
    elif fw>180:
        fw = fw-360
    assert(-180<fw<=180)
    return fw, d