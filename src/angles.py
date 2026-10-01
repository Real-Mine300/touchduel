import math
def angle(a, b, c):
    u = (a[0] - b[0], a[1] - b[1])
    v = (c[0] - b[0], c[1] - b[1])
    return math.degrees(math.acos((u[0]*v[0]+u[1]*v[1])/(math.sqrt(u[0]**2 + u[1]**2)*math.sqrt(v[0]**2 + v[1]**2))))

print(angle((1,0), (0,0), (0,1)))