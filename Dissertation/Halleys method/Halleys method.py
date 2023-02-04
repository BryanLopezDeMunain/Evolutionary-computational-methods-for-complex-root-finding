import numpy as np

# Test function is f(x)=sin(x), so f'(x)=cos(x), and f''(x)=-sin(x).

def f(xn):
    return(xn**3)

def der(xn):
    return(3*xn**2)

def sec_der(xn):
    return(6*xn)


xn=150
count=0
while np.abs(f(xn)) > 0.00001:
    xn = xn - ((2*f(xn)*der(xn))/((2*(der(xn)**2))-(f(xn)*sec_der(xn))))
    print(xn)
    print(f(xn))
    count += 1
    
print(count)