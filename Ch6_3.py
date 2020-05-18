import numpy as np
import cv2
from numpy.fft import fft2,ifft2



def frequency_filtering(f,filter,D0,order):
    nr,nc = f.shape[:2]

    fp = np.zeros([nr,nc])
    for x in range(nr):
        for y in range(nc):
            fp[x,y] = pow(-1,x+y)*f[x,y]

    F = fft2(fp)
    G =F.copy()

    if filter == 1:
        for u in range(nr):
            for v in range(nc):
                dist = np.sqrt((u-nr/2)*(u-nr/2)+
                    (v-nc/2)*(v-nc/2))
                if dist >D0:
                    G[u,v]=0

    elif filter == 2:
        for u in range(nr):
            for v in range(nc):
                dist = np.sqrt((u-nr/2)*(u-nr/2)+
                    (v-nc/2)*(v-nc/2))
                if dist <= D0:
                    G[u,v]=0
    
    elif filter == 3:
        for u in range(nr):
            for v in range(nc):
                dist = np.sqrt((u-nr/2)*(u-nr/2)+
                    (v-nc/2)*(v-nc/2))
                H = np.exp(-(dist*dist)/(2*D0*D0))
                G[u,v] *= H

    elif filter == 4:
        for u in range(nr):
            for v in range(nc):
                dist = np.sqrt((u-nr/2)*(u-nr/2)+
                    (v-nc/2)*(v-nc/2))
                H = 1-np.exp(-(dist*dist)/(2*D0*D0))
                G[u,v] *= H

    elif filter == 5:
        for u in range(nr):
            for v in range(nc):
                dist = np.sqrt((u-nr/2)*(u-nr/2)+
                    (v-nc/2)*(v-nc/2))
                H = 1.0/(1.0+pow(dist/D0,2*order))
                G[u,v] *= H

    elif filter == 6:
        for u in range(nr):
            for v in range(nc):
                dist = np.sqrt((u-nr/2)*(u-nr/2)+
                    (v-nc/2)*(v-nc/2))
                H = 1.0-1.0/(1.0+pow(dist/D0,2*order))
                G[u,v] *= H
    
    gp = ifft2(G)

    gp2 = np.zeros([nr,nc])
    for x in range(nr):
        for y in range(nc):
            gp2[x,y]=round(pow(-1,x+y)*np.real(gp[x,y]),0)
    g = np.uint8(np.clip(gp2,0,255))

    return g 


def main():
    print("Filering in the Frequency Domain")
    cutoff = 50
    order = 1
    img1 = cv2.imread("2165001.bmp",0)
    for filter in range(6):
        img2 = frequency_filtering(img1,filter+1,cutoff,order)
        cv2.imshow("Foltering in the Frequency Domain%d" %(filter+1) ,img2)
    cv2.imshow("Original Image",img1)
    cv2.waitKey(0)


main() 
