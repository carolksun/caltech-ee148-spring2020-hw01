import numpy as np
from PIL import Image
from scipy.signal import find_peaks

width = 29
height = 66

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_ex():
    bb = [25, 320, 91, 349]
    path = 'data/RedLights2011_Medium/RL-010.jpg'

    I = Image.open(path)
    I = np.asarray(I)

    train = I[bb[0]:bb[2], bb[1]: bb[3], :]
    return train

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    train = get_ex()
    fl = train.flatten()
    normed_fl = fl / np.linalg.norm(fl)
    ls = np.lib.stride_tricks.sliding_window_view(I, train.shape)
    inners = []

    for i in range(len(ls)):
        for j in range(len(ls[i])):
            arr = ls[i][j][0]
            arr = arr.flatten()
            arr = arr / np.linalg.norm(arr)
            inners.append((np.inner(arr, normed_fl), [j, i, j + width, i + height]))
            
    inprod = [i[0] for i in inners]
    smoothed = smooth(inprod, 7)
    peaks, _ = find_peaks(smoothed, height=0.88)
    bounding_tl = [inners[p][1] for p in peaks]
    return bounding_tl
