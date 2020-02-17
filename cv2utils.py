import numpy as np
import cv2 as cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time

def to_u(a):
    a -= np.min(a)
    a /= np.max(a)
    a *= 255
    return np.uint8(a)

def optical_flow(one, two,pure=False, pyr_scale = 0.5, levels=1, winsize = 2, iterations = 2, poly_n = 5, poly_sigma = 1.1, flags = 0, only_flow = False):
    """
    method taken from (https://chatbotslife.com/autonomous-vehicle-speed-estimation-from-dashboard-cam-ca96c24120e4)
    """
    one_g = cv2.cvtColor(one, cv2.COLOR_RGB2GRAY)
    two_g = cv2.cvtColor(two, cv2.COLOR_RGB2GRAY)
    hsv = np.zeros(np.array(one).shape) #    hsv = np.zeros((392,640, 3))

    # set saturation
    hsv[:,:,1] = cv2.cvtColor(two, cv2.COLOR_RGB2HSV)[:,:,1]
    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(one_g, two_g, flow=None,    #https://www.programcreek.com/python/example/89313/cv2.calcOpticalFlowFarneback
                                        pyr_scale=pyr_scale, levels=levels, winsize=winsize, #15 je bil winsize
                                        iterations=iterations,
                                        poly_n=poly_n, poly_sigma=poly_sigma, flags=flags)
    if only_flow:
        return flow
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hue corresponds to direction

    if not pure:
        # ang is in radians (0 to 2pi)
        hsv[:,:,0] = ang * (255/ np.pi / 2)
        # value corresponds to magnitude
        #hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        # convert HSV to int32's
        hsv = np.asarray(hsv, dtype= np.float32)
        return hsv
    else:
        return mag, ang



def generator_from_video(file  = 'C:/ultrasound_testing\ALOKA_US/26081958/20181220/26081958_20181220_MSK-_VIDEO_0003.AVI', maxframes = 2000):
    """
    :param file: string file location
    :param maxframes: int frames to array
    :return:
    """
    vidcap = cv2.VideoCapture(file)
    array=[]
    frame = 0
    success, image = vidcap.read()
    while success:
        array.append(image)
        frame+=1
        success, image = vidcap.read()
        yield np.array(image)
    #while True:
    #    yield array[np.random.randint(0,frame)]

def array_from_video(file  = 'C:/ultrasound_testing\ALOKA_US/26081958/20181220/26081958_20181220_MSK-_VIDEO_0003.AVI', maxframes = 2000):
    """
    :param file: string file location
    :param maxframes: int frames to array
    :return:
    """
    vidcap = cv2.VideoCapture(file)
    array=[]
    frame = 0
    success, image = vidcap.read()
    while success:
        array.append(image)
        frame+=1
        success, image = vidcap.read()
        #array.append(np.array(image))
    return np.array(array)

def array_to_video(array, file = "../{}.avi".format(time()), to_u_ = True):
    out = cv2.VideoWriter(file, apiPreference=0, fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=18,
                          frameSize=(array[0].shape[1], array[0].shape[0]))
    for frame in array:
        if frame.dtype != np.uint8 and to_u_:
            frame= to_u(frame)
        out.write(frame)


def optical_flow_from_array(array, pyr_scale = 0.5, levels=1, winsize = 2, iterations = 2, poly_n = 5, poly_sigma = 1.1, flags = 0):
    prev_frame = []
    optical_f = []
    for frame in array:
        if prev_frame == []:
            prev_frame = frame
            optical_f.append(np.zeros(shape=frame.shape, dtype=frame.dtype))
        else:
            of = optical_flow(prev_frame, frame,pure=False, pyr_scale=pyr_scale, levels=levels, winsize=winsize, iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma, flags=flags)
            #std = np.std(of)
            #of[of < ] = 0 #100
            #optical_f.append(to_u(of))
            optical_f.append(np.uint8(of))
            prev_frame = frame
    return np.array(optical_f)


#plt.imshow(array_from_video()[55])
#plt.show()