# @Author: Yan Tang
# @Date:   2018-06-27 

'''
----------------------------------------------------------------
Facial image pre-processing including:
face detection
landmarks detection
facial rotation correction
facial cropping, resize
geometric feature extraction

Example:
import FaceProcessUtil as fpu
flag, img=fpu.calibrateImge(image_path)
if flag:
    imgr = fpu.getLandMarkFeatures_and_ImgPatches(img)
#####
img is the pre-processed image for DGFN
imgr[0] is the pre-processed image for DFSN and DFSN-I
imgr[2] is the static geometric feature  of img
----------------------------------------------------------------
'''

import math
import cv2
import dlib
import numpy as np
from PIL import Image as IM
import time

rescaleImg = [1.4504, 1.5843, 1.4504, 1.3165]
mpoint = [63.78009, 41.66620]
target_size = 128

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./dlibmodel/shape_predictor_68_face_landmarks.dat")

#get geometric feature
def __getLandmarkFeatures(X, Y):###
    w,h = getImgWH(X,Y)

    mou_geo_fea = getMouthGeometry(X,Y,w,h)

    eyebrow_geo_fea = getEyeBrowGeometry(X,Y,w,h)

    eye_geo_fea = getEyeGeometry(X,Y,w,h)

    nose_geo_fea = getNoseGeometry(X,Y,w,h)

    jaw_geo_fea = getJawGeometry(X,Y,w,h)

    gobal_d_geo_fea = getGobalDistanceGeometry(X,Y,w,h)

    gobal_a_geo_fea = getGobalAreaGeometry(X,Y,w,h)

    local_features = mou_geo_fea+eyebrow_geo_fea+eye_geo_fea+nose_geo_fea+jaw_geo_fea
    
    gobal_features = gobal_d_geo_fea+getGobalAreaGeometry(X,Y,w,h)

    features = local_features+gobal_features

    return features

def __UnifiedOutputs(rescaleImg, Geof, Geo_features, Patchf, eyepatch, foreheadpatch, mouthpatch, innerface):
    ''''add additional operations'''

    return rescaleImg, Geof, Geo_features, Patchf, eyepatch, foreheadpatch, mouthpatch, innerface


def __genLMFandIP(img, w, h, LM, Patches, regular=False, X=None, Y=None):
    """Return the Geometry features from landmarks and Images patches.
    If regularize is set to False, it will always return cosine True and three images for the patches operation.
    Otherwise, it could return cosine False and four None values
    X and Y are ndarray"""
    if LM or Patches:
        if X is None or Y is None:
            #pl.write(' 0\n')
            print(">>>***%%%Warning [__genLMFandIP()]: No face was detected in the image.")
            if not LM:
                if regular:
                    print(">>>***%%%Warning [__genLMFandIP()]: Processing the default config on the image")
            
                    eye_patch = __getEyePatch(img, w, h)
                    forehead_patch = __getMiddlePatch(img, w, h)
                    mouth_patch = __getMouthPatch(img, w, h)
                    inner_face = __getInnerFace(img, w, h)

                    return __UnifiedOutputs(img, False, None, True, eye_patch, forehead_patch, mouth_patch, inner_face)
                else:
                    print(">>>***%%%Warning [__genLMFandIP()]: Return img, False, None, False, None, None, None, None")
                    return  img, False, None, False, None, None, None, None
            elif not Patches:
                return __UnifiedOutputs(img, False, None, False, None, None, None, None)
            else:
                eye_patch = __getEyePatch(img, w, h)
                forehead_patch = __getMiddlePatch(img, w, h)
                mouth_patch = __getMouthPatch(img, w, h)
                inner_face = __getInnerFace(img, w, h)

                return __UnifiedOutputs(img, False, None, True, eye_patch, forehead_patch, mouth_patch, inner_face)
            
        if not LM:
            eye_patch = __getEyePatch(img, w, h, X, Y)
            forehead_patch = __getMiddlePatch(img, w, h, X, Y)
            mouth_patch = __getMouthPatch(img, w, h, X, Y)
            inner_face = __getInnerFace(img, w, h)

            return __UnifiedOutputs(img, False, None, True, eye_patch, forehead_patch, mouth_patch, inner_face)
        elif not Patches:
            landmark_features= __getLandmarkFeatures(X, Y)

            return __UnifiedOutputs(img, True, landmark_features, False, None, None, None, None)
        else:
            eye_patch = __getEyePatch(img, w, h, X, Y)
            forehead_patch = __getMiddlePatch(img, w, h, X, Y)
            mouth_patch = __getMouthPatch(img, w, h, X, Y)
            landmark_features= __getLandmarkFeatures(X, Y)
            inner_face = __getInnerFace(img, w, h)

            return __UnifiedOutputs(img, True, landmark_features, True, eye_patch, forehead_patch, mouth_patch, inner_face)
    else:
        return __UnifiedOutputs(img, False, None, False, None, None, None, None)

def __cropImg(img, shape=None, LM=False, Patches=False, regularize=False, trg_size=target_size, rescale=rescaleImg):
    """Rescale, adjust, and crop the images.
    If shape is None, it will recale the img without croping and return __genLMFandIP"""
    if not shape==None:
        nLM = shape.num_parts
        lms_x = np.asarray([shape.part(i).x for i in range(0,nLM)])
        lms_y = np.asarray([shape.part(i).y for i in range(0,nLM)])

        tlx = float(min(lms_x))#top left x
        tly = float (min(lms_y))#top left y
        ww = float (max(lms_x) - tlx)
        hh = float(max(lms_y) - tly)
        # Approximate LM tight BB
        h = img.shape[0]
        w = img.shape[1]
        cx = tlx + ww/2
        cy = tly + hh/2
        #tsize = max(ww,hh)/2
        tsize = ww/2

        # Approximate expanded bounding box
        btlx = int(round(cx - rescale[0]*tsize))
        btly = int(round(cy - rescale[1]*tsize))
        bbrx = int(round(cx + rescale[2]*tsize))
        bbry = int(round(cy + rescale[3]*tsize))
        nw = int(bbrx-btlx)
        nh = int(bbry-btly)

        #adjust relative location
        x0=(np.mean(lms_x[36:42])+np.mean(lms_x[42:48]))/2
        y0=(np.mean(lms_y[36:42])+np.mean(lms_y[42:48]))/2
        Mpx=int(round((mpoint[0]*nw/float(target_size))-x0+btlx))
        Mpy=int(round((mpoint[1]*nh/float(target_size))-y0+btly))
        btlx=btlx-Mpx
        bbrx=bbrx-Mpx
        bbry=bbry-Mpy
        btly=btly-Mpy
        print('coordinate adjustment')
        print(Mpx, Mpy)
        Xa=np.round((lms_x-btlx)*trg_size/nw)
        Ya=np.round((lms_y-btly)*trg_size/nh)
        
        #few=open(eyelog,'a')
        #few.write('%lf %lf\n'%((np.mean(Xa[36:42])+np.mean(Xa[42:48]))/2,(np.mean(Ya[36:42])+np.mean(Ya[42:48]))/2))
        #few.close()

        imcrop = np.zeros((nh,nw), dtype = "uint8")

        blxstart = 0
        if btlx < 0:
            blxstart = -btlx
            btlx = 0
        brxend = nw
        if bbrx > w:
            brxend = w+nw - bbrx#brxend=nw-(bbrx-w)
            bbrx = w
        btystart = 0
        if btly < 0:
            btystart = -btly
            btly = 0
        bbyend = nh
        if bbry > h:
            bbyend = h+nh - bbry#bbyend=nh-(bbry-h)
            bbry = h
        imcrop[btystart:bbyend, blxstart:brxend] = img[btly:bbry, btlx:bbrx]
        im_rescale=cv2.resize(imcrop,(trg_size, trg_size))
        im_rescale = crop_face_only(img,shape)
        return __genLMFandIP(im_rescale, trg_size, trg_size, LM, Patches, regular=regularize, X=Xa, Y=Ya)
    else:
        im_rescale=cv2.resize(img, (trg_size, trg_size))
        return __genLMFandIP(im_rescale, trg_size, trg_size, LM, Patches, False)

def getLandMarkFeatures_and_ImgPatches(img):
    withLM=True
    withPatches=False
    fromfacedataset=True
    if len(img.shape) == 3 and img.shape[2]==3:
        g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g_img = img
    td1= time.time()
    #f_ds=detector(g_img, 1)#1 represents upsample the image 1 times for detection
    f_ds=detector(g_img, 0)
    td2 = time.time()
    print('Time in detecting face: %fs'%(td2-td1))
    if len(f_ds) == 0:
        #pl.write('0')
        f_ds = detector(g_img, 1)
        if len(f_ds) == 0:
            print(">>>***%%%Warning [getLandMarkFeatures_and_ImgPatches()]: No face was detected from the image")
        if not fromfacedataset:
            print(">>>***%%%Warning [getLandMarkFeatures_and_ImgPatches()]: No face was detected, and return False and None values")

            return None, False, None, False, None, None, None, None
        else:
            print(">>>***%%%Warning [getLandMarkFeatures_and_ImgPatches()]: Processing the default config on the image")
            return __cropImg(g_img, LM=withLM, Patches=withPatches, regularize=fromfacedataset)
    elif len(f_ds) > 1:
        print(">>>***%%%Warning [getLandMarkFeatures_and_ImgPatches()]: Only process the first face detected.")
    f_shape = predictor(g_img, f_ds[0])
    #pl.write('1')
    return __cropImg(g_img, shape=f_shape, LM=withLM, Patches=withPatches, regularize=fromfacedataset)

######
#
#The followings are for calibrate the image
def __RotateTranslate(image, angle, center =None, new_center =None, resample=IM.BICUBIC):
    '''Rotate the image according to the angle'''
    if center is None:  
        return image.rotate(angle=angle, resample=resample)  
    nx,ny = x,y = center  
    if new_center:  
        (nx,ny) = new_center  
    cosine = math.cos(angle)  
    sine = math.sin(angle)  
    c = x-nx*cosine-ny*sine  
    d =-sine  
    e = cosine
    f = y-nx*d-ny*e  
    return image.transform(image.size, IM.AFFINE, (cosine,sine,c,d,e,f), resample=resample)
def __RotaFace(image, eye_left=(0,0), eye_right=(0,0)):
    '''Rotate the face according to the eyes'''
    # get the direction from two eyes
    eye_direction = (eye_right[0]- eye_left[0], eye_right[1]- eye_left[1])
    # calc rotation angle in radians
    rotation =-math.atan2(float(eye_direction[1]),float(eye_direction[0]))
    # rotate original around the left eye  
    image = __RotateTranslate(image, center=eye_left, angle=rotation)
    return image
def __shape_to_np(shape):
    '''Transform the shape points into numpy array of 68*2'''
    nLM = shape.num_parts
    x = np.asarray([shape.part(i).x for i in range(0,nLM)])
    y = np.asarray([shape.part(i).y for i in range(0,nLM)])
    return x,y
def calibrateImge(imgpath):
    '''Calibrate the image of the face'''
    imgcv_gray=cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    if imgcv_gray is None:
        print('Unexpected ERROR: The value read from the imagepath is None. No image was loaded')
        exit(-1)
    dets = detector(imgcv_gray,1)
    if len(dets)==0:
        dets = detector(imgcv_gray,0)
        if len(dets)==0:
               print("No face was detected^^^^^^^^^^^^^^")
               return False, imgcv_gray
    lmarks=[]
    for id, det in enumerate(dets):
        if id > 0:
            print("ONLY process the first face>>>>>>>>>")
            break
        shape = predictor(imgcv_gray, det)
        x, y = __shape_to_np(shape)
    lmarks = np.asarray(lmarks, dtype='float32')
    pilimg=IM.fromarray(imgcv_gray)
    rtimg=__RotaFace(pilimg, eye_left=(np.mean(x[36:42]),np.mean(y[36:42])),
                                           eye_right=(np.mean(x[42:48]),np.mean(y[42:48])))
    imgcv_gray=np.array(rtimg)
    return True, imgcv_gray
def crop_face_only(img, shape=None, trg_size=128):
    if not shape==None:
        nLM = shape.num_parts
        lms_x = np.asarray([shape.part(i).x for i in range(0,nLM)])
        lms_y = np.asarray([shape.part(i).y for i in range(0,nLM)])

        tlx = float(min(lms_x[17:67]))#top left x
        tly = float (min(lms_y[17:67]))#top left y
        brx = float(max(lms_x[17:67]))
        bry = float((lms_y[57]+lms_y[8])/2)
        ww = float(brx-tlx)
        hh = float(bry-tly)
        
        tlx = round(tlx)
        tly = round(tly)
        brx = round(brx)
        bry = round(bry)
        
        imcrop = img[tly:bry, tlx:brx]
        im_rescale=cv2.resize(imcrop,(trg_size, trg_size))
        return im_rescale

    else:
        im_rescale=cv2.resize(img, (trg_size, trg_size))
        return im_rescale

####static geometry geatures---------------------- 
def getImgWH(X,Y):
    return abs(X[16]-X[0]),abs(Y[24]-Y[8])

def getDistance(X,Y,w,h,a,b):
    dx = (X[a]-X[b])/w
    dy = (Y[a]-Y[b])/h
    d = math.sqrt(dx**2+dy**2)
    return d

def getTriangleArea(X,Y,w,h,x1,x2,x3):
    a = getDistance(X,Y,w,h,x1,x2)
    b = getDistance(X,Y,w,h,x1,x3)
    c = getDistance(X,Y,w,h,x2,x3)
    p = (a+b+c)/2
    H = p*(p-a)*(p-b)*(p-c)
    if H<0:
        H=0
    s = math.sqrt(H)
    return s

def getMouthGeometry(X,Y,w,h):
    mouth_geo_feature = []
    #open mouth area
    width=math.sqrt(((X[64]-X[60])/w)**2+((Y[64]-Y[60])/h)**2)
    height = math.sqrt(((X[62]-X[66])/w)**2+((Y[62]-Y[66])/h)**2)
    open_mouth_square = width*height*math.pi/4

    mouth_geo_feature.append(open_mouth_square)

    #lower lip curvature
    x1 = X[48]
    y1 = Y[48]
    x2 = X[54]
    y2 = Y[54]
    x3 = X[66]
    y3 = Y[66]
 
    A = np.mat([[x1*x1,x1,1],[x2*x2,x2,1],[x3*x3,x3,1]])
    b = np.mat([[y1],[y2],[y3]])
    a = A.I*b
    a = a[0]
    a = float(a)

    mouth_geo_feature.append(a)

     #upper lip curvature
    x1 = X[48]
    y1 = Y[48]
    x2 = X[54]
    y2 = Y[54]
    x3 = X[62]
    y3 = Y[62]
 
    A = np.mat([[x1*x1,x1,1],[x2*x2,x2,1],[x3*x3,x3,1]])
    b = np.mat([[y1],[y2],[y3]])
    a1 = A.I*b
    a1 = a1[0]
    a1 = float(a1)

    mouth_geo_feature.append(a1)

    #outer mouth width and height
    ow = math.sqrt(((X[54]-X[48])/w)**2+((Y[54]-Y[48])/h)**2)
    oh = math.sqrt(((X[57]-X[51])/w)**2+((Y[57]-Y[51])/h)**2)

    mouth_geo_feature.append(ow)
    

    return mouth_geo_feature#4 dimension

def getEyeBrowGeometry(X,Y,w,h):
    eye_geo_feature=[]
    eyecenterXL = (X[36]+X[37]+X[38]+X[39]+X[40]+X[41])/6
    eyecenterYL = (Y[36]+Y[37]+Y[38]+Y[39]+Y[40]+Y[41])/6
    eyecenterXR = (X[42]+X[43]+X[44]+X[45]+X[46]+X[47])/6
    eyecenterYR = (Y[42]+Y[43]+Y[44]+Y[45]+Y[46]+Y[47])/6
    
    #eyebrow distance
    lebd1 = math.sqrt(((X[17]-eyecenterXL)/w)**2+((Y[17]-eyecenterYL)/h)**2)
    lebd2 = math.sqrt(((X[19]-eyecenterXL)/w)**2+((Y[19]-eyecenterYL)/h)**2)
    lebd3 = math.sqrt(((X[21]-eyecenterXL)/w)**2+((Y[21]-eyecenterYL)/h)**2)

    rebd1 = math.sqrt(((X[22]-eyecenterXR)/w)**2+((Y[22]-eyecenterYR)/h)**2)
    rebd2 = math.sqrt(((X[24]-eyecenterXR)/w)**2+((Y[24]-eyecenterYR)/h)**2)
    rebd3 = math.sqrt(((X[26]-eyecenterXR)/w)**2+((Y[26]-eyecenterYR)/h)**2)

    eye_geo_feature.append(lebd1)
    eye_geo_feature.append(lebd2)
    eye_geo_feature.append(lebd3)
    eye_geo_feature.append(rebd1)
    eye_geo_feature.append(rebd2)
    eye_geo_feature.append(rebd3)
    
    return eye_geo_feature#6 dimension

def getEyeGeometry(X,Y,w,h):
    eye_geo_feature=[]
    
    #eye distance-----------
    leldX = (X[37]-X[41])/w
    leldY = (Y[37]-Y[41])/h
    leld = math.sqrt((leldX**2+leldY**2))
    eye_geo_feature.append(leld)

    lerdX = (X[38]-X[40])/w
    lerdY = (Y[38]-Y[40])/h
    lerd = math.sqrt((lerdX**2+lerdY**2))
    eye_geo_feature.append(lerd)

    reldX = (X[43]-X[47])/w
    reldY = (Y[43]-Y[47])/h
    reld = math.sqrt((reldX**2+reldY**2))
    eye_geo_feature.append(reld)

    rerdX = (X[44]-X[46])/w
    rerdY = (Y[44]-Y[46])/h
    rerd = math.sqrt((rerdX**2+rerdY**2))
    eye_geo_feature.append(rerd)
    #eye distance end------------

    #eye area square
    #left eye
    alx = (X[36]-X[39])/w
    aly = (Y[36]-Y[39])/h
    al = math.sqrt(alx**2+aly**2)
    bl = (leld+lerd)/2
    areal = al*bl*math.pi
    eye_geo_feature.append(areal)

    #right eye
    arx = (X[42]-X[45])/w
    ary = (Y[42]-Y[45])/h
    ar = math.sqrt(arx**2+ary**2)
    br = (reld+rerd)/2
    arear = ar*br*math.pi
    eye_geo_feature.append(arear)
    
     #eye area square----end

    return eye_geo_feature#6 dimension
     
def getNoseGeometry(X,Y,w,h):
    nose_geo_fea=[]
    #nose edge curvature
    x1 = X[31]
    y1 = Y[31]
    x2 = X[33]
    y2 = Y[33]
    x3 = X[35]
    y3 = Y[35]
 
    A = np.mat([[x1*x1,x1,1],[x2*x2,x2,1],[x3*x3,x3,1]])
    b = np.mat([[y1],[y2],[y3]])
    a = A.I*b
    a = a[0]
    a = float(a)

    nose_geo_fea.append(a)

    x1 = (X[31]-X[33])/w
    y1 = (Y[31]-Y[33])/h
    ld = math.sqrt(x1**2+y1**2)
    x2 = (X[35]-X[33])/w
    y2 = (Y[35]-Y[33])/h
    rd = math.sqrt(x2**2+y2**2)
    td = ld+rd
    nose_geo_fea.append(td)

    x3 = (X[27]-X[30])/w
    y3 = (Y[27]-Y[30])/h
    nh = math.sqrt(x3**2+y3**2)
    nose_geo_fea.append(nh)

    d = getDistance(X,Y,w,h,29,27)
    nose_geo_fea.append(d)
    nose_geo_fea.append(d) #double

    return nose_geo_fea#5 dimension

def getJawGeometry(X,Y,w,h):
    jaw_geo_fea=[]
    #jaw curvature 1
    x1 = X[5]
    y1 = Y[5]
    x2 = X[8]
    y2 = Y[8]
    x3 = X[11]
    y3 = Y[11]
 
    A = np.mat([[x1*x1,x1,1],[x2*x2,x2,1],[x3*x3,x3,1]])
    b = np.mat([[y1],[y2],[y3]])
    a1 = A.I*b
    a1 = a1[0]
    a1 = float(a1)
    jaw_geo_fea.append(a1)
    #jaw curvature 2
    x1 = X[4]
    y1 = Y[4]
    x2 = X[8]
    y2 = Y[8]
    x3 = X[12]
    y3 = Y[12]
 
    A = np.mat([[x1*x1,x1,1],[x2*x2,x2,1],[x3*x3,x3,1]])
    b = np.mat([[y1],[y2],[y3]])
    a2 = A.I*b
    a2 = a2[0]
    a2 = float(a2)
    jaw_geo_fea.append(a2)

    #jaw width
    x1 = ((X[2]+X[3])/2-(X[13]+X[14])/2)/w
    y1 = ((Y[2]+Y[3])/2-(Y[13]+Y[14])/2)/h
    width = math.sqrt(x1**2+y1**2)
    jaw_geo_fea.append(width)

    d17 = getDistance(X,Y,w,h,3,13)
    jaw_geo_fea.append(d17)

    return jaw_geo_fea#4 dimension

def getGobalDistanceGeometry(X,Y,w,h):
    gobal_geo_fea = []
    mouthcx = (X[60]+X[62]+X[64]+X[66])/4
    mouthcy = (Y[60]+Y[62]+Y[64]+Y[66])/4

    nosecx = (X[28]+X[29]+X[30])/3
    nosecy = (Y[28]+Y[29]+Y[30])/3

    d1x = (X[19]-nosecx)/w
    d1y = (Y[19]-nosecx)/h
    d1 = math.sqrt(d1x**2+d1y**2)
    gobal_geo_fea.append(d1)

    d2x = (X[24]-nosecx)/w
    d2y = (Y[24]-nosecx)/h
    d2 = math.sqrt(d2x**2+d2y**2)
    gobal_geo_fea.append(d2)

    d3x = (mouthcx-nosecx)/w
    d3y = (mouthcy-nosecx)/h
    d3 = math.sqrt(d3x**2+d3y**2)
    gobal_geo_fea.append(d3)

    eyecenterXL = (X[36]+X[37]+X[38]+X[39]+X[40]+X[41])/6
    eyecenterYL = (Y[36]+Y[37]+Y[38]+Y[39]+Y[40]+Y[41])/6
    eyecenterXR = (X[42]+X[43]+X[44]+X[45]+X[46]+X[47])/6
    eyecenterYR = (Y[42]+Y[43]+Y[44]+Y[45]+Y[46]+Y[47])/6

    d4x = (eyecenterXL-X[31])/w
    d4y = (eyecenterYL-Y[31])/h
    d4 = math.sqrt(d4x**2+d4y**2)
    gobal_geo_fea.append(d4)

    d5x = (eyecenterXR-X[35])/w
    d5y = (eyecenterYR-Y[35])/h
    d5 = math.sqrt(d5x**2+d5y**2)
    gobal_geo_fea.append(d5)

    d8x = (X[48]-eyecenterXL)/w
    d8y = (Y[48]-eyecenterYL)/h
    d8 = math.sqrt(d8x**2+d8y**2)
    gobal_geo_fea.append(d8)

    d9x = (X[54]-eyecenterXR)/w
    d9y = (Y[54]-eyecenterYR)/h
    d9 = math.sqrt(d9x**2+d9y**2)
    gobal_geo_fea.append(d9)

    d12x = (X[8]-X[57])/w
    d12y = (Y[8]-Y[57])/h
    d12 = math.sqrt(d12x**2+d12y**2)
    gobal_geo_fea.append(d12)

    d13x = (X[21]-X[27])/w
    d13y = (Y[21]-Y[27])/h
    d13 = math.sqrt(d13x**2+d13y**2)
    gobal_geo_fea.append(d13)

    d14x = (X[22]-X[27])/w
    d14y = (Y[22]-Y[27])/h
    d14 = math.sqrt(d14x**2+d14y**2)
    gobal_geo_fea.append(d14)

    d15x = (X[33]-X[51])/w
    d15y = (Y[33]-Y[51])/h
    d15 = math.sqrt(d15x**2+d15y**2)
    gobal_geo_fea.append(d15)

    d16 = getDistance(X,Y,w,h,21,22)
    gobal_geo_fea.append(d16)

    return gobal_geo_fea#12dimension

def getGobalAreaGeometry(X,Y,w,h):
    area_geo_fea = []
    s = getTriangleArea(X,Y,w,h,6,10,8)
    area_geo_fea.append(s)
    s = getTriangleArea(X,Y,w,h,19,24,66)
    area_geo_fea.append(s)
    return area_geo_fea#2dimension

####static geometry geatures---------------------- end
