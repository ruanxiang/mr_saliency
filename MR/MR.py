####################################################################
## Author:
##       Xiang Ruan
##       httpr://ruanxiang.net
##       ruanxiang@gmail.com
## License:
##       GPL 2.0
##       NOTE: the algorithm itself is patented by OMRON, co, Japan
##             my employer, so please do not use the algorithm in
##             any commerical product
## Version:
##       1.0
##
## ----------------------------------------------------------------
## A python implementation of manifold ranking saliency
## Usage:
##      import MR
##      mr = MR.MR_saliency()
##      sal = mr.saliency(img)
##
## Check paper.pdf for algorithm details 
## I leave all th parameters open to maniplating, however, you don't
## have to do it, default values work pretty well, unless you really
## know what you want to do to modify the parameters


import scipy as sp
import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import camera
from scipy.linalg import inv
import matplotlib.pyplot as plt

cv_ver = int(cv2.__version__.split('.')[0])
_cv2_LOAD_IMAGE_COLOR = cv2.IMREAD_COLOR if cv_ver >= 3 else cv2.CV_LOAD_IMAGE_COLOR

class MR_saliency(object):
    """Python implementation of manifold ranking saliency"""
    weight_parameters = {'alpha':0.99,
                         'delta':0.1}
    superpixel_parameters = {'segs':200,
                             'compactness':10,
                             'max_iter':10,
                             'sigma':1,
                             'spacing':None,
                             'multichannel':True,
                             'convert2lab':None,
                             'enforce_connectivity':False,
                             'min_size_factor':0.5,
                             'max_size_factor':3,
                             'slic_zero':False}
    binary_thre = None

    def __init__(self, alpha = 0.99, delta = 0.1,
                 segs = 200, compactness = 10,
                 max_iter = 10, sigma = 1,
                 spacing = None, multichannel = True,
                 convert2lab = None, enforce_connectivity = False,
                 min_size_factor = 0.5, max_size_factor = 3,
                 slic_zero = False):
        self.weight_parameters['alpha'] = alpha
        self.weight_parameters['delta'] = delta
        self.superpixel_parameters['segs'] = segs
        self.superpixel_parameters['compactness'] = compactness
        self.superpixel_parameters['max_iter'] = max_iter
        self.superpixel_parameters['sigma'] = sigma
        self.superpixel_parameters['spacing'] = spacing
        self.superpixel_parameters['multichannel'] = multichannel
        self.superpixel_parameters['convert2lab'] = convert2lab
        self.superpixel_parameters['enforce_connectivity'] = enforce_connectivity
        self.superpixel_parameters['min_size_factor'] = min_size_factor
        self.superpixel_parameters['max_size_factor'] = max_size_factor
        self.superpixel_parameters['slic_zero'] = slic_zero

    def saliency(self,img):
        # read image
        img = self.__MR_readimg(img)
        # superpixel
        labels = self.__MR_superpixel(img)
        # affinity matrix
        aff = self.__MR_affinity_matrix(img,labels)
        # first round
        first_sal = self.__MR_first_stage_saliency(aff,labels)
        # second round
        fin_sal = self.__MR_final_saliency(first_sal, labels,aff)
        return self.__MR_fill_superpixel_with_saliency(labels,fin_sal)

    
    def __MR_superpixel(self,img):
        return slic(img,self.superpixel_parameters['segs'],
                    self.superpixel_parameters['compactness'],
                    self.superpixel_parameters['max_iter'],
                    self.superpixel_parameters['sigma'],
                    self.superpixel_parameters['spacing'],
                    self.superpixel_parameters['multichannel'],
                    self.superpixel_parameters['convert2lab'],
                    self.superpixel_parameters['enforce_connectivity'],
                    self.superpixel_parameters['min_size_factor'],
                    self.superpixel_parameters['max_size_factor'],
                    self.superpixel_parameters['slic_zero'])

    def __MR_superpixel_mean_vector(self,img,labels):
        s = sp.amax(labels)+1
        vec = sp.zeros((s,3)).astype(float)
        for i in range(s):
            mask = labels == i
            super_v = img[mask].astype(float)
            mean = sp.mean(super_v,0)
            vec[i] = mean
        return vec

    def __MR_affinity_matrix(self,img,labels):        
        W,D = self.__MR_W_D_matrix(img,labels)
        aff = inv(D-self.weight_parameters['alpha']*W)
        aff[sp.eye(sp.amax(labels)+1).astype(bool)] = 0.0 # diagonal elements to 0
        return aff

    def __MR_saliency(self,aff,indictor):
        return sp.dot(aff,indictor)

    def __MR_W_D_matrix(self,img,labels):
        s = sp.amax(labels)+1
        vect = self.__MR_superpixel_mean_vector(img,labels)
        
        adj = self.__MR_get_adj_loop(labels)
        
        W = sp.spatial.distance.squareform(sp.spatial.distance.pdist(vect))
        
        W = sp.exp(-1*W / self.weight_parameters['delta'])
        W[adj.astype(np.bool)] = 0
        

        D = sp.zeros((s,s)).astype(float)
        for i in range(s):
            D[i, i] = sp.sum(W[i])

        return W,D

    def __MR_boundary_indictor(self,labels):
        s = sp.amax(labels)+1
        up_indictor = (sp.zeros((s,1))).astype(float)
        right_indictor = (sp.zeros((s,1))).astype(float)
        low_indictor = (sp.zeros((s,1))).astype(float)
        left_indictor = (sp.zeros((s,1))).astype(float)
    
        upper_ids = sp.unique(labels[0,:]).astype(int)
        right_ids = sp.unique(labels[:,labels.shape[1]-1]).astype(int)
        low_ids = sp.unique(labels[labels.shape[0]-1,:]).astype(int)
        left_ids = sp.unique(labels[:,0]).astype(int)

        up_indictor[upper_ids] = 1.0
        right_indictor[right_ids] = 1.0
        low_indictor[low_ids] = 1.0
        left_indictor[left_ids] = 1.0

        return up_indictor,right_indictor,low_indictor,left_indictor

    def __MR_second_stage_indictor(self,saliency_img_mask,labels):
        s = sp.amax(labels)+1
        # get ids from labels image
        ids = sp.unique(labels[saliency_img_mask]).astype(int)
        # indictor
        indictor = sp.zeros((s,1)).astype(float)
        indictor[ids] = 1.0
        return indictor

    def __MR_get_adj_loop(self, labels):
        s = sp.amax(labels) + 1
        adj = np.ones((s, s), np.bool)

        for i in range(labels.shape[0] - 1):
            for j in range(labels.shape[1] - 1):
                if labels[i, j] != labels[i+1, j]:
                    adj[labels[i, j],       labels[i+1, j]]              = False
                    adj[labels[i+1, j],   labels[i, j]]                  = False
                if labels[i, j] != labels[i, j + 1]:
                    adj[labels[i, j],       labels[i, j+1]]              = False
                    adj[labels[i, j+1],   labels[i, j]]                  = False
                if labels[i, j] != labels[i + 1, j + 1]:
                    adj[labels[i, j]        ,  labels[i+1, j+1]]       = False
                    adj[labels[i+1, j+1],  labels[i, j]]               = False
                if labels[i + 1, j] != labels[i, j + 1]:
                    adj[labels[i+1, j],   labels[i, j+1]]              = False
                    adj[labels[i, j+1],   labels[i+1, j]]              = False
        
        upper_ids = sp.unique(labels[0,:]).astype(int)
        right_ids = sp.unique(labels[:,labels.shape[1]-1]).astype(int)
        low_ids = sp.unique(labels[labels.shape[0]-1,:]).astype(int)
        left_ids = sp.unique(labels[:,0]).astype(int)
        
        bd = np.append(upper_ids, right_ids)
        bd = np.append(bd, low_ids)
        bd = sp.unique(np.append(bd, left_ids))
        
        for i in range(len(bd)):
            for j in range(i + 1, len(bd)):
                adj[bd[i], bd[j]] = False
                adj[bd[j], bd[i]] = False

        return adj
        
    def __MR_fill_superpixel_with_saliency(self,labels,saliency_score):
        sa_img = labels.copy().astype(float)
        for i in range(sp.amax(labels)+1):
            mask = labels == i
            sa_img[mask] = saliency_score[i]
        return cv2.normalize(sa_img,None,0,255,cv2.NORM_MINMAX)

    def __MR_first_stage_saliency(self,aff,labels):
        up,right,low,left = self.__MR_boundary_indictor(labels)
        up_sal = 1- self.__MR_saliency(aff,up)
        up_img = self.__MR_fill_superpixel_with_saliency(labels,up_sal)
    
        right_sal = 1-self.__MR_saliency(aff,right)
        right_img = self.__MR_fill_superpixel_with_saliency(labels,right_sal)

        low_sal = 1-self.__MR_saliency(aff,low)
        low_img = self.__MR_fill_superpixel_with_saliency(labels,low_sal)
    
        left_sal = 1-self.__MR_saliency(aff,left)
        left_img = self.__MR_fill_superpixel_with_saliency(labels,left_sal)

        return 1- up_img*right_img*low_img*left_img


    def __MR_final_saliency(self,integrated_sal, labels, aff):
        # get binary image
        if self.binary_thre == None:
            thre = sp.median(integrated_sal.astype(float))

        mask = integrated_sal > thre
        # get indicator
        ind = self.__MR_second_stage_indictor(mask,labels)
    
        return self.__MR_saliency(aff,ind)

    # read image
    def __MR_readimg(self,img):
        if isinstance(img,str): # a image path
            img = cv2.imread(img, _cv2_LOAD_IMAGE_COLOR)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(float)/255
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB).astype(float)/255
        img = cv2.cvtColor(img,cv2.COLOR_RGB2LAB).astype(float)/255
        h = 100
        w = int(float(h)/float(img.shape[0])*float(img.shape[1]))
        return cv2.resize(img,(w,h))


class MR_debuger(MR_saliency):
    def MR_showsuperpixel(self,img=None):
        if img == None:
            img = cv2.cvtColor(camera(),cv2.COLOR_RGB2BGR)
        img = self._MR_saliency__MR_readimg(img)
        labels = self._MR_saliency__MR_superpixel(img)

        plt.axis('off')
        plt.imshow(mark_boundaries(img,labels))
        plt.show()

    def MR_boudnary_extraction(self,img=None):
        if img == None:
            img = cv2.cvtColor(camera(),cv2.COLOR_RGB2BGR)
        lab_img = self._MR_saliency__MR_readimg(img)
        mark_color = (1,0,0)
        labels = self._MR_saliency__MR_superpixel(lab_img)

        up_img = lab_img.copy()
        up_ids = sp.unique(labels[0,:]).astype(int)
        up_mask = sp.zeros(labels.shape).astype(bool)
        for i in up_ids:
            up_mask = sp.logical_or(up_mask,labels==i)
        up_img[up_mask] = mark_color
        up_img = mark_boundaries(up_img,labels)

        right_img = lab_img.copy()
        right_ids = sp.unique(labels[:,labels.shape[1]-1]).astype(int)
        right_mask = sp.zeros(labels.shape).astype(bool)
        for i in right_ids:
            right_mask = sp.logical_or(right_mask,labels==i)
        right_img[right_mask] = mark_color
        right_img = mark_boundaries(right_img,labels)


        low_img = lab_img.copy()
        low_ids = sp.unique(labels[labels.shape[0]-1,:]).astype(int)
        low_mask = sp.zeros(labels.shape).astype(bool)
        for i in low_ids:
            low_mask = sp.logical_or(low_mask,labels==i)
        low_img[low_mask] = mark_color
        low_img = mark_boundaries(low_img,labels)
        
        left_img = lab_img.copy()
        left_ids = sp.unique(labels[:,0]).astype(int)
        left_mask = sp.zeros(labels.shape).astype(bool)
        for i in left_ids:
            left_mask = sp.logical_or(left_mask,labels==i)
        left_img[left_mask] = mark_color
        left_img = mark_boundaries(left_img,labels)

        plt.subplot(2,2,1)
        plt.axis('off')
        plt.title('up')
        plt.imshow(up_img)

        plt.subplot(2,2,2)
        plt.axis('off')
        plt.title('bottom')
        plt.imshow(low_img)


        plt.subplot(2,2,3)
        plt.axis('off')
        plt.title('left')
        plt.imshow(left_img)

        plt.subplot(2,2,4)
        plt.axis('off')
        plt.title('right')
        plt.imshow(right_img)

        plt.show()


    def MR_boundary_saliency(self,img=None):
        if img == None:
            img = cv2.cvtColor(camera(),cv2.COLOR_RGB2BGR)
        lab_img = self._MR_saliency__MR_readimg(img)
    
        labels = self._MR_saliency__MR_superpixel(lab_img)
        
        up,right,low,left = self._MR_saliency__MR_boundary_indictor(labels)
        aff = self._MR_saliency__MR_affinity_matrix(lab_img,labels)

        up_sal = 1- self._MR_saliency__MR_saliency(aff,up)
        up_img = self._MR_saliency__MR_fill_superpixel_with_saliency(labels,up_sal)
        up_img = up_img.astype(np.uint8)
    
        right_sal = 1-self._MR_saliency__MR_saliency(aff,right)
        right_img =  self._MR_saliency__MR_fill_superpixel_with_saliency(labels,right_sal)
        right_img = right_img.astype(np.uint8)

        low_sal = 1-self._MR_saliency__MR_saliency(aff,low)
        low_img = self._MR_saliency__MR_fill_superpixel_with_saliency(labels,low_sal)
        low_img = low_img.astype(np.uint8)
    
        left_sal = 1-self._MR_saliency__MR_saliency(aff,left)
        left_img = self._MR_saliency__MR_fill_superpixel_with_saliency(labels,left_sal)
        left_img = left_img.astype(np.uint8)
        

        plt.subplot(3,2,1)
        plt.title('orginal')
        plt.axis('off')
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        
        plt.subplot(3,2,2)
        plt.title('up')
        plt.axis('off')
        plt.imshow(up_img,'gray')
        
        plt.subplot(3,2,3)
        plt.title('right')
        plt.axis('off')
        plt.imshow(right_img,'gray')
        
        plt.subplot(3,2,4)
        plt.title('low')
        plt.axis('off')
        plt.imshow(low_img,'gray')
        
        plt.subplot(3,2,5)
        plt.title('left')
        plt.axis('off')
        plt.imshow(left_img,'gray')
        
        plt.subplot(3,2,6)
        plt.title('integrated')
        plt.axis('off')
        saliency_map = MR_debuger().saliency(img).astype(np.uint8)
        plt.imshow( saliency_map,'gray')
        plt.show()
