import skimage
import skimage.io
import os
import numpy as np
import cv2

class DataLoader:
    def __init__(self,opt):
        self.batchSize = opt.batch_size
        self.dataPath = opt.data_path
        self.isTrain = opt.is_train
        self.imgResizeW = opt.img_resize_w
        self.imgResizeH = opt.img_resize_h
        self.trainFolderName = opt.train_folder_name
        self.trainImageFolderName = opt.train_image_folder_name
        self.trainDetFolderName = opt.train_det_folder_name
        self.trainGtFolderName = opt.train_Gt_folder_name
        self.trainDetFileName = opt.train_det_file_name
        self.trainGtFileName = opt.train_Gt_file_name
        self.testFolderName = opt.test_folder_name
        self.testImageFolderName = opt.test_image_folder_name
        self.testDetFolderName = opt.test_det_folder_name
        self.testDetFileName = opt.test_det_file_name
        self.imgBeginIndex = opt.img_begin_index
        self.imgIndexBits = opt.img_index_bits
        self.imgIndexTrans = '%0'+str(self.imgIndexBits)+'d'
        self.imgCacheSize = opt.img_cache_size
        self.imgFileType = opt.img_file_type
        self.maxTargetNumber = opt.max_target_number
        self.detThreshold = opt.det_threshold
        self.detDim = opt.det_dim
        self.productDim = opt.product_dim
        self.imgCache = {}
        self.detCache = []
        self.gtCache = []

        if self.isTrain:
            self.workPath = os.path.join(self.dataPath,self.trainFolderName)
        else:
            self.workPath = os.path.join(self.dataPath,self.testFolderName)

        self.workDirs = os.listdir(self.workPath)
        for i in range(len(self.workDirs)):
            tmpPath = os.path.join(self.workPath,self.workDirs[i])
            if not os.path.isdir(tmpPath):
                del self.workDirs[i]
        
    def flashLoader(self):
        self.loaderIndex = self.imgBeginIndex
        self.imgCacheIndex = self.imgBeginIndex
        self.cachedet()
        self.cacheimg()
        self.cachegt()

    def endLoader(self):
        for _ in range(self.batchSize):
            dir = self.workDirs.pop(0)
            self.workDirs.append(dir)
        self.detCache.clear()
        self.gtCache.clear()
        self.imgCache.clear()

    def next(self):
        if self.isTrain:
            img = self.nextimg()
            det = self.nextdet()
            gt = self.nextgt()
        else:
            img = self.nextimg()
            det = self.nextdet()
            gt = None

        self.loaderIndex += 1
        return {'img':img,'det':det,'gt':gt}
            
    def nextimg(self):
        index = self.imgIndexTrans%self.loaderIndex
        ret = []
        if index in self.imgCache:
            ret = self.imgCache[index]
            del self.imgCache[index]
        else:
            self.cacheimg()
            ret = self.nextimg()
        return ret
    
    def nextdet(self):
        ret = []
        if self.loaderIndex<=len(self.detCache):
            ret = self.detCache[self.loaderIndex-1]
        return ret

    def nextgt(self):
        ret = []
        if self.loaderIndex<=len(self.gtCache):
            ret = self.gtCache[self.loaderIndex-1]
        return ret

    def cachedet(self):
        batch=[]
        maxIndex = 0
        for i in range(self.batchSize):
            tmp = {}
            path = os.path.join(self.workPath,self.workDirs[i],self.trainDetFolderName,self.trainDetFileName)
            if os.path.exists(path):
                with open(path,"r+") as inputf:
                    while True:
                        line = inputf.readline()
                        if not line:
                            break
                        list = line.strip("\n").split(",")
                        index = int(list[0])
                        if index > maxIndex:
                            maxIndex = index
                        if float(list[6])>self.detThreshold:
                            if index not in tmp:
                                tmp[index]=[[float(list[2]),float(list[3]),float(list[4]),float(list[5]),float(list[6])]]
                            else:
                                tmp[index].append([float(list[2]),float(list[3]),float(list[4]),float(list[5]),float(list[6])])
                            
            batch.append(tmp)
        self.detCache = self.makeDetTarget(batch,maxIndex,self.detDim)

    def makeDetTarget(self,batch,maxIndex,dim):
        ret = []
        for i in range(maxIndex):
            tmp = []
            for j in range(len(batch)):
                map = batch[j]
                if i+1 in map:
                    subitem = self.fillDetTarget(map[i+1],dim)
                    tmp.append(np.array(subitem).reshape(1,self.maxTargetNumber,dim))
            tmp_2 = np.concatenate(tmp,0)
            ret.append(tmp_2)
        return ret

    def fillDetTarget(self,batch,dim):
        for i in range(self.maxTargetNumber-len(batch)):
            batch.append([0]*dim)
        return batch

    def cachegt(self):
        batch=[]
        maxIndex = 0
        for i in range(self.batchSize):
            tmp = {}
            path = os.path.join(self.workPath,self.workDirs[i],self.trainGtFolderName,self.trainGtFileName)
            if os.path.exists(path):
                with open(path,"r+") as inputf:
                    while True:
                        line = inputf.readline()
                        if not line:
                            break
                        list = line.strip("\n").split(",")
                        index = int(list[0])
                        if index > maxIndex:
                            maxIndex = index
                        if index not in tmp:
                            tmp[index]={}
                        targetindex = int(list[1])
                        tmp[index][targetindex] = [float(list[2]),float(list[3]),float(list[4]),float(list[5])]
                            
            batch.append(tmp)
        self.gtCache = self.makeGtTarget(batch,maxIndex,self.productDim)

    def makeGtTarget(self,batch,maxIndex,dim):
        ret = []
        for i in range(maxIndex):
            tmp = []
            for j in range(len(batch)):
                map = batch[j]
                if i+1 in map:
                    subitem = self.fillGtTarget(map[i+1],dim)
                    tmp.append(np.array(subitem).reshape(1,self.maxTargetNumber,dim))
            tmp_2 = np.concatenate(tmp,0)
            ret.append(tmp_2)
        return ret

    def fillGtTarget(self,batch,dim):
        ret = []
        for i in range(1,self.maxTargetNumber+1):
            if i in batch:
                ret.append(batch[i])
            else:
                ret.append([0]*dim)
        return ret

    def cacheimg(self):
        index =self.imgCacheIndex
        strIndex = self.imgIndexTrans%index
        if strIndex in self.imgCache and self.imgCache[strIndex] == []:
            self.imgCache[strIndex] = []
        else:
            batchPathList = []
            for i in range(self.batchSize):
                path = os.path.join(self.workPath,self.workDirs[i])
                batchPathList.append(path)
            for _ in range(self.imgCacheSize):
                
                imgarray = []
                strIndex = self.imgIndexTrans%index
                imgname = strIndex+self.imgFileType
                for path in batchPathList:
                    imgPath = os.path.join(path,self.trainImageFolderName,imgname)
                    if os.path.exists(imgPath):
                        # img = skimage.io.imread(imgPath)
                        img = cv2.imread(imgPath,cv2.IMREAD_COLOR)
                        w,h,c = img.shape
                        dim_diff = np.abs(h - w)
                        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
                        pad = ((pad1, pad2), (0, 0), (0, 0)) if h >= w else ((0, 0), (pad1, pad2), (0, 0))
                        img = np.pad(img, pad, 'constant', constant_values=0)
                        img = img / 255.0
                        img = skimage.transform.resize(img, (self.imgResizeH, self.imgResizeW))
                        # short_edge = min(img.shape[:2])
                        # yy = int((img.shape[0] - short_edge) / 2)
                        # xx = int((img.shape[1] - short_edge) / 2)
                        # crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
                        # resized_img = skimage.transform.resize(crop_img, (224, 224))[None, :, :, :]
                        reshapeimg = img.reshape((1,self.imgResizeH,self.imgResizeW,c))
                        imgarray.append(reshapeimg)
                if len(imgarray)==0:
                    self.imgCache[strIndex]=[]
                    break
                else:
                    imgbatch = np.concatenate(imgarray,0)
                    self.imgCache[strIndex]=imgbatch
                index += 1
            self.imgCacheIndex = index