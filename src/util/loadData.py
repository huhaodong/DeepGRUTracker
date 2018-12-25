import skimage
import skimage.io
import os
import numpy as np
import cv2
import configparser


class DataLoader:
    def __init__(self, opt):
        self.batchSize = opt.batch_size
        self.dataPath = opt.data_path
        self.isTrain = opt.is_train
        self.imgResizeW = opt.img_resize_w
        self.imgResizeH = opt.img_resize_h

        self.trainFolderName = opt.train_folder_name

        self.sequenceInfoFileName = opt.sequence_info_file_name

        self.testFolderName = opt.test_folder_name

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
        self.seqinfoList = []
        self.seqPicInfo = []
        self.conf = configparser.ConfigParser()

        self.imageFolderName = opt.train_image_folder_name
        self.detFolderName = opt.train_det_folder_name
        self.gtFolderName = opt.train_gt_folder_name
        self.detFileName = opt.train_det_file_name
        self.gtFileName = opt.train_gt_file_name

        if self.isTrain:
            self.workPath = os.path.join(self.dataPath, self.trainFolderName)

        else:
            self.workPath = os.path.join(self.dataPath, self.testFolderName)
            self.imageFolderName = opt.test_image_folder_name
            self.detFolderName = opt.test_det_folder_name
            self.detFileName = opt.test_det_file_name

        self.workDirs = os.listdir(self.workPath)
        for i in range(len(self.workDirs)):
            tmpPath = os.path.join(self.workPath, self.workDirs[i])
            if not os.path.isdir(tmpPath):
                del self.workDirs[i]
        print("in data loader")
        for i in self.workDirs:
            print("++"+i)
    def flashLoader(self, gtFlage=True):
        self.loaderIndex = self.imgBeginIndex
        self.imgCacheIndex = self.imgBeginIndex

        for i in range(self.batchSize):
            path = os.path.join(
                self.workPath, self.workDirs[i], self.sequenceInfoFileName)
            if os.path.exists(path):
                self.conf.read(path)
                self.seqinfoList.append(
                    self.conf.getint("Sequence", "seqLength"))
                self.seqPicInfo.append({
                    "height": self.conf.getint("Sequence", "imHeight"),
                    "width": self.conf.getint("Sequence", "imWidth")
                })

        self.cacheimg()
        self.cachedet()
        if gtFlage:
            self.cachegt()
        else:
            pass

    def endLoader(self, gtFlage=True):
        for _ in range(self.batchSize):
            dir = self.workDirs.pop(0)
            self.workDirs.append(dir)
        self.detCache.clear()
        if gtFlage:
            self.gtCache.clear()
        else:
            pass
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
        return {'img': img, 'det': det, 'gt': gt}

    def nextimg(self):
        index = self.imgIndexTrans % self.loaderIndex
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
        if self.loaderIndex <= len(self.detCache):
            ret = self.detCache[self.loaderIndex-1]
        return ret

    def nextgt(self):
        ret = []
        if self.loaderIndex <= len(self.gtCache):
            ret = self.gtCache[self.loaderIndex-1]
        return ret

    def cachedet(self):
        batch = []
        maxIndex = 0
        for i in range(self.batchSize):
            tmp = {}
            path = os.path.join(
                self.workPath, self.workDirs[i], self.detFolderName, self.detFileName)
            if os.path.exists(path):
                with open(path, "r+") as inputf:
                    while True:
                        line = inputf.readline()
                        if not line:
                            break
                        list = line.strip("\n").split(",")
                        index = int(list[0])
                        if index > maxIndex:
                            maxIndex = index
                        if float(list[6]) > self.detThreshold:
                            if index not in tmp:
                                tmp[index] = [[float(list[2]), float(list[3]), float(
                                    list[4]), float(list[5]), float(list[6])]]
                            else:
                                tmp[index].append([float(list[2]), float(list[3]), float(
                                    list[4]), float(list[5]), float(list[6])])

            batch.append(tmp)
        # batch = self.alignFrame(batch,self.detDim)
        self.detCache = self.alignDetTarget(batch, self.detDim)

    # def alignFrame(self,batch,dim):
    #     '''
    #         align the detection map
    #         det: [{1:[[x,y,w,h]]},{}] # batch_size*{}
    #     '''
    #     maxlen = max(self.seqinfoList)
    #     for i in range(1,maxlen+1):
    #         for j in batch:
    #             if i not in j:
    #                 j[i] = [[float(0)]*dim]

    def alignDetTarget(self, batch, dim):
        ret = []
        maxlen = max(self.seqinfoList)
        for i in range(1, maxlen+1):
            tmp = []
            for j in range(len(batch)):
                map = batch[j]
                if i not in map:
                    map[i] = [[float(0)]*dim]
                subitem = self.fillDetTarget(map[i], dim)
                tmp.append(np.array(subitem).reshape(
                    1, self.maxTargetNumber, dim))
            tmp_2 = np.concatenate(tmp, 0)
            # shape [sequence*bantchSize*maxTargetNumber*detDim]
            ret.append(tmp_2)
        return ret

    def fillDetTarget(self, batch, dim):
        for _ in range(self.maxTargetNumber-len(batch)):
            batch.append([float(0)]*dim)
        return batch

    def cachegt(self):
        batch = []
        maxIndex = 0
        for i in range(self.batchSize):
            tmp = {}
            path = os.path.join(
                self.workPath, self.workDirs[i], self.gtFolderName, self.gtFileName)
            if os.path.exists(path):
                with open(path, "r+") as inputf:
                    while True:
                        line = inputf.readline()
                        if not line:
                            break
                        list = line.strip("\n").split(",")
                        index = int(list[0])
                        if index > maxIndex:
                            maxIndex = index
                        if index not in tmp:
                            tmp[index] = {}
                        targetindex = int(list[1])
                        tmp[index][targetindex] = [float(list[2]), float(
                            list[3]), float(list[4]), float(list[5])]

            batch.append(tmp)
        self.gtCache = self.alignGtTarget(batch, self.productDim)

    def alignGtTarget(self, batch, dim):
        '''
            batch:[{frame:{target:[x,y,w,h]}},...]
        '''
        ret = []
        maxlen = max(self.seqinfoList)
        for i in range(1, maxlen+1):
            tmp = []
            for j in range(len(batch)):
                map = batch[j]
                if i not in map:
                    map[i] = self.alignGtFrame(dim)
                subitem = self.fillGtTarget(map[i], dim)
                tmp.append(np.array(subitem).reshape(
                    1, self.maxTargetNumber, dim))
            tmp_2 = np.concatenate(tmp, 0)
            # shape [sequence*bantchSize*maxTargetNumber*dim]
            ret.append(tmp_2)
        return ret

    def alignGtFrame(self, dim):
        ret = {}
        ret[1] = [float(0)]*dim
        return ret

    def fillGtTarget(self, batch, dim):
        ret = []
        for i in range(1, self.maxTargetNumber+1):
            if i in batch:
                ret.append(batch[i])
            else:
                ret.append([0]*dim)
        #shape [maxTargetNumber*dim]
        return ret

    def cacheimg(self):

        maxlen = max(self.seqinfoList)  # max frame

        index = self.imgCacheIndex
        strIndex = self.imgIndexTrans % index
        if strIndex in self.imgCache and self.imgCache[strIndex] == []:
            self.imgCache[strIndex] = []
        else:
            batchPathList = []
            for i in range(self.batchSize):
                path = os.path.join(self.workPath, self.workDirs[i])
                batchPathList.append(path)
            for _ in range(self.imgCacheSize):

                imgarray = []
                if index > maxlen:
                    pass
                else:
                    strIndex = self.imgIndexTrans % index
                    imgname = strIndex+self.imgFileType
                    for i in range(len(batchPathList)):
                        path = batchPathList[i]
                        imgPath = os.path.join(
                            path, self.imageFolderName, imgname)
                        if os.path.exists(imgPath):
                            # img = skimage.io.imread(imgPath)
                            img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
                            w, h, c = img.shape
                            dim_diff = np.abs(h - w)
                            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
                            pad = ((pad1, pad2), (0, 0), (0, 0)) if h >= w else (
                                (0, 0), (pad1, pad2), (0, 0))
                            img = np.pad(img, pad, 'constant',
                                         constant_values=0)
                            img = img / 255.0
                            img = skimage.transform.resize(
                                img, (self.imgResizeH, self.imgResizeW))
                            # short_edge = min(img.shape[:2])
                            # yy = int((img.shape[0] - short_edge) / 2)
                            # xx = int((img.shape[1] - short_edge) / 2)
                            # crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
                            # resized_img = skimage.transform.resize(crop_img, (224, 224))[None, :, :, :]
                            reshapeimg = img.reshape(
                                (1, self.imgResizeH, self.imgResizeW, c))
                            imgarray.append(reshapeimg)
                        else:
                            # picInfoMap = self.seqPicInfo[i]
                            bpic = np.zeros(
                                (1, self.imgResizeH, self.imgResizeW, 3))
                            imgarray.append(bpic)

                if len(imgarray) == 0:
                    self.imgCache[strIndex] = []
                    break
                else:
                    imgbatch = np.concatenate(imgarray, 0)
                    self.imgCache[strIndex] = imgbatch
                index += 1
            self.imgCacheIndex = index
