# Load required libraries and image
import cv2
import numpy as np
from sklearn.cluster import KMeans
#from skimage.segmentation import slic
import os
# assign directory
directory = 'images-kaggle'

color_dict = {
    0: 'infected',
    1: 'non infected',
    2: 'background',
    3: 'infected',
    4: 'non-infected'
}

color_label = {
    0: [75,67,145],
    1: [141,43,86],
    2: [176,220,190],
    3: [120,140,170],
    4: [180,140,160]
}

#HSV
'''
color_label = {
    0: [168,76,113],
    1: [182,105,167],
    2: [216,41,71]
}
'''


for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        image_path = str(f) # path of the image
        img = cv2.imread(image_path)
        h, w, e = img.shape
        # GrayScale image
        #hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_img = img
        hsv_img = hsv_img.reshape((hsv_img.shape[0] * hsv_img.shape[1], 3))
        # cluster the pixel intensities
        clt = KMeans(n_clusters = 3)
        clt.fit(hsv_img)
        print('clustering done')
        seeds = cv2.ximgproc.createSuperpixelSEEDS(w, h, e, 100, 4, 2, 5)
        seeds.iterate(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), 10)
        # retrieve the segmentation result
        labels = seeds.getLabels()
        mask = seeds.getLabelContourMask(False)
        print('seeds done')
        
        color_img = np.zeros((h,w,3), np.uint8)
        color_img[:] = (0, 0, 255)
        mask_inv = cv2.bitwise_not(mask)
        result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
        result_fg = cv2.bitwise_and(color_img, img, mask=mask)
        result = cv2.add(result_bg, result_fg)
        
        cv2.namedWindow('result',0)
        
        # grab the number of different clusters and create a histogram
        # # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        print(clt.labels_.shape)
        (hist, _) = np.histogram(clt.labels_, bins = numLabels)
        # normalize the histogram
        hist = hist.astype("float")
        hist /= hist.sum()
        # representing the number of pixels labeled to each color
        centroids = clt.cluster_centers_
        color_values = centroids.astype("uint8").tolist()
        colours = {}
        for colour_val, hist_val in zip(color_values, hist):
            b1,g1,r1 = colour_val
            min_dis = 9999999
            for k in range(0,5):
                r2,g2,b2 = color_label[k]
                dist = (b1-b2)*(b1-b2) + (g1-g2)*(g1-g2) + (r1-r2)*(r1-r2)
                #dist = (b1-b2)*(b1-b2)
                if dist < min_dis:
                    min_dis = dist
                    index = k
            if color_dict[index] not in colours:   
                colours[color_dict[index]] = round(hist_val * 100, 4)

        imageLabel = ''
        prevy = 25
        for state, perc in colours.items():
            imageLabel = str(state) + ' : ' + str(perc) + '%'
            cv2.putText(img,imageLabel,(5,prevy),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10,10,90),2)
            prevy = prevy + 25
            #imageLabel = imageLabel + str(state) + ':' + str(perc) + 

        cv2.imshow('image',img)
        cv2.imwrite('converted/'+str(f),img)

        cv2.waitKey(30)
        print('Finished processing: '+ str(f))

cv2.destroyAllWindows()