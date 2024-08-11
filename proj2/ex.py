# STEP 1 : import modules
import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2 : create inference instance
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

# STEP 3 : load input image
img1= cv2.imread("iu1.jpg")
img2= cv2.imread("iu2.jpg")
img3= cv2.imread("iu3.jpeg")
img4= cv2.imread("s1.jpg")
img5= cv2.imread("s2.jpg")

# STEP 4 : inference
faces1 = app.get(img1)
faces2 = app.get(img2)
faces3 = app.get(img3)
faces4 = app.get(img4)
faces5 = app.get(img5)
# print(len(faces1))
# print(len(faces2))

# SETP 5-2 : face similarity
# then print all-to-all face similarity
feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
feat3 = np.array(faces3[0].normed_embedding, dtype=np.float32)
feat4 = np.array(faces4[0].normed_embedding, dtype=np.float32)
feat5 = np.array(faces5[0].normed_embedding, dtype=np.float32)
sims = np.dot(feat4, feat5.T)
print(sims)




# assert len(faces1)==1
# assert len(faces2)==1

# print(faces)

# STEP 5-1 : draw detection result
# rimg = app.draw_on(img, faces)
# # cv2.imshow("test", rimg)
# # cv2.waitKey(0)
# cv2.imwrite("./t1_output.jpg", rimg)

# # SETP 5-2 : face similarity
# # then print all-to-all face similarity
# feats = []
# for face in faces:
#     feats.append(face.normed_embedding)
# feats = np.array(feats, dtype=np.float32)
# sims = np.dot(feats, feats.T)
# print(sims)
