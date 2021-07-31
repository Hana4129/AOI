# -*- coding: utf-8 -*-
#=========================================================
# プリント基板ハンダ検査装置
# 前処理 ハンダ位置抽出処理
#=========================================================
import sys
import os
import numpy as np
import cv2
import ntpath
import matplotlib.pyplot as plt
#---------------------------------------------------------
#PATH = os.path.dirname(_file_)
from pathlib import Path
PATH = Path().resolve() # カレントパス取得
INPUT_FILE = os.path.join(PATH,"data/img/test_pbc_img.tif")  # プリント基板イメージ

fiducialTemplate = './data/templates/fiducial.tif' # 基盤端取得用のテンプレート画像
fiducialPositions = list()  # 基盤端の基準点格納用リスト
#---------------------------------------------------------

# 動作確認用　イメージ表示処理
def showImg(rgbImg,binImg):

  # WindowNameを定義    
  rgbWnd = "RGBImage"
  binWmd = "BinaryImage"

  #Window縦サイズは固定サイズ
  WndHight = 600

  # RGB 画像を表示
  cv2.namedWindow(rgbWnd, cv2.WINDOW_NORMAL)
  (h,w) = rgbImg.shape[:2]
  ratio = int(h / WndHight)
  wndWidth = int(w / ratio)

  cv2.resizeWindow(rgbWnd,wndWidth,WndHight)
  cv2.imshow(rgbWnd, rgbImg)

  # 2値画像を表示
  cv2.namedWindow(binWmd, cv2.WINDOW_NORMAL)
  (h,w) = binImg.shape[:2]
  ratio = int(h / WndHight)
  wndWidth = int(w / ratio)  
  cv2.resizeWindow(binWmd,wndWidth,WndHight)
  cv2.imshow(binWmd, binImg)

  cv2.waitKey(0)
  cv2.destroyAllWindows()



# 輪郭描画処理
# 概要：ハンダ位置候補の輪郭抽出箇所に矩形を描画
def drawRect(srcImage,dstImage):
  
  contours = cv2.findContours(srcImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

  # 矩形描画
  for idx, rect in enumerate(contours):
    if len(rect) > 0:
      x, y, w, h = cv2.boundingRect(rect)
      cv2.rectangle(dstImage,(x, y),(x + w, y + h),(0, 125, 0),20)

# 抽出用 画像フィルタリング処理
# 概要：元画像の抽出する上でのノイズ除去、２値画像を取得する
def filter(image):
  
  grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  ksize = 31
  imgMask = cv2.medianBlur(grayImg,ksize)
  imgMask = cv2.bitwise_not(imgMask)

  ret , imgMask = cv2.threshold(imgMask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  return imgMask


# プリント基盤部分の抽出処理
# 概要：元画像の背景部分を除き、プリント基盤部分だけを抽出する。
# Reference: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = np.asarray(pts, np.float32) # order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped


# 基盤端判定用の基準点取得処理
# 概要：テンプレート画像と一致する部分を検出し、プリント基盤の端を特定する
# テンプレート画像はボードの端の特徴となるビス止め箇所とする
def findFiducial(imgGray, fiducial, region):
    x, y, w, h = region
    #print('ROI width: ' + str(w) + " height: " + str(h))

    # imgGray画像よりテンプレート画像(fiducial)にマッチする箇所を取得
    roi = imgGray[y:y + h, x:x + w]
    res = cv2.matchTemplate(roi, fiducial, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9

    templateWidth, templateHeight =  fiducial.shape[::-1]
    
    # マッチング結果の類似度 最大最小を取得
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #print('Max loc: ' + str(max_loc))

    # 類似度が最大の座標を保存
    fiducialPos = [(x + max_loc[0], y + max_loc[1]), (x + max_loc[0] + templateWidth, y + max_loc[1] + templateHeight)]
    fiducialCenterPos = (fiducialPos[0][0] + int(templateWidth / 2),fiducialPos[0][1] + int(templateHeight / 2))    
    fiducialPositions.append(fiducialCenterPos)


def main():
  
  in_f = cv2.imread(INPUT_FILE)

  imgGray = cv2.cvtColor(in_f, cv2.COLOR_BGR2GRAY)

  # 基盤の端を判定するための基準点を取得
  fiducial = cv2.imread(fiducialTemplate, 0)
  imageHeight, imageWidth = imgGray.shape[:2]
  findFiducial(imgGray, fiducial, (0, 0, int(imageWidth / 2), int(imageHeight / 2)))
  findFiducial(imgGray, fiducial, (int(imageWidth / 2), 0, int(imageWidth / 2), int(imageHeight / 2)))
  findFiducial(imgGray, fiducial, (int(imageWidth / 2), int(imageHeight / 2), int(imageWidth / 2), int(imageHeight / 2)))
  findFiducial(imgGray, fiducial, (0, int(imageHeight / 2), int(imageWidth / 2), int(imageHeight / 2)))

  # 基盤部分の画像だけを切り出す
  imgAdj = four_point_transform(in_f, fiducialPositions)

  # 2値化する
  binaryImg = filter(imgAdj)

  # ハンダ箇所の候補を抽出
  drawRect(binaryImg,imgAdj)

  # 処理前後の画像を表示
  showImg(imgAdj,binaryImg)

main()

