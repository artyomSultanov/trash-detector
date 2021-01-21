import cv2 as cv # импортирую библиотеку компьютерного зрения
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os



iteratorOfData = 0
# функция обработки изображений
def processData(way, file):
	global iteratorOfData
	img = cv.imread(os.path.join(way, file))

	resImg = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA) # изменяю размер изображений для эффективности обучения

	ang = 0

	while ang < 360:
		center = (resImg.shape[1]//2, resImg.shape[0]//2)
		M = cv.getRotationMatrix2D(center, ang, 1.0)
		img = cv.warpAffine(resImg, M, (resImg.shape[1], resImg.shape[0]))
		cv.imwrite(way.split('/bottles')[0] + '/' + 'data/bottles/' + str(iteratorOfData) + str(ang) + 'r.jpg', img)

		iteratorOfData += 1
		flip = cv.flip(img, 1)

		cv.imwrite(way.split('/bottles')[0] + '/' + 'data/bottles/' + str(iteratorOfData) + str(ang) + 'f.jpg', flip)
		iteratorOfData += 1
		ang += 90


	# arrImg = [] # инициализирую массив пикселов

	# ang = 0
	# while ang < 360:
	# 	arrImg.append(increaseData(resImg, ang)[0])
	# 	arrImg.append(increaseData(resImg, ang)[1])
	# 	ang += 90
	# return arrImg # вывод массива пикселей 

# def increaseData(img, angle):
# 	setSimilar = [[]]
# 	center = (img.shape[1]//2, img.shape[0]//2)

# 	M = cv.getRotationMatrix2D(center, angle, 1.0)
# 	rotImg = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))
# 	for i in range(128):
# 		for j in range(128):
# 			setSimilar[0].append(rotImg[i][j])

# 	flip = cv.flip(rotImg, 1)
# 	for i in range(128):
# 		for j in range(128):
# 			setSimilar[1].append(flip[i][j])
			
# 	return setSimilar

n = 200 # количество изображений в одной папке
# инициализация массивов для представления картинок в численном виде

path = str(input('Input path...'))
path = os.path.join(os.getcwd(), path)

# добавление массивов пикселов в массив картинок
for file_name in os.listdir(path):
	processData(path, file_name)

