import cv2 as cv # импортирую библиотеку компьютерного зрения
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
# функция обработки изображений
def processData(way, iterImg):

	img = cv.imread(os.path.join(way, iterImg))

	resImg = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA) # изменяю размер изображений для эффективности обучения

	
	cv.imwrite(way.split('/new')[0] + '/cans_new/' + iterImg, resImg)
	# ang = 0

	# while ang < 360:
	# 	center = (resImg.shape[1]//2, resImg.shape[0]//2)
	# 	M = cv.getRotationMatrix2D(center, ang, 1.0)
	# 	img = cv.warpAffine(resImg, M, (resImg.shape[1], resImg.shape[0]))
	# 	cv.imwrite(way.split('/')[0] + '_new/' + '%s' % iterImg + str(ang) + 'r' + '.jpg', img)

	# 	flip = cv.flip(img, 1)

	# 	cv.imwrite(way.split('/')[0] + '_new/' + '%s' % iterImg + str(ang) + 'f' + '.jpg', flip)
	# 	ang += 90


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
# arrCans = []
# arrBottles = []
# arrPapers = []

path = str(input('Input path...'))
path = os.path.join(os.getcwd(), path)

for file_name in os.listdir(path):
	processData(path, file_name)

# добавление массивов пикселов в массив картинок
# for i in range(n):
	# arrCans.append(processData('cans/', i))
	# arrBottles.append(processData('bottles/', i))
	# arrPapers.append(processData('papers/', i))

# X = [] # набор признаков
# Y = [] # набор меток

# X = (arrCans + arrBottles + arrPapers) # присоединение массивов изображений разных классов

# # нехитрое создание индексной зависимости меток от признаков (признаку i-ного элемента соответствует i-ный элемент метки) 
# for i in range(3 * n):
# 	if i <= 59:
# 		Y.append('can')
# 	if i > 59 and i <= 119:
# 		Y.append('bottle')
# 	if i > 119:
# 		Y.append('paper')

# logistic = LogisticRegression()

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, shuffle = True)
	
# logistic.fit(X_train, Y_train)
# logistic.score(X_train, Y_train)
# print('Coefficient train: \n', logistic.coef_)
# print('Intercept train: \n', logistic.intercept_)
# print('R² Value train: \n', logistic.score(X_train, Y_train))

# print('Coefficient test: \n', logistic.coef_)
# print('Intercept test: \n', logistic.intercept_)
# print('R² Value test: \n', logistic.score(X_test, Y_test))

# print(logistic.predict(X_test))
# print(Y_test)