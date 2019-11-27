import numpy as np
from sklearn.svm import SVC

#3.1

spiralX = np.load("data_lab5/spiral_X.npy")
spiralY = np.load("data_lab5/spiral_Y.npy")

'''
clf = SVC(kernel='poly', max_iter=100000, degree=2) #2 is best
ft = clf.fit(spiralX, spiralY)
pred = clf.predict(spiralX)
score = clf.score(spiralX, spiralY)
'''

#TO DO reszta obliczen i wykres dla wszystkich

'''
rbf = SVC(kernel='rbf', max_iter=100000, gamma='auto') #gamma = 0.5
rbf_ft = rbf.fit(spiralX, spiralY)
rbf_pred = rbf.predict(spiralX)
rbf_score = rbf.score(spiralX,spiralY)
'''

#3.2
chessX = np.load("data_lab5/chess33_X.npy")
chessY = np.load("data_lab5/chess33_Y.npy")

'''
chess_rbf = SVC(C=np.inf, kernel='rbf', max_iter=100000, gamma=0.01)
chess_ft = chess_rbf.fit(chessX, chessY)
chess_pred = chess_rbf.predict(chessX)
chess_score = chess_rbf.score(chessX, chessY)
print(chess_rbf.support_vectors_.shape)
print(chess_score)
'''

#3.3
n_chessX = np.load("data_lab5/chess33n_X.npy")
n_chessY = np.load("data_lab5/chess33n_Y.npy")

n_chess_rbf = SVC(C=1, kernel='rbf', max_iter=100000, gamma=0.01)
n_chess_ft = n_chess_rbf.fit(chessX, chessY)
n_chess_pred = n_chess_rbf.predict(chessX)
n_chess_score = n_chess_rbf.score(chessX, chessY)
print(n_chess_rbf.support_vectors_.shape)
print(n_chess_score)