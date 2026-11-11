import cv2
import numpy as np

COLS, ROWS, CASE = 10, 7, 200   # 10x7 cases de 200 px chacune (donc 9x6 coins intérieurs)

img = np.full(((ROWS + 2) * CASE, (COLS + 2) * CASE), 255, np.uint8)   # fond blanc + bordure d'une case
for j in range(ROWS):
    for i in range(COLS):
        if (i + j) % 2 == 0:
            img[(j + 1) * CASE:(j + 2) * CASE, (i + 1) * CASE:(i + 2) * CASE] = 0

cv2.imwrite("damier.png", img)