import numpy as np
import cv2
from ArucoMarkerInfo import ArucoMarkerInfo

class ArucoMarker:
    def __init__(self):
        self.cells = np.zeros((7, 7), dtype=int)
        self.rows = 5
        self.cols = 5
        self.rotation = 0
        self.id = -1
        self.validated = False
        self.info = ArucoMarkerInfo()
        self.projected = []

    def attach_info(self, _info):
        self.info = _info

    def calculate_id(self):
        self.id = 0
        for i in range(1, 6):
            self.id <<= 1
            self.id |= self.cells[i][2]
            self.id <<= 1
            self.id |= self.cells[i][4]
        return self.id

    def validate(self):
        self.validated = False
        if len(self.projected) == 0:
            return False

        bad = 0
        for i in range(7):
            if self.cells[i][0] != 0 or self.cells[i][6] != 0 or self.cells[0][i] != 0 or self.cells[6][i] != 0:
                bad += 1
                if bad > 3:
                    return False

        for _ in range(4):
            if self.hamming_distance() == 0:
                self.calculate_id()
                self.validated = True
                return True
            self.rotate()

        return False

    def rotate(self):
        self.cells = np.rot90(self.cells)
        self.rotation += 1
        self.projected = self.projected[1:] + self.projected[:1]

    def hamming_distance(self):
        ids = np.array([
            [1, 0, 0, 0, 0],
            [1, 0, 1, 1, 1],
            [0, 1, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ])
        dist = 0
        for i in range(1, 6):
            min_sum = np.inf
            for j in range(4):
                sum_ = np.sum(self.cells[i, 1:6] != ids[j])
                print (self.cells[i,1:7], ids[j],self.cells[i, 1:7] != ids[j], sum_,min_sum)
                if sum_ < min_sum:
                    min_sum = sum_
            dist += min_sum
        return dist

    def print(self):
        print(f"{{")
        print(f"    Valid: {self.validated}")
        print(f"    Hamming: {self.hamming_distance()}")
        print(f"    ID: {self.id}")
        print(f"    Cells: [")
        for i in range(7):
            for j in range(7):
                print(f"{self.cells[i][j]}, ", end='')
            print("")
        print(f"    Rotation: {self.rotation}")
        for point in self.projected:
            print(f"    Projected: {point[0]}, {point[1]}")
        self.info.print()
        print(f"}}")
