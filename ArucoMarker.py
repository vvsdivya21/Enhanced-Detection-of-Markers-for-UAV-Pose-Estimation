import numpy as np
import cv2
from ArucoMarkerInfo import ArucoMarkerInfo

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

class ArucoMarker:
    def __init__(self):
        self.cells = np.zeros((8, 8), dtype=int)
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

    def bytes_to_6x6_matrix(self, byte_pattern):
        # Convert bytes to binary representation (8 bits per byte)
        bits = np.unpackbits(byte_pattern)
        # Flatten the bits array
        flat_bits = bits.flatten()
        # We need exactly 36 bits for a 6x6 matrix
        if len(flat_bits) < 36:
            raise ValueError("Not enough bits to form a 6x6 matrix.")
        # Extract the first 36 bits
        six_by_six_bits = flat_bits[:36]
        # Reshape into 6x6 matrix
        bit_pattern_6x6 = six_by_six_bits.reshape(6, 6)
        return bit_pattern_6x6
    
    def hamming_distance(self):
        bit_pattern_6x6 = []
        hamming_dist=[]
        for i in range (4):
            byte_pattern = aruco_dict.bytesList[i]
            bit_pattern_6x6= self.bytes_to_6x6_matrix(byte_pattern)
            bit_pattern_6x6 = bit_pattern_6x6[:-1, :]
            

            dist = 0
            for k in range(1, 7):
                min_sum = np.inf
                for j in range(len(bit_pattern_6x6)):
                    sum_ = np.sum(self.cells[k, 1:7] != bit_pattern_6x6[j])
                    print (self.cells[k,1:7], bit_pattern_6x6[j],self.cells[k, 1:7] != bit_pattern_6x6[j], sum_,min_sum)

                    if sum_ < min_sum:
                        min_sum = sum_
                dist += min_sum
            print ('________________________end of marker id______',i+1, 'dist=', dist)
            hamming_dist.append(dist)
        return hamming_dist



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
        self.info.print_info()
        print(f"}}")
