import cv2
import numpy as np

class Quadrilateral:
    def __init__(self, a=None, b=None, c=None, d=None):
        if a is None or b is None or c is None or d is None:
            self.points = [np.float32([0.0, 0.0])] * 4
        else:
            self.points = [a, b, c, d]

    def area(self):
        return cv2.contourArea(np.array(self.points))

    def contains_point(self, p):
        return cv2.pointPolygonTest(np.array(self.points), p, False) >= 0.0

    def draw(self, image, color, weight=1):
        for j in range(3):
            cv2.line(image, tuple(self.points[j]), tuple(self.points[j + 1]), color, weight, 8)
        cv2.line(image, tuple(self.points[3]), tuple(self.points[0]), color, weight, 8)

    def print(self):
        print(f"[{self.points[0]}, {self.points[1]}, {self.points[2]}, {self.points[3]}]")

    @staticmethod
    def bigger_quadrilateral(quads):
        max_quad = quads[0]
        max_area = quads[0].area()

        for quad in quads[1:]:
            area = quad.area()
            if area > max_area:
                max_quad = quad
                max_area = area

        return max_quad

    @staticmethod
    def draw_vector(image, quads, color):
        for quad in quads:
            quad.draw(image, color)
