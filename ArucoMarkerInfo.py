import numpy as np

class Transformations:
    @staticmethod
    def rotation_matrix(euler):
        """
        Creates a new rotation matrix from Euler rotation.
        
        :param euler: A tuple or list containing the Euler angles (x, y, z).
        :return: Rotation matrix created from Euler rotation.
        """
        x, y, z = euler
        
        # Rotation on x axis
        rx = np.array([[1, 0, 0],
                       [0, np.cos(x), -np.sin(x)],
                       [0, np.sin(x), np.cos(x)]])
        
        # Rotation on y axis
        ry = np.array([[np.cos(y), 0, np.sin(y)],
                       [0, 1, 0],
                       [-np.sin(y), 0, np.cos(y)]])
        
        # Rotation on z axis
        rz = np.array([[np.cos(z), -np.sin(z), 0],
                       [np.sin(z), np.cos(z), 0],
                       [0, 0, 1]])
        
        return rz @ ry @ rx

class ArucoMarkerInfo:
    def __init__(self, _id=-1, _size=1.0, _position=(0.0, 0.0, 0.0), _rotation=(0.0, 0.0, 0.0)):
        """
        Aruco marker constructor.
        :param _id: Marker id.
        :param _size: Marker size in meters.
        :param _position: Marker world position as a tuple (x, y, z).
        :param _rotation: Marker world Euler rotation as a tuple (x, y, z).
        """
        self.id = _id
        self.size = _size
        self.position = np.array(_position)
        self.rotation = np.array(_rotation)
        self.world = []
        self.calculate_world_points()

    def calculate_world_points(self):
        """
        Calculate the marker world points, considering the marker center position and rotation.
        First the marker is rotated and is translated after so the rotation is always relative to the marker center.
        """
        half = self.size / 2.0

        self.world = [
            np.array([-half, -half, 0]),
            np.array([-half, +half, 0]),
            np.array([half, +half, 0]),
            np.array([half, -half, 0])
        ]

        rot = Transformations.rotation_matrix(self.rotation)

        for i in range(len(self.world)):
            temp = np.array([self.world[i][0], self.world[i][1], self.world[i][2]])
            transf = np.dot(rot, temp)
            self.world[i] = transf + self.position

    def print_info(self):
        """
        Print info about this marker to the stdout.
        """
        print("{")
        print(f"    ID: {self.id}")
        print(f"    Size: {self.size}")
        print(f"    Position: {self.position[0]}, {self.position[1]}, {self.position[2]}")
        print(f"    Rotation: {self.rotation[0]}, {self.rotation[1]}, {self.rotation[2]}")
        for point in self.world:
            print(f"    World: {point[0]}, {point[1]}, {point[2]}")
        print("}")
'''
# Example usage
aruco_marker = ArucoMarkerInfo(1, 0.15, (1.0, 2.0, 3.0), (0.1, 0.2, 0.3))
aruco_marker.print_info()
'''