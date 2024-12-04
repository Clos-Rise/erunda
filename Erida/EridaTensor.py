import numpy as np

class EridaTensor:
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = self.data.shape

    def reshape(self, new_shape):
        try:
            self.data = self.data.reshape(new_shape)
            self.shape = self.data.shape
        except ValueError as e:
            print(f"[Э - шибка] reshaping tensor: {e}")

    def add_dimension(self, axis, value):
        self.data = np.expand_dims(self.data, axis=axis)
        self.data = np.insert(self.data, 0, value, axis=axis)
        self.shape = self.data.shape

    def remove_dimension(self, axis):
        self.data = np.take(self.data, indices=0, axis=axis)
        self.shape = self.data.shape

    def __repr__(self):
        return f"EridaTensor(shape={self.shape}, data=\n{self.data})"

data = [[1, 2, 3], [4, 5, 6]]
erida_tensor = EridaTensor(data)
print("Original:")
print(erida_tensor)

erida_tensor.reshape((3, 2))
print("\nReshaped:")
print(erida_tensor)

erida_tensor.add_dimension(axis=0, value=0)
print("\nAdded dimension:")
print(erida_tensor)

erida_tensor.remove_dimension(axis=0)
print("\nRemoved dimension:")
print(erida_tensor)
