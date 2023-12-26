import os
basepath = os.path.abspath(__file__)
folder = os.path.dirname(basepath)
data_path = os.path.join(folder, 'data.txt')

print(basepath)
print(folder)
print(data_path)