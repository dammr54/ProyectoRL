import pickle

def load(filename):
    file = open(filename, "rb")
    data = pickle.load(file)
    file.close()
    return data

def dump(filename, obj):
    file = open(filename, "wb")
    pickle.dump(obj, file)
    file.close()