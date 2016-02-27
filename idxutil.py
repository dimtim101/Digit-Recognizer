import idx2numpy
import numpy as np


def load_mnsit(features_file,labels_file):
    data = idx2numpy.convert_from_file(features_file)
    labels = idx2numpy.convert_from_file(labels_file)

    features = []
    for image in data:
        features.append(image.flatten())

    features = np.array(features)
    
    return features,labels
