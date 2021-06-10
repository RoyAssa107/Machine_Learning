# Computing PCA on MNIST-DATA
# Of EigenFaces
import scipy.io
import numpy as np
import collections
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import accuracy_score

TRAIN = 8
TEST = 3


# Function that taken faces matrix as input
# Prints all faces in a single image (using cv2.imshow)
def Plot_Faces(allFaces):
    # Creating image of 32*32 for each face in order to plot it
    allFaces = allFaces.reshape(len(allFaces), 32, 32)

    # Rotating image clockwise in order to show correctly
    for i in range(len(allFaces)):
        allFaces[i] = np.array(allFaces[i]).transpose()

    img_concate_Verti = np.concatenate(tuple([allFaces[i] for i in range(len(allFaces))]), axis=1)
    cv2.imshow('concatenated_Hori', img_concate_Verti)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()


def Create_Train_Test_Data(allFaces, allLabels):
    training_data = np.empty((0, 1024))
    training_labels = list()

    testing_data = np.empty((0, 1024))
    testing_labels = list()

    # Creating training and test data with corresponding labels
    for i in range(15):
        current_class_train = np.array([allFaces[11 * i + j] for j in range(TRAIN)])
        current_class_test = [allFaces[11 * i + TRAIN + j] for j in range(TEST)]
        training_data = np.append(training_data, current_class_train, axis=0)
        testing_data = np.append(testing_data, current_class_test, axis=0)

        training_labels.extend([allLabels[11 * i + j] for j in range(TRAIN)])
        testing_labels.extend([allLabels[11 * i + TRAIN + j] for j in range(TEST)])

    return training_data, training_labels, testing_data, testing_labels


# Function that computes PCA on given faces
def Compute_PCA(training_data, training_labels, testing_data, testing_labels):
    # Compute mean faces of all
    meanFaces = np.empty((0, 1024))
    for i in range(15):
        mean_face = np.sum(training_data[i * TRAIN:i * TRAIN + TRAIN], axis=0) / 120
        meanFaces = np.append(meanFaces, [mean_face], axis=0)

    # Computing mean face of all faces (classes) in training data
    mean_face_train = np.sum(training_data, axis=0) / len(training_data)
    mean_face_test = np.sum(testing_data, axis=0) / len(testing_data)

    # Printing the mean face
    print(f"The mean face train is:\n {mean_face_train}\n")
    print(f"The mean face test is:\n {mean_face_test}\n")

    # Need to make training data not Affinity (i.e subtract the mean face from each train data)
    training_data_affine = training_data - mean_face_train
    testing_data_affine = testing_data - mean_face_test

    # Creating affine data
    X_data = training_data_affine
    X_data_T = training_data_affine.T

    # Creating the auto-correlation matrix
    A = np.dot(X_data_T, X_data) / N

    # Calculating eigen vectors and values of auto-correlation matrix A
    from scipy.linalg import eigh
    eigen_values, eigen_vectors = eigh(A, eigvals=(1023 - 200, 1023))
    eigen_values = eigen_values[::-1]
    eigen_vectors = eigen_vectors[::-1]

    for K in range(1, 120):
        # Taking K most dominant eigen vectors of matrix A
        error_rate = list()
        dominant_eigen_vectors_Mat = eigen_vectors[:, 0:K]

        # Projecting each training data to the K-dimensional space of the K most dominant eigen vectors
        projected_training_data = np.dot(training_data_affine, dominant_eigen_vectors_Mat)
        projected_testing_data = np.dot(testing_data_affine, dominant_eigen_vectors_Mat)

        output_labels = list()
        for i in range(len(testing_data)):
            dist = np.array([np.linalg.norm(projected_training_data[j] - projected_testing_data[i])
                             for j in range(len(projected_training_data))])
            output_labels.append(training_labels[np.argmin(dist)])

        # print(f"Original testing labels:\n{testing_labels}\n")
        # print(f"PCA's output labels:\n{output_labels}\n")
        print(f"Accuracy rate is: {accuracy_score(testing_labels, output_labels)}")


if __name__ == "__main__":
    # Compute K-Means to MNIST_DATA
    path = "imported path…"  # Change to the path of the MNIST-DATA
    path = "C:\\Users\\Roy\\Desktop\\תואר ראשון בר אילן\\שנה ג\\סמסטר ב\\מבוא ללמידת מכונה\\תרגילי בית\\Targil4\\facesData.mat"

    data = scipy.io.loadmat(path)
    print(f"Keys are: {data.keys()}")

    # Getting faces from data
    faces = np.array(data['faces'])
    N = len(faces)
    print(f"All faces:\n {faces}\n")

    # Getting labels of faces
    faces_labels = np.array(data['labeles']).transpose().flatten()

    # Creating training and test data with corresponding labels
    train_data, train_labels, test_data, test_labels = Create_Train_Test_Data(faces, faces_labels)

    # Compute_PCA(train_data,train_labels)
    Compute_PCA(train_data, train_labels, test_data, test_labels)
