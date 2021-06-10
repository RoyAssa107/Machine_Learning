import numpy as np
import collections
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math

N = 1000


# Function that plots a 3D normal distribution according to input mean and variance
def Plot_3D_Distribution(mean_Vector, variance_Matrix, alpha):
    # Create grid and multivariate normal
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    # rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
    rv = multivariate_normal(mean_Vector, variance_Matrix)

    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
    ax.plot_surface(X, Y, rv.pdf(pos) * alpha, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap='coolwarm')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')


# Function that generates N data points according to different given distributions
def Generate_Points(mean_vectors, variance_matrices):
    generated_points = np.array([[0] * 3])  # [x_value ,y_value ,label ] for each point we generate
    points_labels = []
    distributions = [1, 2, 3]
    probabilities = [alpha1, alpha2, alpha3]

    # Randomizing Zi (= distribution #) for each of the N points
    Z = np.random.choice(distributions, N, p=probabilities)

    # A dictionary that contains the amount of generated points in each distribution
    distribution_counter = collections.Counter(Z)
    distribution_counter = collections.OrderedDict(sorted(distribution_counter.items()))

    # Generating points according to the chosen distribution
    for distribution in distribution_counter:
        M = distribution_counter[distribution]  # The amount of points we generate from the current distribution
        currentPoints = np.random.multivariate_normal(mean_vectors[distribution],
                                                      variance_matrices[distribution], M)

        currentLabel = np.array([[distribution] * M]).reshape(M, 1)
        currentPoints = np.append(currentPoints, currentLabel, axis=1)

        # Appending the new label's
        points_labels.append([distribution] * M)
        generated_points = np.append(generated_points, currentPoints, axis=0)

    print(distribution_counter)
    print(dict(distribution_counter).keys())
    print(dict(distribution_counter).values())
    return generated_points[1:, :], points_labels, distribution_counter


# Function that calculates the pdf of a given normal distribution with specified point (2D array)
def norm_pdf_multivariate(point, mu, sigma):
    size = len(point)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        # Calculating the pdf at the point according to the given normal distribution
        norm_const = 1.0 / (math.pow((2 * np.pi), float(size) / 2) * math.pow(det, 1.0 / 2))
        x_mu = point - mu
        var_Inv = np.linalg.inv(sigma)
        result = math.pow(math.e, -0.5 * (np.dot(np.dot(x_mu, var_Inv), x_mu.reshape(size, 1))))

        val = norm_const * result
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")


# Function that computes GMM clustering algorithm
def Compute_GMM(testingPoints):
    W = np.zeros((N, 3))
    mu = np.zeros((3, 2))  # holds the mean vector of each distribution (referred as [1-3])
    sigma = np.array([[[5, 0], [0, 2]], [[7, 0], [0, 1]],
                      [[8, 0], [0, 2]]])  # holds the variance matrix of each distribution (referred as [1-3])

    tempProb = [0.33, 0.33, 0.34]  # initialize random probabilities of distributions
    tempMU = np.array([[2, 0], [0, 2], [-2, 2]])
    tempSIGMA = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]],
                          [[1, 0], [0, 1]]])  # holds the variance matrix of each distribution (referred as [1-3])
    count = 1

    while count != 100:
        probabilities = tempProb
        mu = tempMU
        sigma = tempSIGMA

        # E-Step
        for i, point in enumerate(testingPoints):
            denominator = [norm_pdf_multivariate(point, mu[m], sigma[m]) * probabilities[m] for m in range(3)]
            denominator = np.array(denominator)

            # Looping on all the 3 different distributions in order to calculate pdf at specific point
            for j in range(3):
                W[i, j] = denominator[j] / np.sum(denominator)

        # M-Step

        for j in range(3):
            tempProb[j] = np.sum(W[:, j]) / N  # updating heights of normal distributions
            tempMU[j] = np.sum((W[:, j].reshape(1, N) * testingPoints), axis=0) / (
                np.sum(W[:, j]))  # updating mean of distributions

            tmp = testingPoints - tempMU[j]

            for i in range(N):
                tempSIGMA[j] = tempSIGMA[j] + W[i, j] * np.dot(tmp[i], tmp[i])
            # print(1/np.sum(W[:, j]))
            # tempSIGMA[j] = tempSIGMA[j] * (1/np.sum(W[:, j]))
            # print()
            count += 1

    return probabilities, mu, sigma


# Function that computes K-Means clustering algorithm of given testing data
def Compute_K_Means(testingPoints):
    D = 2  # number of features in each data point
    centroids = [testingPoints[N // 3], testingPoints[N // 2], testingPoints[N // 4]]
    clusters = [np.empty((0, D), int), np.empty((0, D), int), np.empty((0, D), int)]

    for i in range(20):
        clusters = [np.empty((0, D), int), np.empty((0, D), int), np.empty((0, D), int)]
        for point in testingPoints:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            closestCentroid = distances.index(min(distances))
            clusters[closestCentroid] = np.append(clusters[closestCentroid], [point], axis=0)

        # updating centroids
        for m in range(len(centroids)):
            centroids[m] = (np.sum(clusters[m], axis=0)) / (clusters[m].shape[0])

    return centroids, clusters


# Function that plots clusters with given centroids
def Plot_Clusters(centroids, clusters, title):
    fig = plt.figure()
    for i in range(len(clusters1)):
        plt.scatter(clusters[i][:, 0], clusters[i][:, 1], label=f'cluster{i + 1}', marker='.')
        plt.scatter(centroids[i][0], centroids[i][1], label=f'centroid{i + 1}', marker='^')
    plt.title(title)


if __name__ == "__main__":
    # distribution 1
    mean_Vector1 = np.array([2, 2])
    variance_Matrix1 = np.array([[1.1, 0], [0, 1.1]])
    alpha1 = 0.5

    # distribution 2
    mean_Vector2 = np.array([-2, 2])
    variance_Matrix2 = np.array([[1.1, 0], [0, 1.1]])
    alpha2 = 0.3

    # distribution 3

    mean_Vector3 = np.array([0, -2])
    variance_Matrix3 = np.array([[1.1, 0], [0, 1.1]])
    alpha3 = 0.2

    # Plotting distribution in a single figure
    # fig = plt.figure()
    # Plot_3D_Distribution(mean_Vector1, variance_Matrix1, alpha1)
    # Plot_3D_Distribution(mean_Vector2, variance_Matrix2, alpha2)
    # Plot_3D_Distribution(mean_Vector3, variance_Matrix3, alpha3)
    # plt.show()

    # Generating 1000 2D points according to the distributions
    mean_Vectors = [0, mean_Vector1, mean_Vector2, mean_Vector3]
    variance_Matrices = [0, variance_Matrix1, variance_Matrix2, variance_Matrix3]

    # Printing points we generated from the 3 different distributions
    np.set_printoptions(precision=5)
    data_points, labels, distributions = Generate_Points(mean_Vectors, variance_Matrices)
    # print(data_points)

    # Results of GMM
    # probabilitiesFound, mu1, sigma1 = Compute_GMM(data_points[:, 0:2])
    # print(probabilitiesFound)
    # print(mu1[1], mean_Vector2)
    # print(sigma1[1], variance_Matrix2)

    centroids1, clusters1 = Compute_K_Means(data_points[:, 0:2])
    clusters1.sort(key=lambda lst: len(lst))

    # Plotting original data with labels before K-Means
    clustersBefore = [0] * 3
    centroidsBefore = [0] * 3
    sum1 = 0
    for i in range(3):
        clustersBefore[i] = data_points[sum1:sum1 + distributions[i + 1], 0:2]
        sum1 += distributions[i + 1]
        centroidsBefore[i] = np.sum(clustersBefore[i], axis=0) / distributions[i+1]

    clustersBefore.sort(key=lambda lst: len(lst))
    Plot_Clusters(centroids1, clustersBefore, "Original data before K-means")
    plt.legend()

    # Plotting data with labels after K-Means
    Plot_Clusters(centroids1, clusters1, "Clusters after running K-means")
    plt.legend()
    plt.show()
