import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def space_3D(matrix, X,Y,Z):

    dots_x = []
    dots_y = []
    dots_z = []
    color = []
    threshold = 0.1

    for i, x in enumerate(X[:,0,0]):
        for j, y in enumerate(Y[0,:,0]):
            for k, z in enumerate(Z[0,0,:]):
                if matrix[i,j,k] > threshold:
                    dots_x.append(x)
                    dots_y.append(y)
                    dots_z.append(z)
                    color.append(matrix[i,j,k])
    
    dots_x = np.array(dots_x)
    dots_y = np.array(dots_y)
    dots_z = np.array(dots_z)
    color = np.array(color)

    print(dots_x.shape)
    print(dots_y.shape)
    print(dots_z.shape)
    print(color.shape)

    plt.scatter(dots_x, dots_y, dots_z, c=color, cmap='viridis', linewidth=0.5)                
    plt.colorbar()
    plt.show()

def sum_in_3_directions(matrix, cmap='inferno', save_as=None):

    sum_x = np.sum(matrix, axis=0)
    sum_y = np.sum(matrix, axis=1)
    sum_z = np.sum(matrix, axis=2)

    fig, axs = plt.subplots(1,3, figsize=(15,5))
    im = axs[0].imshow(sum_x, cmap=cmap)
    fig.colorbar(im)
    im = axs[1].imshow(sum_y, cmap=cmap)
    fig.colorbar(im)
    im = axs[2].imshow(sum_z, cmap=cmap)
    fig.colorbar(im)

    if save_as is not None:
        plt.savefig(save_as)
    else:
        plt.show()