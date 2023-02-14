import numpy as np
import matplotlib.pyplot as plt

N = 100

cube = np.zeros((N,N,N))

cube[:,0,0] = 1
cube[0,:,0] = 2
cube[0,0,:] = 3
cube[0,0,0] = 5

n = N//3

cube[n:-n,n,n] = 1
cube[n,n:-n,n] = 2
cube[n,n,n:-n] = 3
cube[n,n,n] = 5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = []
ys = []
zs = []
color = []
for x in range(N):
    for y in range(N):
        for z in range(N):
            if cube[x,y,z] != 0:
                xs.append(x)
                ys.append(y)
                zs.append(z)
                if cube[x,y,z] == 1:
                    color.append('blue')
                elif cube[x,y,z] == 2:
                    color.append('red')   
                elif cube[x,y,z] == 3:
                    color.append('green')
                elif cube[x,y,z] == 5:
                    color.append('black')

ax.scatter(xs, ys, zs, c=color, cmap='viridis', s=100)

plt.show()