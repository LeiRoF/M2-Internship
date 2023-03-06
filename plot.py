import numpy as np
import matplotlib.pyplot as plt

def rotate_data(cloud, v):

    cloud_obs = []
    vx_obs = []
    vy_obs = []
    vz_obs = []

    vx, vy, vz = v

    for i in range(6):
        cloud_obs.append(np.zeros((N, N, N)))
        vx_obs.append(np.zeros((N, N, N)))
        vy_obs.append(np.zeros((N, N, N)))
        vz_obs.append(np.zeros((N, N, N)))

    for x in range(N):
        for y in range(N):
            for z in range(N):

                cloud_obs[0][x,y,z] = cloud[z,x,y]
                cloud_obs[1][x,y,z] = cloud[x,-(z+1),y]
                cloud_obs[2][x,y,z] = cloud[x,y,z]

                cloud_obs[3][x,y,z] = cloud[-(z+1),-(x+1),y]
                cloud_obs[4][x,y,z] = cloud[-(x+1),z,y]
                cloud_obs[5][x,y,z] = cloud[-(x+1),y,-(z+1)]

                vx_obs[0][x,y,z] =  vy[z,x,y]
                vy_obs[0][x,y,z] =  vz[z,x,y]
                vz_obs[0][x,y,z] =  vx[z,x,y]

                vx_obs[1][x,y,z] =  vx[x,-(z+1),y]
                vy_obs[1][x,y,z] =  vz[x,-(z+1),y]
                vz_obs[1][x,y,z] =  vy[x,-(z+1),y]

                vx_obs[2][x,y,z] =  vx[x,y,z]
                vy_obs[2][x,y,z] =  vy[x,y,z]
                vz_obs[2][x,y,z] =  vz[x,y,z]

                vx_obs[3][x,y,z] =  vy[-(z+1),-(x+1),y]
                vy_obs[3][x,y,z] =  vz[-(z+1),-(x+1),y]
                vz_obs[3][x,y,z] =  vx[-(z+1),-(x+1),y]

                vx_obs[4][x,y,z] =  vx[-(x+1),z,y]
                vy_obs[4][x,y,z] =  vz[-(x+1),z,y]
                vz_obs[4][x,y,z] =  vy[-(x+1),z,y]

                vx_obs[5][x,y,z] =  vx[-(x+1),y,-(z+1)]
                vy_obs[5][x,y,z] =  vy[-(x+1),y,-(z+1)]
                vz_obs[5][x,y,z] =  vz[-(x+1),y,-(z+1)]

    return cloud_obs, vx_obs, vy_obs, vz_obs

def plot(cube, ax):
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

N = 20
cloud = np.zeros((N,N,N))
vx = np.zeros((N,N,N))
vy = np.zeros((N,N,N))
vz = np.zeros((N,N,N))

cloud[:,0,0] = 1
cloud[0,:,0] = 2
cloud[0,0,:] = 3
cloud[0,0,0] = 5

n = N//5

vx[n:-n,n,n] = 1
vy[n,n:-n,n] = 2
vz[n,n,n:-n] = 3
vx[n,n,n] = 5
vy[n,n,n] = 5
vz[n,n,n] = 5

n = N//7

pov = np.zeros((N,N,N))
pov[N//2-n,N//2-n,-1] = 5
pov[N//2-n:N//2+n,N//2-n,-1] = 1
pov[N//2-n,N//2-n:N//2+n,-1] = 2

fig = plt.figure()
ax = fig.add_subplot(332, projection='3d')

plot(cloud, ax)
plot(vx, ax)
plot(vy, ax)
plot(vz, ax)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Initial state")

cloud_obs, vx_obs, vy_obs, vz_obs = rotate_data(cloud, (vx, vy, vz))



for i in range(6):
    ax = fig.add_subplot(330 + 4+i, projection='3d') # 
    plot(cloud_obs[i], ax)
    plot(vx_obs[i], ax)
    plot(vy_obs[i], ax)
    plot(vz_obs[i], ax)
    # plot(pov, ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=90., azim=-90)

    match i:
        case 0: ax.set_title("Seen from X+")
        case 1: ax.set_title("Seen from Y-")
        case 2: ax.set_title("Seen from Z+")
        case 3: ax.set_title("Seen from X-")
        case 4: ax.set_title("Seen from Y+")
        case 5: ax.set_title("Seen from Z-")


plt.show()