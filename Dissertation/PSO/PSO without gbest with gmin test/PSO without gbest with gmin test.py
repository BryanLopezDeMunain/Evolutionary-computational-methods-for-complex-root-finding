# Importing numpy.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time

# Initiate a stop watch.
start_time = time.time()

# Define the function at hand. For efficiency, the function will be coded to act
# on a full array, and to output a full array.
def f(array):
    dummy = array**2
    dummy1, dummy2 = np.hsplit(dummy,2)
    output = dummy1 + dummy2
    return(output)

# Define control paramters (these are explained when they're implemented).
c1,nx,ny,w,err = 0.5,2,2,0.2,0.000001

# Define search area.
xmin, xmax, ymin, ymax = -10**6, 10**6, -10**6, 10**6

# Initialise a variable to count the iterations of the loop.
step = 0

# Generate grid of initial positions. nx is size of the horizontal side of the
# grid of particles. ny is the analogous quantity for the vertical side.
x = np.linspace(xmin,xmax,nx)
y = np.linspace(ymin,ymax,ny)

# Initialise an empty list to store the positions.
positions = np.empty((0,1))

# Generate grids of coordinates for all particles at first instance.
for i in x:
    for j in y:
        positions = np.append(positions,(i,j)).reshape(-1,2)

# Calculate the values of the function at the first positions. This is where the function
# is hard-coded; generalisation considerations are necessary.
fval = positions**2
f1,f2 = np.hsplit(fval,2)
fvalues = f1+f2

# Generate random initial velocities.
velocities = np.random.rand(nx*ny,2)

# First step must be done manually.
position = positions + velocities

# Combine the 2D arrays of both the first positions and the second positions
# into the array for the position history. This is the step where the positions
# array becomes 3D.
positions = np.stack([positions,position])

# Count increased.
step += 1

# The f(x,y) values of the new positions are calculated manually. Two arrays
# of fvalues is needed to calculate the personal best so this is done outside
# the loop.
fval = positions[step]**2
f1,f2 = np.hsplit(fval,2)
fvalue = f1+f2
fvalues = np.stack([fvalues,fvalue])

# Define two arrays for stop condition. Here the variable err is our accepted
# error margin.
zeros_min = np.zeros((nx*ny,1))
zeros_min -= err
zeros_max = np.zeros((nx*ny,1))
zeros_max += err

# Calculate the personal best manually once. God bless vectorised operations.
keep = np.where(fvalues[step-1]>fvalues[step],0,1)
change = np.where(fvalues[step-1]<fvalues[step],0,1)

C = change*positions[step]
K = keep*positions[step-1]

personalbest = C+K

# Big loop.
while ((zeros_min<fvalues[step])&(fvalues[step]<zeros_max)).any()==False:
     
      # Evaluate the personal best.
      keep = np.where(f(personalbest)>fvalues[step],0,1)
      change = np.where(f(personalbest)<fvalues[step],1,0)
      
      C = change*positions[step]
      K = keep*personalbest
      
      personalbest = C+K
         
      # Generate the new velocities.
      velocities = w*velocities+c1*(personalbest-positions[step])
     
      # Move the particles one step.
      p = np.empty((nx*ny,2))
      p = positions[step]+velocities
      positions = np.vstack([positions, p[None, :, :]])
      
      # Calculate the values of the function at this step.
      fval = positions[step]**2
      f1,f2 = np.hsplit(fval,2)
      fvalue = f1+f2
      fvalues = np.vstack([fvalues,fvalue[None,:,:]])
      
      # Keep count of how many iterations have occured.
      step += 1

# Print the time the program took.
print("My program took", time.time() - start_time, "to run")      
# Plot the final positions.
x,y=np.hsplit(positions[step],2)
plt.xlim(-3,3), plt.ylim(-3,3), plt.scatter(x,y)

# Prepare the data for animation.
fig, ax = plt.subplots()
ax.set_xlim([xmin*2, xmax*2])
ax.set_ylim([ymin*2,ymax*2])

gbestindex = np.unravel_index(np.argmin(fvalues, axis=None), fvalues.shape) # This alone returns
# (4,3,0) for example, which would give us just the x coordinate when we call it in positions
# so the following two lines are to remove the third element using slicing.
last_index = len(gbestindex) - 1
gbestindex = gbestindex[:last_index]
gbest = positions[gbestindex]
fig = ax.scatter(positions[:,:,0],positions[:,:,1]), plt.title(f'Steps:{step}, Time taken:{np.round(time.time()-start_time,6)}s, Root at: {np.round(gbest,4)}')


def animate(i):
    im = plt.scatter(positions[i,:,0],positions[i,:,1])
    return im,

ani = FuncAnimation(fig, animate, repeat=True, frames=len(positions) - 1, interval=50,blit=False)

#To save the animation using Pillow as a gif
# writer = animation.PillowWriter(fps=15,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# ani.save('scatter.gif', writer=writer)

plt.show()