# Importing numpy.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time
from numpy import random
import warnings

warnings.filterwarnings('ignore')

# Initiate a stop watch.
start_time = time.time()

# Define the function at hand. For efficiency, the function will be coded to act
# on a full array, and to output a full array.
# def f(array):
#     dummy = array**2
#     dummy1, dummy2 = np.hsplit(dummy,2)
#     output = dummy1 + dummy2
#     return(output)

# Define a useful function.
def f(inarray):
    x,y = np.hsplit(positions[step],2)
    u = -np.cos(y)+2*y*np.cos(y**2)*np.cos(2*x)
    v = -np.sin(x)+2*np.sin(x**2)*np.sin(2*x)
    return(u**2+v**2)

def f1darr(inarray):
    x,y = np.hsplit(inarray,2)
    u = -np.cos(y)+2*y*np.cos(y**2)*np.cos(2*x)
    v = -np.sin(x)+2*np.sin(x**2)*np.sin(2*x)
    return(u**2+v**2)

# Define control paramters (these are explained when they're implemented).
c1,c2,nx,ny,w,err,n,conv =1,1,2,2,0.2,0.1,60,0.1

# Define search area.
xmin, xmax, ymin, ymax = -4, 5, -2, 5

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
x,y = np.hsplit(positions,2)
u = -np.cos(y)+2*y*np.cos(y**2)*np.cos(2*x)
v = -np.sin(x)+2*np.sin(x**2)*np.sin(2*x)
fvalues = u**2 + v**2

# Generate random initial velocities.
velocities = np.random.rand(nx*ny,2)

# First step must be done manually.
p = positions + velocities

# First, isolate only the positions we want to deal with. 
x,y = np.hsplit(p,2)

# Now, initialise arrays for the boundaries.
zeros = np.zeros((nx*ny,1))
xminarr = zeros + xmin
xmaxarr = zeros + xmax
yminarr = zeros + ymin
ymaxarr = zeros + ymax

# We will use our comparison framework to isolate and replace positions
# that exceed the boundaries. We now define our relevant comparison arrays.
changexmin = np.where(x < xmin,1,0)
changexmax = np.where(x > xmax,1,0)
changeymin = np.where(y < ymin,1,0)
changeymax = np.where(y > ymax,1,0)
# seems to work but might be issues with the 1s and 0s.

# Now we isolate the elements of the array that meet the boundary conditions. 
keepx = np.where((xmaxarr > x)&(x > xminarr).any(),1,0)
keepy = np.where((ymaxarr > x)&(x > xminarr).any(),1,0)

x = keepx*x + changexmin*xminarr + changexmax*xmaxarr
y = keepy*y + changeymin*yminarr + changeymax*ymaxarr

p=np.hstack([x,y])
changex = changexmin+changexmax
changey = changeymin+changeymax

changearr = np.hstack([changex, changey])
vel = changearr*-velocities
keeparr = np.hstack([keepx, keepy])
velocities = changearr + keeparr

# Combine the 2D arrays of both the first positions and the second positions
# into the array for the position history. This is the step where the positions
# array becomes 3D.
positions = np.stack([positions,p])

# Count increased.
step += 1

# The f(x,y) values of the new positions are calculated manually. Two arrays
# of fvalues is needed to calculate the personal best so this is done outside
# the loop.
fvalue = f(positions)
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

# Evaluate the global best.
gbestindex = np.unravel_index(np.argmin(fvalues, axis=None), fvalues.shape) # This alone returns
# (4,3,0) for example, which would give us just the x coordinate when we call it in positions
# so the following two lines are to remove the third element using slicing.
last_index = len(gbestindex) - 1
gbestindex = gbestindex[:last_index]
gbest = positions[gbestindex]

# Initiate the found roots array.
foundroots = np.empty((0,1))

# Big loop.
while len(foundroots) != n:
      
      # Evaluate the global best.
      gbestindex = np.unravel_index(np.argmin(fvalues, axis=None), fvalues.shape) # This alone returns
      # (4,3,0) for example, which would give us just the x coordinate when we call it in positions
      # so the following two lines are to remove the third element using slicing.
      last_index = len(gbestindex) - 1
      gbestindex = gbestindex[:last_index]
      gbest = positions[gbestindex]
     
      # Evaluate the personal best.
      keep = np.where(f(personalbest)<fvalues[step],1,0)
      change = np.where(f(personalbest)>fvalues[step],1,0)
      
      C = change*positions[step]
      K = keep*personalbest
      
      personalbest = C+K
      
      # Generate stochastic kicks to hopefully converge faster.
      r1, r2 = random.rand(), random.rand()
         
      # Generate the new velocities.
      velocities = w*velocities+r2*c2*(gbest-positions[step])+r1*c1*(personalbest-positions[step])
     
      # Move the particles one step.
      p = np.empty((nx*ny,2))
      p = positions[step]+velocities
      
      # After movement, check if particles escaped the search area. If so, return
      # them to the boundary. To implement 'bouncing' off of the boundary,
      # change the sign of their relevant velocity component.
      
      # First, isolate only the positions we want to deal with. 
      x,y = np.hsplit(p,2)
      
      # Now, initialise arrays for the boundaries.
      zeros = np.zeros((nx*ny,1))
      xminarr = zeros + xmin
      xmaxarr = zeros + xmax
      yminarr = zeros + ymin
      ymaxarr = zeros + ymax
      
      # We will use our comparison framework to isolate and replace positions
      # that exceed the boundaries. We now define our relevant comparison arrays.
      changexmin = np.where(x < xmin,1,0)
      changexmax = np.where(x > xmax,1,0)
      changeymin = np.where(y < ymin,1,0)
      changeymax = np.where(y > ymax,1,0)
      # seems to work but might be issues with the 1s and 0s.
      
      # Now we isolate the elements of the array that meet the boundary conditions. 
      keepx = np.where((xmaxarr > x)&(x > xminarr).any(),1,0)
      keepy = np.where((ymaxarr > x)&(x > xminarr),1,0)
      
      x = keepx*x + changexmin*xminarr + changexmax*xmaxarr
      y = keepy*y + changeymin*yminarr + changeymax*ymaxarr
      
      p = np.hstack([x,y])
      positions = np.vstack([positions, p[None, :, :]])
      changex = changexmin+changexmax
      changey = changeymin+changeymax
      
      # changearr = np.hstack([changex, changey])
      # vel = changearr*-velocities
      # keeparr = np.hstack([keepx, keepy])
      # velocities = changearr + keeparr
      
      # Calculate the values of the function at this step.
      fvalue = f(positions)
      fvalues = np.vstack([fvalues,fvalue[None,:,:]])
      
      # Check if a root was found in this iteration.
      if((gbest+err>positions[step])&(positions[step]>(gbest-err))).all()==True:
          
          # The found-root procedure is carried out.
          foundroots = np.append(foundroots, gbest)
          print('Root found!')
          
      
      if step%10000==0:
          print(positions[step])
          print(gbest)
      # Keep count of how many iterations have occured.
      step += 1
      
      # The convergence variable 'conv' is defined as 10% of the maximum value of the function.
      # This is to scale up convergence when particles get close enough. May be worth removing
      # because it could get the swarm stuck at local minima more often.
      # conv = 0.1*np.max(fvalues)
      
      # conv_arr = np.zeros((1,2))

      # conv_arr_min = conv_arr - conv
      # conv_arr_max = conv_arr + conv
      
      # if ((f1darr(gbest)-conv_arr_min*f1darr(gbest) < f(positions))&(f(positions)+f1darr(gbest)*(conv_arr_max)).all()==True.
      #      c2=1
      #      w=0.5
      #      c1=0.5

# Print the time the program took.
print("My program took", time.time() - start_time, "to run")      
# Plot the final positions.
x,y=np.hsplit(positions[step],2)
plt.xlim(-3,3), plt.ylim(-3,3), plt.scatter(x,y)

# Prepare the data for animation.
plt.axes()
rectangle = plt.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, fc='none', ec="red")
plt.gca().add_patch(rectangle)
plt.axis((xmin*2,xmax*2,ymin*2,ymax*2))
plt.scatter(positions[:,:,0],positions[:,:,1]), plt.title(f'Steps:{step}, Time taken:{np.round(time.time()-start_time,6)}s')
plt.show()


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