# Importing numpy.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time
from numpy import random

# Initiate a stop watch.
start_time = time.time()

# Define a function.
def f(inarray):
    x,y = np.hsplit(inarray,2)
    u = -np.cos(y)+2*y*np.cos(y**2)*np.cos(2*x)
    v = -np.sin(x)+2*np.sin(y**2)*np.sin(2*x)
    return(u**2+v**2)

def f1darr(inarray):
    x,y = np.hsplit(inarray,2)
    u = -np.cos(y)+2*y*np.cos(y**2)*np.cos(2*x)
    v = -np.sin(x)+2*np.sin(y**2)*np.sin(2*x)
    return(u**2+v**2)

def G(foundroots):
    running_tally = 0.0 
    if len(foundroots) != 0:
        x,y=np.hsplit(positions[step-1],2)
        b=0
        for i in range(len(foundroots)):
            x_root,y_root=np.split(foundroots[int(i),:],2)
            gauss=1.0/800.0*(np.exp(-(x-x_root)**2)/0.001**2)*1.0/800.0*(np.exp(-(y-y_root)**2)/0.001**2)
            running_tally += gauss
            b += 1
    return(running_tally)

# Define control paramters (these are explained when they're implemented).
c1,c2,c3,nx,ny,w,err,n = 0.8,0.5,1.5,2,2,0.5,0.6,10

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

# Calculate the values of the function at the first positions.
fvalues = f(positions)

# Generate random initial velocities.
velocities = np.random.rand(nx*ny,2)

personalbest=positions

# Count increased.
step += 1

# First step must be done manually.
position = positions + velocities

# Calculate the fvalues, and if necessary, add a Gaussian bump over a
# found root.
fvalue = f(positions)
fvalues = np.stack([fvalues,fvalue])

# Step 1: Separate x and y coordinates.
x,y = np.hsplit(positions,2)

# Step 2: Find fugitives.
changexmin = np.where(x<xmin,1,0)
changexmax = np.where(x>xmax,1,0)
changeymin = np.where(y<ymin,1,0)
changeymax = np.where(y>ymax,1,0)

# Step 3: Find keepers.
keepx = np.where(((x<xmax)&(x>xmin)),1,0)
keepy = np.where(((y<ymax)&(y>ymin)),1,0)

# Step 4: Reconstruct the x and y arrays for the fugitives.
changex = changexmin*xmin + changexmax*xmax
changey = changeymin*ymin + changeymax*ymax

# Step 5: Reconstruct the positions array for the fugitives.
change = np.hstack([changex, changey])

# Step 6: Reconstruct the positions array for the fugitives.
keep = np.hstack([keepx, keepy])

# Step 7: Substitute the original positions of the keepers.
keep = keep*positions

# Step 8: Reassign positions.
position = keep + change

# Combine the 2D arrays of both the first positions and the second positions
# into the array for the position history. This is the step where the positions
# array becomes 3D.
positions = np.stack([positions,position])

# Initiate the found roots array.
foundroots = np.empty((0,1)).reshape(-1,2)

# Calculate the fvalues, and if necessary, add a Gaussian bump over a
# found root.
fvalue = f(positions[step])+G(foundroots)
fvalues = np.stack([fvalues,fvalue])

# Big loop.
while ((len(foundroots)!=n)&(step<np.array(100000))).any()==True:

      # Evaluate the global best.
      gbestindex = np.unravel_index(np.argmin(fvalues), fvalues.shape)
      last_index = len(gbestindex) - 1
      gbestindex = gbestindex[:last_index]
      gbest = positions[gbestindex]

      # Evaluate the global worst.
      gworstindex = np.unravel_index(np.argmax(fvalues), fvalues.shape)
      lastindex = len(gworstindex) - 1
      gworstindex = gworstindex[:lastindex]
      gworst = positions[gworstindex]
    
      # Evaluate the personal best.
      keep = np.where(f(personalbest)>fvalues[step],1,0)
      change = np.where(f(personalbest)<fvalues[step],1,0)
      
      C = change*positions[step]
      K = keep*personalbest
      
      personalbest = C+K
      
      # Generate stochastic kicks to hopefully converge faster.
      r1, r2 = random.rand(), random.rand()
      
      # Generate the new velocities.
      velocities = w*velocities+r2*c2*(gbest-positions[step])+r1*c1*(personalbest-positions[step])-c3*(gworst-positions[step])
     
      # Move the particles one step.
      position = positions[step]+velocities
      positions = np.vstack([positions,position[None,:,:]])
      
      # After movement, check if particles escaped the search area. If so, return
      # them to the boundary. To implement 'bouncing' off of the boundary,
      # change the sign of their relevant velocity component.
      
      # Step 1: Separate x and y coordinates.
      x,y = np.hsplit(positions[step],2)
    
      # Step 2: Find fugitives.
      changexmin = np.where(x<xmin,1,0)
      changexmax = np.where(x>xmax,1,0)
      changeymin = np.where(y<ymin,1,0)
      changeymax = np.where(y>ymax,1,0)
    
      # Step 3: Find keepers.
      keepx = np.where(((x<xmax)&(x>xmin)),1,0)
      keepy = np.where(((y<ymax)&(y>ymin)),1,0)
    
      # Step 4: Reconstruct the x and y arrays for the fugitives.
      changex = changexmin*xmin + changexmax*xmax
      changey = changeymin*ymin + changeymax*ymax
    
      # Step 5: Reconstruct the positions array for the fugitives.
      change = np.hstack([changex, changey])
    
      # Step 6: Reconstruct the positions array for the fugitives.
      keep = np.hstack([keepx, keepy])
    
      # Step 7: Substitute the original positions of the keepers.
      keep = keep*positions[step]
    
      # Step 8: Reassign positions.
      positions[step] = keep + change
      
      # changearr = np.hstack([changex, changey])
      # vel = changearr*-velocities
      # keeparr = np.hstack([keepx, keepy])
      # velocities = changearr + keeparr
      
      # Check if a root was found in this iteration.
      if(fvalues[step]<err).any()==True:
          
          # The found-root procedure is carried out.
          foundroots = np.vstack((foundroots, gbest))
          print('Root found!')
      
      # Calculate the values of the function at this step.
      fvalue= f(positions[step])+G(foundroots)
      fvalues = np.vstack([fvalues,fvalue[None,:,:]])
      
      # Evaluate the global best.
      gbestindex = np.unravel_index(np.argmin(fvalues), fvalues.shape)
      last_index = len(gbestindex) - 1
      gbestindex = gbestindex[:last_index]
      gbest = positions[gbestindex]
      
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
      
      # if ((conv_arr_min < f(gbest))&(f(gbest) < conv_arr_max)).all()==True:# should be the f values of global best.
      #      c2=1
      #      w=0.5
      #      c1=0.5

# Print the time the program took.
print("My program took", time.time() - start_time, "to run")      
# Plot the final positions.
x,y=np.hsplit(positions[step],2)
plt.xlim(-3,3), plt.ylim(-3,3), plt.scatter(x,y)

# Prepare the data for animation.
fig, ax = plt.subplots()
ax.set_xlim([xmin*2, xmax*2])
ax.set_ylim([ymin*2,ymax*2])

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