# Importing numpy.
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import random

#-----------------------------------------------------------------------------#

# Initiate a stop watch.
start_time = time.time()

#-----------------------------------------------------------------------------#

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

#-----------------------------------------------------------------------------#

# Define control paramters (these are explained when they're implemented).
c1,c2,c3,nx,ny,w,err,n = 0.8,0.5,15,2,2,0.5,0.1,10

# Define search area.
xmin, xmax, ymin, ymax = -3, 5, -3, 5

# Initialise a variable to count the iterations of the loop.
step = 0

#-----------------------------------------------------------------------------#

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
        
positions = np.array([positions])

#-----------------------------------------------------------------------------#

# Calculate the values of the function at the first positions.
fvalues = f(positions[step])
fvalues = np.array([fvalues])

#-----------------------------------------------------------------------------#

# Evaluate the personal best at step 0.
personalbest=positions[step]

#-----------------------------------------------------------------------------#

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

#-----------------------------------------------------------------------------#

# Initiate the found roots array.
foundroots = np.empty((0,1)).reshape(-1,2)

#Initiate the velocities array.
velocities = np.empty((nx*ny,2)).reshape(-1,2)

#-----------------------------------------------------------------------------#

# Big loop.
while ((len(foundroots)!=n)&(step<np.array(50000))).any()==True:
      
      step += 1
    
      # Generate stochastic kicks to hopefully converge faster.
      r1, r2 = random.rand(), random.rand()
      
      #-----------------------------------------------------------------------------#
      if step ==1:
          # Generate the new velocities.
          velocities = w*velocities+r2*c2*(gbest-positions[step-1])+r1*c1*(personalbest-positions[step-1])-c3*(gworst-positions[step-1])
          
          # Move the particles one step.
          position = positions[step-1]+velocities
      else:
          keep = w*keep + r2*c2*(gbest-positions[step-1]) + r1*c1*(personalbest-positions[step-1]) - c3*(gworst-positions[step-1])
          
          # Move the particles one step.
          position = positions[step-1] + velkeep + velchange
      #-----------------------------------------------------------------------------#
    
      # Move the particles one step.
      position = positions[step-1]+velocities
      positions = np.vstack([positions, np.array([position])])
      
      #-----------------------------------------------------------------------------#
      
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
      keepx = np.where(((x<=xmax)&(x>=xmin)),1,0)
      keepy = np.where(((y<=ymax)&(y>=ymin)),1,0)
    
      # Step 4: Reconstruct the x and y arrays for the fugitives.
      changex = changexmin*xmin + changexmax*xmax
      changey = changeymin*ymin + changeymax*ymax
    
      # Step 5: Reconstruct the positions array for the fugitives.
      change = np.hstack([changex, changey])
    
      # Step 6: Reconstruct the positions array for the fugitives.
      keep = np.hstack([keepx, keepy])
    
      # Step 7: Substitute the original positions of the keepers.
      keep_pos = keep*positions[step]
    
      # Step 8: Reassign positions.
      positions[step] = keep_pos + change
      
      #-----------------------------------------------------------------------------#
      
      # Calculate the values of the function at this step.
      fvalue= f(positions[step])+G(foundroots)
      fvalues = np.vstack([fvalues,fvalue[None,:,:]])
      
      #-----------------------------------------------------------------------------#
      
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
    
      #-----------------------------------------------------------------------------#
    
      # Evaluate the personal best.
      keep = np.where(f(personalbest)>fvalues[step],1,0)
      change = np.where(f(personalbest)<fvalues[step],1,0)
      
      C = change*positions[step]
      K = keep*personalbest
      
      personalbest = C+K
      
      #-----------------------------------------------------------------------------#
      
      # To implement boundary bouncing, first we isolate the elements of the
      # velocities of each particle that needs to change.
      changex = changexmin + changexmax
      changey = changeymin + changeymax
      change = np.hstack([changex, changey])
      
      # Now we invert the relevant velocity component of the fugitives.
      velchange = change*-velocities
      velkeep = keep*velocities
      
      # Check if a root was found in this iteration.
      if(fvalues[step]<err).any()==True:
          
          # The found-root procedure is carried out.
          foundroots = np.vstack((foundroots, gbest))
          print('Root found!')
      
      if step%10000==0:
          print(positions[step])
          print(gbest)

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

plt.show()