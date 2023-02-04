import numpy as np
import time
start_time = time.time()

# Defining the initial guess, the pre-calculated x and y components of the gradient,
# a reasonable step size, the stepsize multiplied by the gradient, and a list that
# will be used to re-calculate the gradient under the gradient function.

guess = [4,5]
stepsize=0.01
n=[2,2]
count = 0

# Redefining every list except 'n' as an array to be able to use numpy commands
# in the 'while' loop.

guess = np.array(guess)
n=[2,2]
stepsize=np.array(stepsize) 

# Defining the chosen test-function for this first draft: f(x,y)= x^2 + y^2.

def f(guess):
    return(guess[0]**2+guess[1]**2)

# Defining a function that takes guess as the input and uses the list 'n' to
# recalculate the gra ( x and y components of the gradient ) in the while loop.

def gradient(guess):
    gra=np.multiply(n,guess)
    grad=np.multiply(gra,stepsize)
    return(grad)

# A while loop that requires f to get within 0.0000000001 of our goal.
# The guess is moved in the opposite direction of the gradient at its position
# by an amount intentionally reduced by the stepsize. The new f(guess) value is
# evaluated. Both 'gra' and 'grad' are recalculated according to the new guess-value.
# The print commands display the new x and y coordinates of the point, the new x
# and y components of the gradient, then the same components modified after multiplication
# with the step size, the value of f(x) at the new position, and then the new position
# itself, respectively.

while f(guess) > 0.0000000001:
   grad=gradient(guess)
   subtracted_array = np.subtract(guess, grad)
   print('------------------------------------------------')
   guess = list(subtracted_array)
   f(guess)
   gra=gradient(subtracted_array)
   print(gra)
   grad=gradient(guess)
   count += 1
   print(count)
   print(grad)
   print(f(guess))
   print(guess)
   
print("My program took", time.time() - start_time, "to run")