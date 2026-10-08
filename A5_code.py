import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Question 1:

"""
#Defining the grid
x = np.linspace(-3,3,400)
y = np.linspace(-3,3,400)
X,Y = np.meshgrid(x,y)


#Psi
rx1 = (X-1)**2+Y**2
rx2 = (X+1)**2 + Y**2
psi = -0.5 * np.log(rx1) -0.5* np.log(rx2)


#Velocity field
u = -Y/rx1 - Y/rx2
v = (X-1)/rx1 + (X+1)/rx2

#Plotting

plt.figure(figsize=(8, 8))

#Stream function
plt.contour(X, Y, psi, levels=20, colors='red', linestyles='dashed', alpha=0.5)
#Velocity field 
plt.streamplot(X, Y, u, v, color='blue', density=1.5)
#Figure 8 flow (separatrix)
plt.contour(X, Y, psi, levels=[0], colors='black',linestyles = 'dashed', linewidths=3)
#Stagnation point at origin
plt.plot(0, 0, 'go', markersize=10)
#Centre of vortex
plt.scatter([1, -1], [0, 0], color='red', s=100)

plt.title("Fluid Trajectories (Blue) vs Streamfunction Iso-Contours (Red)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


#To check numerically we can compare the contour against a specific movement path
#Function defines the velocity field at different times t
def velocity_field(t, pos):
    px, py = pos
    d1 = (px - 1)**2 + py**2 + 1e-9
    d2 = (px + 1)**2 + py**2 + 1e-9
    vx = -py/d1 - py/d2
    vy = (px - 1)/d1 + (px + 1)/d2
    return [vx, vy]

#Finds the solution for a given IV
start_pos = [0.0, 1.5]
t_span = (0, 10)
trajectory = solve_ivp(velocity_field, t_span, start_pos,
                       t_eval=np.linspace(0, 10, 100),rtol = 1e-9,atol = 1e-12)

# Calculate psi values along the integrated trajectory
traj_x, traj_y = trajectory.y
psi_values = -0.5 * np.log((traj_x - 1)**2 + traj_y**2) - 0.5 * np.log((traj_x + 1)**2 + traj_y**2)

#The difference between the velocity field and stream function
print(f"Max deviation in psi: {np.max(np.abs(psi_values - psi_values[0])):.6e}")
"""




#Question 2: 
#Up until the #Poincare section, will complete part a

"""
#We first want to define the change a point goes through when a vortex is on:
def rotation(x,y,x0,T):
    substeps = 25
    r2 = (x-x0)**2 + y**2
    d_theta = T/(2*r2)

    angles = np.linspace(0,d_theta,substeps)
    
    #We rotate x and y
    x_new = np.cos(angles)*(x-x0)- np.sin(angles)*y + x0
    y_new = np.sin(angles)*(x-x0) + np.cos(angles)*y 

    return x_new, y_new


# Parameters (can change these)
T = 2.4
n_steps = 25 # Number of full periods

#Initialisation
start_pos = [2, 2] 
curr_x = start_pos[0]
curr_y = start_pos[1]
x_traj = [curr_x]
y_traj = [curr_y]

for _ in range(n_steps):
    #Vortex at x= 1 
    x_pts, y_pts = rotation(curr_x,curr_y,1,T)
    x_traj.extend(x_pts)
    y_traj.extend(y_pts)
    curr_x,curr_y = x_pts[-1],y_pts[-1]

    #Switch to vortex at x= -1
    x_pts,y_pts = rotation(curr_x,curr_y,-1,T)
    x_traj.extend(x_pts)
    y_traj.extend(y_pts)
    curr_x,curr_y = x_pts[-1],y_pts[-1]

# Plotting the arcs
plt.figure(figsize=(8, 8))
plt.plot(x_traj, y_traj, 'b-', lw=1, alpha=0.8)
plt.scatter([1, -1], [0, 0], color='red', s=100, zorder=5)
plt.title(f"Blinking Vortex Trajectory (T={T})")
plt.gca().set_aspect('equal')
plt.show()


#Poincare section

def Poincare(T,n_periods):
    n_particles = 50 #We will follow 50 particles through T
    x = np.linspace(-3,3,n_particles) #We space them along the x-axis
    y = np.zeros(n_particles)
    
    #Plotting
    plt.figure(figsize=(10, 8))
    vortex_pos = [1.0, -1.0]
    colours = (['red'] * 15) + (['limegreen'] * 15) + (['royalblue'] * 20)
    
    #The math
    for _ in range(n_periods):
        for x0 in vortex_pos:
            
            r2 = (x-x0)**2 + y**2
            d_theta = T/(2*r2)

            #We rotate x and y
            x_new = np.cos(d_theta)*(x-x0)- np.sin(d_theta)*y + x0
            y_new = np.sin(d_theta)*(x-x0) + np.cos(d_theta)*y

            x, y = x_new, y_new

        plt.scatter(x,y,s=0.2,c=colours, alpha = 0.5)

    plt.scatter(vortex_pos, [0, 0], color='red', marker='o', s=100, label='Vortices')
    plt.title(f"Poincaré Section(T={T})")
    plt.gca().set_aspect('equal')
    plt.show()
"""

#Question 3:
"""

#A function that makes the plot look good when two points are separated greater than a threshold
def refine_contour(x, y):
    threshold = 0.10
    max_points = 150000
    
    # Calculate distances between all adjacent points 
    dx = np.append(np.diff(x), x[0] - x[-1])
    dy = np.append(np.diff(y), y[0] - y[-1])
    dist = np.sqrt(dx**2 + dy**2)

    too_far = dist > threshold
    
    if not np.any(too_far):
        return x, y
        
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    
    mid_x = (x + x_next) / 2.0
    mid_y = (y + y_next) / 2.0
    
    new_x = []
    new_y = []
    
    for i in range(len(x)):
        new_x.append(x[i])
        new_y.append(y[i])

        if too_far[i] and (len(new_x) < max_points):
            new_x.append(mid_x[i])
            new_y.append(mid_y[i])
            
    return np.array(new_x), np.array(new_y)



#Simulates dye in a liquid for the alternating dual vortex and finding the distance over t
#Stop_t must be a multiple of T

def dye_move_alt(T, n_periods,stop_t):

    theta = np.linspace(0, 2*np.pi, 500) #the droplet of dye is 500 points around a circle
    x = 0.1 + 0.1*np.cos(theta)
    y = 0.1 + 0.1*np.sin(theta)
    
    #Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    #Set up
    vortex_pos = [1.0, -1.0]
    lengths = []
    total_periods = int(stop_t/T)
    total_half_periods = total_periods*2
    times = np.arange(1, total_periods + 1)
    colours = plt.cm.viridis(np.linspace(0, 1, total_periods))

    
    
    #The math
    for i in range(total_half_periods):

        x0 = vortex_pos[0] if (i % 2 == 0) else vortex_pos[1]
            
        r2 = (x-x0)**2 + y**2
        d_theta = T/(2*r2)

        #We rotate x and y
        x_new = np.cos(d_theta)*(x-x0)- np.sin(d_theta)*y + x0
        y_new = np.sin(d_theta)*(x-x0) + np.cos(d_theta)*y

        x, y = x_new, y_new

        x,y = refine_contour(x,y)

        #Length calculation after one whole period
        if i % 2 ==1:
            total_length = np.sum(np.sqrt(np.diff(x)**2+np.diff(y)**2))
            total_length += np.sqrt((x[0]-x[-1])**2+(y[0]-y[-1])**2)
            lengths.append(total_length)

    #Plotting advection
    x_plot = np.append(x,x[0])
    y_plot = np.append(y,y[0])
    ax1.plot(x_plot, y_plot, color=colours[-1], lw=0.15,alpha=0.9)
    ax1.scatter(vortex_pos, [0, 0], color='red', marker='o', s=100)
    ax1.set_title(f"Advection Snapshot (t= {stop_t})")
    ax1.set_aspect('equal')
    ax1.legend()

    #plotting length over time
    ax2.plot(times, lengths, 'o-', color='black')
    ax2.set_xlabel("Periods")
    ax2.set_ylabel("Length L(t)")
    ax2.set_title("Contour Length Growth")
    ax2.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.show()
    

#Simulates dye in a liquid for the stationary dual vortex

#We take our velocity function from part 1, but changed slightly
def velocity_field(t, state):
    m = len(state)//2
    px, py = state[:m],state[m:]
    
    d1 = (px - 1)**2 + py**2 
    d2 = (px + 1)**2 + py**2 
    vx = -py/d1 - py/d2
    vy = (px - 1)/d1 + (px + 1)/d2
    
    return np.concatenate([vx, vy])

    
def dye_move_stat(T,n_periods,stop_t):
    
    T_max = T*n_periods
    stop_loop = int(round(stop_t / T))
    
    n_points = 500
    theta = np.linspace(0, 2*np.pi, n_points)
    x = 0.1 + 0.1*np.cos(theta)
    y = 0.1 + 0.1*np.sin(theta)

    fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    
    initial_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)) + np.sqrt((x[0]-x[-1])**2 + (y[0]-y[-1])**2)
    lengths = [initial_length]
    times = [0.0]

    #Solve:
    for i in range(stop_loop+1):
        initial_state = np.concatenate([x, y])

        sol = solve_ivp(velocity_field, (0, T), initial_state, method='RK23', rtol=1e-5)

        m = len(sol.y) // 2
        x, y = sol.y[:m, -1], sol.y[m:, -1]

        #Finding length
        total_length = np.sum(np.sqrt(np.diff(x)**2+np.diff(y)**2))
        total_length += np.sqrt((x[0]-x[-1])**2+(y[0]-y[-1])**2)
        
        lengths.append(total_length)
        times.append((i + 1) * T)

    #Plotting the length
    ax2.plot(times, lengths, 'o-', color='black')
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Length L(t)")
    ax2.set_title(f"Contour Length Growth (t = {stop_t})")
    ax2.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.show()

"""
