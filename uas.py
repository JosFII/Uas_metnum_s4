import streamlit as st
import base64
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
st.markdown(
'''
<style>
    .stApp {
   background-color: white;
    }
 
       .stWrite,.stMarkdown,.stTextInput,h1, h2, h3, h4, h5, h6 {
            color: purple !important;
        }
</style>
''',
unsafe_allow_html=True
)
st.title("Differential Equation: Runge-Kutta Method for Simulation")
st.header("Flowchart")
st.image("flow.jpg")
st.header("Runge-Kutta Method for Temperature Simulation")
# Parameters
T_env = 25.0  
T0 = 90.0     
k = 0.07      
h = 0.1       
t_final = 60  

# Define the differential equation
def f(t, T):
    return -k * (T - T_env)

# 4th-order Runge-Kutta method
def runge_kutta(f, T0, t_final, h):
    N = int(t_final / h)
    T = np.zeros(N+1)
    t = np.linspace(0, t_final, N+1)
    T[0] = T0

    for n in range(N):
        k1 = f(t[n], T[n])
        k2 = f(t[n] + h/2, T[n] + h/2 * k1)
        k3 = f(t[n] + h/2, T[n] + h/2 * k2)
        k4 = f(t[n] + h, T[n] + h * k3)
        T[n+1] = T[n] + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    return t, T
st.markdown('berikut equation yang akan digunakan:')
st.latex('''
dT/dt=-k*(T-Ts) 
''')
st.write('''T(t)= suhu objek pada waktu t \n
Ts= suhu lingkungan \n
k=konstan''')
st.markdown("equation yang digunakan adalah newton's law of cooling yang berbunyi panas yang hilang proporsional dengan perbedaan suhu")
# Solve the ODE
t, T = runge_kutta(f, T0, t_final, h)
st.markdown('''
berikut hasil dari runge kutta untuk hukum pendinginan newton:''')
dftm=pd.DataFrame({"time":t,"temp":T})
dftm
st.markdown('''
jadi bisa dilihat bahwa dari suhu awal 90 c menjadi suhu 29.97 c \n
berikut gambar plot suhu dengan waktu dari hasil runge kutta:
''')
# Plot the results
plt.plot(t, T, label="Numerical Solution")
plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.title("Newton's Law of Cooling")
plt.grid(True)
plt.legend()
st.pyplot(plt.gcf())
st.markdown('''
jadi bersarkan gambar plot bisa dilihat bahwa suhu menurun secara exponensial berhubung dengan waktu, dimana awalnya berturun dengan 
cepat dan melambat semakin dekat suhu objek mendekati suhu lingkungan.
''')
st.markdown(''' berikut kode yang digunakan \n
  
''')
st.code('''\n
        T_env = 25.0  
        T0 = 90.0     
        k = 0.07      
        h = 0.1       
        t_final = 60  
           
        def f(t, T):
        return -k * (T - T_env)
        def rk(f, T0, t_final, h):
         N = int(t_final / h)
         T = np.zeros(N+1)
         t = np.linspace(0, t_final, N+1)
         T[0] = T0

         for n in range(N):
         k1 = f(t[n], T[n])
         k2 = f(t[n] + h/2, T[n] + h/2 * k1)
         k3 = f(t[n] + h/2, T[n] + h/2 * k2)
         k4 = f(t[n] + h, T[n] + h * k3)
         T[n+1] = T[n] + h/6 * (k1 + 2*k2 + 2*k3 + k4)

         return t, T
''')

st.header("Runge-Kutta Method for Motion Simulation (Mechanics)")
st.markdown('berikut equation yang akan digunakan:')
st.latex('''dx/dt = v''')
st.latex('''dv/dt = F(t)/m''')
st.markdown('''
v= velocity\n
F= force\n
m= massa\n
t= waktu''')
st.markdown('equation yang digunakan adalah newton second law yang berbunyi force sebuah benda sam dengan percepatannya dikali massnya ')
st.markdown('''
berikut hasil dari runge kutta pada hukum kedua newton:
''')
m = 1  # mass in kg
F = 10  # force in N
h = 0.1  # time step in s
t_max = 10  # maximum time in s
steps = int(t_max / h +1)

# Initial conditions
x = 0  # initial position in m
v = 0  # initial velocity in m/s

# Arrays to store results
time = np.linspace(0, t_max, steps)
positions = np.zeros(steps)
velocities = np.zeros(steps)

# Runge-Kutta 4th order method
for i in range(steps):
    positions[i] = x
    velocities[i] = v
    k1x = h * v
    k1v = h * F / m
    k2x = h * (v + 0.5 * k1v)
    k2v = h * F / m
    k3x = h * (v + 0.5 * k2v)
    k3v = h * F / m
    k4x = h * (v + k3v)
    k4v = h * F / m
    x += (k1x + 2*k2x + 2*k3x + k4x) / 6
    v += (k1v + 2*k2v + 2*k3v + k4v) / 6

dffma=pd.DataFrame({"time(s)":time,"position": positions,"velocity":velocities})
dffma
st.markdown('''
jadi dengan force 10N dan massa 1kg, setelah 10 detik telah menempuh jarak 500 m, dan velocity 10m/s
''')
# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(time, positions)
plt.title('Position vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')

plt.subplot(1, 2, 2)
plt.plot(time, velocities)
plt.title('Velocity vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')

plt.tight_layout()
st.pyplot(plt.gcf())
st.markdown('''
jadi bedasarkan grafik position, bisa dilihat bahwa jarak yang yang ditempuh bertambah secara exponential seiring waktu. \n
sedangkan, pada grafik velocity, velocity bertambah secara linear seirng waktu.
''')
st.markdown('kode yang digunakan:')
st.code('''
m = 1  # mass in kg
F = 10  # force in N
h = 0.1  # time step in s
t_max = 10  # maximum time in s
steps = int(t_max / h +1)

x = 0  
v = 0  

time = np.linspace(0, t_max, steps)
positions = np.zeros(steps)
velocities = np.zeros(steps)


for i in range(steps):
    positions[i] = x
    velocities[i] = v
    k1x = h * v
    k1v = h * F / m
    k2x = h * (v + 0.5 * k1v)
    k2v = h * F / m
    k3x = h * (v + 0.5 * k2v)
    k3v = h * F / m
    k4x = h * (v + k3v)
    k4v = h * F / m
    x += (k1x + 2*k2x + 2*k3x + k4x) / 6
    v += (k1v + 2*k2v + 2*k3v + k4v) / 6
''')


st.header("Runge-Kutta Method for Simulating fluid flow or heat conduction over time")
st.markdown('berikut equation yang akan digunakan:')
st.latex("du/dt=α*d^2a/dx^2")
st.markdown('''
α= thermal diffusion rate \n
t= time \n
x∈ [0, L]= position \n
u(x,t)= temperature
''')
st.markdown(''''
berikut suhu batang berdasarkan posisinya dimana baris menunjukan progresi waktu:
'''
)
alpha = 0.01  # Thermal diffusivity
L = 10.0  # Length of the rod
Nx = 100  # Number of spatial points
dx = L / (Nx - 1)  # Spatial step size
dt = 0.1  # Time step size
Nt = 100  # Number of time steps


# Discretized spatial domain
x = np.linspace(0, L, Nx)

# Initial temperature distribution
T = np.zeros(Nx)
T[int(Nx/2)] = 100  # Initial heat pulse at the center

# Function to compute the derivative
def dTdt(T):
    dT = np.zeros_like(T)
    dT[1:-1] = alpha * (T[2:] - 2*T[1:-1] + T[:-2]) / dx**2
    return dT

# Runge-Kutta 4th order method
plt.figure(figsize=(8, 6))
T2=[T.copy()]
for n in range(Nt):
    k1 = dTdt(T)
    k2 = dTdt(T + 0.5 * dt * k1)
    k3 = dTdt(T + 0.5 * dt * k2)
    k4 = dTdt(T + dt * k3)
    T += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    T2.append(T.copy())
    if n % 10 == 0:  # Plot every 10 time steps
        plt.plot(x, T, label=f't = {n*dt:.2f} s')
dftm = pd.DataFrame(T2, columns=[f"x={round(xi, 2)}" for xi in x])
dftm
st.markdown('berikut grafik untuk suhu pada batang:')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.title('Heat Conduction in a Rod')
plt.legend()
st.pyplot(plt.gcf())
st.markdown('''
jadi bisa dilihat pada grafik bahwa suhu yang awalnya hanya terdapat di tengah batang telah menyebar ke bagian lain seiring waktu, akan tetapi suhu yang berada
 di tengah berkurang seiring waktu.
''')

st.header("Runge-Kutta Method for Simulating Predator-prey dynamics (Lotka-Volterra)")

def rk4(r, t, h):                   
        """ Runge-Kutta 4 method """
        k1 = h*f(r, t)
        k2 = h*f(r+0.5*k1, t+0.5*h)
        k3 = h*f(r+0.5*k2, t+0.5*h)
        k4 = h*f(r+k3, t+h)
        return (k1 + 2*k2 + 2*k3 + k4)/6

def f(r, t):
        alpha = 1.0
        beta = 0.5
        gamma = 0.5
        sigma = 2.0
        x, y = r[0], r[1]
        fxd = x*(alpha - beta*y)
        fyd = -y*(gamma - sigma*x)
        return np.array([fxd, fyd], float)

h=0.001                               
tpoints = np.arange(0, 30, h)         
xpoints, ypoints  = [], []
tval=[]
r = np.array([2, 2], float)
for t in tpoints:
        xpoints.append(r[0])         
        ypoints.append(r[1])  
        tval.append(t)        
        r += rk4(r, t, h)  
xpoints.append(r[0])
ypoints.append(r[1])
tval.append(t+h)           
st.markdown('berikut equation yang akan digunakan:')
st.latex('''dx/dt = α*x - β*x*y''')
st.latex('''dy/dt = δ*x*y - γ*y''')
st.markdown('''
α= prey growth \n
β= prey death rate by predator \n
δ= predator growth due to prey \n
γ= predator death rate
''')
st.markdown('equation yang digunakan adalah lotka volterra yang merupakan model matematika yang mwnggambarkan dinamika populasu prey dan predator')
st.markdown('berikut hasil dari simulasi jumlah prey dan predator')
dfpr=pd.DataFrame({'time':tval,'prey':xpoints,'pred':ypoints})
dfpr
st.markdown('jadi setelah 30 tahun prey yang awalnya 2 menjadi 0.996 atau dibulatkan menjadi 1,' 
' dan predator yang awalnya 2 menjadi 8.118531 atau dibulatkan menjadi 8')
plt.figure(figsize=(8, 6))
plt.plot(tval, xpoints,label='prey')
plt.plot(tval, ypoints,label='predator')
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Lotka-Volterra Model")
plt.legend()
st.pyplot(plt.gcf())

st.markdown("diatas adalah plot untuk jumlah prey dan predator dalam jangka waktu dari hasil simulasi." \
" berdasarkan hasil grafik bisa dilihat bahwa jumlah prey cepat berkurang dan hanya bertambah saat jumlah predator sedikit. " \
"sedangkan jumlah predator cepat bertambah sampai jumlah prey sangat sedikit, dimana jumlah predatorcepat menurun.")
plt.figure(figsize=(8, 6))
plt.plot(xpoints, ypoints)
plt.xlabel("Prey")
plt.ylabel("Predator")
st.pyplot(plt.gcf())
st.markdown('bisa dilihat dari grafik jumlah predator berhubung dengan jumlah prey diatas, bahwa semakin banyak jumlah predator semakin sedikit jumlah prey dan sebaliknya.')
st.markdown('''
berikut kode yang digunakan:''')
st.code('''def rk4(r, t, h):                   
        """ Runge-Kutta 4 method """
        k1 = h*f(r, t)
        k2 = h*f(r+0.5*k1, t+0.5*h)
        k3 = h*f(r+0.5*k2, t+0.5*h)
        k4 = h*f(r+k3, t+h)
        return (k1 + 2*k2 + 2*k3 + k4)/6

        h=0.001                               
        tpoints = np.arange(0, 30, h)         
        xpoints, ypoints  = [], []
        tval=[]
        r = np.array([2, 2], float)
        for t in tpoints:
         xpoints.append(r[0])         
         ypoints.append(r[1])  
         tval.append(t)        
         r += rk4(r, t, h)  
        xpoints.append(r[0])
        ypoints.append(r[1])
        tval.append(t+h)           
''')
st.header("Runge-Kutta Method for Simulating Finance and Economic Modeling")
# Define the differential equation: dA/dt = rA
def f(t, A, r):
    return r * A

# Runge-Kutta 4th Order Method
def rk4(f, A0, r, t_span, h):
    t_values = np.arange(t_span[0], t_span[1]+1, h)
    A_values = np.zeros(len(t_values))
    A_values[0] = A0

    for i in range(1, len(t_values)):
        t = t_values[i-1]
        A = A_values[i-1]
        k1 = h * f(t, A, r)
        k2 = h * f(t + 0.5*h, A + 0.5*k1, r)
        k3 = h * f(t + 0.5*h, A + 0.5*k2, r)
        k4 = h * f(t + h, A + k3, r)
        A_values[i] = A + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t_values, A_values

# Parameters
A0 = 10000000  
r = 0.20      
t_span = (0,20)  
h = 1     

# Compute the solution
t_values, A_values = rk4(f, A0, r, t_span, h)
st.markdown('berikut equation yang akan digunakan:')
st.latex("dA/dt=rA,  A(0)=A0")
st.write('''
A= uang setelah bunga\n
r= bunga \n
A0= uang awal
''')
st.markdown('equation yang digunakan adalah euation dari compound interest')
st.markdown('berikut hasil dari compound interest:')
dfci=pd.DataFrame({'time':t_values,'money':A_values})
dfci
st.markdown('jadi setelah 20 tahun uang yang awalnya hanya 10,000,000 menjadi 545,956,842.255' \
'berikut plot dari hasil compound interest')
# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(t_values, A_values, label=f'RK4: A(t) with r={r*100}%', color='b', marker='o')
plt.title('Compound Interest Over Time (RK4 Approximation)')
plt.xlabel('Time (Years)')
plt.ylabel('Amount (IDR)')
plt.grid(True)
plt.legend()
plt.tight_layout()
st.pyplot(plt.gcf())
st.markdown('jadi berdasrkan plot dari compound interest, bisa dilihat bahwa jumlah uang bertambah secara exponential seiring waktu')
st.markdown('''
berikut kode yang digunakan:
''')
st.code('''
0 = 10000000  
r = 0.20     
t_span = (0,20) 
h = 1       
        
def rk4(f, A0, r, t_span, h):
    t_values = np.arange(t_span[0], t_span[1]+1, h)
    A_values = np.zeros(len(t_values))
    A_values[0] = A0

    for i in range(1, len(t_values)):
        t = t_values[i-1]
        A = A_values[i-1]
        k1 = h * f(t, A, r)
        k2 = h * f(t + 0.5*h, A + 0.5*k1, r)
        k3 = h * f(t + 0.5*h, A + 0.5*k2, r)
        k4 = h * f(t + h, A + k3, r)
        A_values[i] = A + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t_values, A_values''')
st.subheader("Evaluation and Discussion")
st.write('''
kelebihan dari runge kutta: \n
 - mudah diimplementasikan \n
 - memiliki akurasi yang tinggi\n
 - self starting ( tidak memerlukan proses khus untuk beberapa step pertama)\n
 - sangat stabil\n
\n
kekurangan runge kutta:\n
- memakan resource dan waktu komputasi yang tinggi\n
- tidak membuat estimasi error yang global\n
- tidak bagus digunakan untuk equation yang stiff\n
jadi runge kutta adalah metode untuk mengestimasi nilai dari difrerential equation yang memiliki banyak aplikasi dalam berbagai bidang. 
karena kelebihanya metode runge kutta sering digunakan di berbagai bidang
.
''')
