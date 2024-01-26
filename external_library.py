from skfdiff import Model, Simulation

import pylab as pl

import numpy as np

from scipy.signal.windows import gaussian
N=56
l=2
n_worms = 48000
Oam=0.21
a = 1.90 * 10 ** (-2)
b = -3.98 * 10 ** (-3)
c = 2.25 * 10 ** (-4)
# tumbling rate
tau = 0.5
# oxygen diffusion coefficient
D0 = 2 * 10 ** (-5)
# oxygen penetration rate
f = 0.65
# oxygen consumption rate by worms and bacteria
kc = 7.3 * 10 ** (-10)
model = Model(["(dxDw+dyDw)*(dxW+dyW) + Dw*(dxxW+dyyW) + (dxB+dyB)*W*(dxO+dyO) + B*((dxW+dyW)*(dxO+dyO)+W*(dxxO+dyyO))",

               "D * (dxxO+dyyO) + f * (o - O) - k * W"],

               ["W(x, y)", "O(x, y)"],

               parameters=["W(x, y)","O(x,y)", "D", "f", "k", "tau", "a", "b", "c", "o"],
               subs=dict(Dw="V*V/(2*tau)",
                         B = "V/(2*tau)*dV",
                         V = "a*O*O+b*O+c",
                         dV = "2*a*O+b"),

               boundary_conditions="periodic")

x = np.linspace(0, l, N)

y = np.linspace(0, l, N)

xx, yy = np.meshgrid(x, y, indexing="ij")

W = 1 / (N * N) * (n_worms * np.ones((N, N)) + np.random.uniform(-n_worms,n_worms,(N, N)))
O = Oam * np.random.normal(0,0.05,(N, N))
initial_fields = model.Fields(x=x, y=y, W=W, O=O, D=D0, f=f, k=kc, tau=tau, a=a, b=b, c=c, o=Oam)
initial_fields["T"].plot()