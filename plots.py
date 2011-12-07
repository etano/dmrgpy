import matplotlib.pyplot as plt
from math import *
import numpy as np

def makePlots(L,nm,E0,cut):

  # Exact Energy
  x0s = np.arange(0,L,1)
  E0s = []
  for i in range(0,L):
    E0s.append(E0)

  Ls = []
  Es = []
  Is = []
  SiSjs = []
  Sis = []
  TruncErr = []
  Svn = []
  ls = []
  m = 2
  for im in range(0,nm):
    m = m << 1
    f = open('data/dmrg-'+str(L)+'-'+str(m),'r')
    data = f.readlines()
    Ls.append([])
    Es.append([])
    Is.append([])
    SiSjs.append([])
    Sis.append([])
    TruncErr.append([])
    Svn.append([])
    ls.append([])
    for i in range(0,len(data)):
      data[i] = map(float,data[i].split())
      Ls[im].append(int(data[i][0]))
      Es[im].append(data[i][1])
      Is[im].append(int(data[i][2]))
      SiSjs[im].append(data[i][3])
      Sis[im].append(data[i][4])
      TruncErr[im].append(data[i][5])
      ls[im].append(data[i][6])
      Svn[im].append(data[i][7])

  # Energy Convergence
  m = 2
  for im in range(0,nm):
    m = m << 1
    plt.plot(Ls[im][0:cut],Es[im][0:cut],marker='o',label='m='+str(m))
  plt.plot(x0s,E0s,linestyle='dashed',label='exact')
  plt.legend()
  plt.xlabel('L')
  plt.ylabel('E/L')
  plt.suptitle('Energy per site')
  plt.savefig('plots/E-'+str(L)+'.png')
  plt.clf()

  # Local Coupling
  m = 2
  for im in range(0,nm):
    m = m << 1
    plt.plot(Is[im][-cut:],SiSjs[im][-cut:],marker='o',linestyle='dashed',label='m='+str(m))
  plt.plot(x0s,E0s,linestyle='dashed',label='E0')
  plt.legend()
  plt.xlabel('i')
  plt.ylabel('<S_i S_i+1>')
  plt.suptitle('Local Coupling')
  plt.savefig('plots/SiSj-'+str(L)+'.png')
  plt.clf()

  # Local Magnetization
  m = 2
  for im in range(0,nm):
    m = m << 1
    plt.plot(Is[im],Sis[im],label='m='+str(m))
  plt.legend()
  plt.xlabel('i')
  plt.ylabel('<Sz_i>')
  plt.suptitle('Local Magnetization')
  plt.savefig('plots/Si-'+str(L)+'.png')
  plt.clf()

  # Relative Error in Energy
  m = 2
  ms = []
  errEs = []
  for im in range(0,nm):
    m = m << 1
    ms.append(m)
    errEs.append(abs((Es[im][-1]-E0)/E0))
  plt.semilogy(ms,errEs,marker='o',linestyle='dashed')
  plt.xlabel('m')
  plt.ylabel('err(E)/E0')
  plt.suptitle('Relative Error in Energy')
  plt.savefig('plots/errE-'+str(L)+'.png')
  plt.clf()

  # Entanglement Entropy
  m = 2
  for im in range(0,nm):
    m = m << 1
    plt.plot(ls[im][-cut:],Svn[im][-cut:],marker='o',linestyle='dashed',label='m='+str(m))
  plt.legend()
  plt.xlabel('L')
  plt.ylabel('S_VN')
  plt.suptitle('Von Neumann Entanglement Entropy')
  plt.savefig('plots/Svn-'+str(L)+'.png')
  plt.clf()

  m = 2
  for im in range(0,nm):
    m = m << 1
    xp = []
    for i in range(0,len(ls[im])):
      xp.append((2*L/pi)*sin(pi*(ls[im][i]+1)/L))
    plt.semilogx(xp[-cut:],Svn[im][-cut:],'o',label='m='+str(m))
  plt.legend(loc=2)
  plt.xlabel('x\'')
  plt.ylabel('S_VN(x\')')
  plt.suptitle('Von Neumann Entanglement Entropy (Scaled)')
  plt.savefig('plots/Svnp-'+str(L)+'.png')
  plt.clf()
