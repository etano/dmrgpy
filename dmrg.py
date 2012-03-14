from math import *
import sys
import numpy as np
import scipy
from scipy.sparse import kron
from scipy.sparse.linalg import eigsh
from scipy.sparse import identity as I
from numpy.linalg import eigh

def TP(M):
  N = len(M)
  A = M[0]
  for i in range(0,N-1):
    A = kron(A,M[i+1])
  return A

def BuildSuperBlock(HB,SzB,SpB,SmB,HU,SzU,SpU,SmU,Sz,Sp,Sm):
  mB = HB.shape[0]
  mU = HU.shape[0]
  HBB = kron(HB,I(2)) + kron(SzB,Sz) + 0.5*(kron(SpB,Sm) + kron(SmB,Sp))
  HLR = kron(Sz,Sz) + 0.5*(kron(Sp,Sm) + kron(Sm,Sp))
  HBU = kron(I(2),HU) + kron(Sz,SzU) + 0.5*(kron(Sm,SpU) + kron(Sp,SmU))
  HSB = kron(HBB,I(2*mU)) + kron(I(mB*2),HBU) + TP((I(mB),HLR,I(mU)))
  #HBB = kron(HB,I(2*2*mU)) + kron(I(mB*2*2),HU) + TP((SzB,Sz,I(2*mU))) + TP((I(mB),Sz,Sz,I(mU))) + TP((I(mB*2),Sz,SzU)) + 0.5*(TP((SpB,Sm,I(2*mU))) + TP((I(mB),Sp,Sm,I(mU))) + TP((I(mB*2),Sp,SmU)) + TP((SmB,Sp,I(2*mU))) + TP((I(mB),Sm,Sp,I(mU))) + TP((I(mB*2),Sm,SpU)))
  return (mB,mU,HSB)

def FormRho(psi,mB,mU,LR):
  Psi = np.zeros((2*mB,2*mU))
  for i in range(0,2*mB):
    k = 0
    for j in range(2*mU*i,2*mU*(i+1)):
      Psi[i,k] = psi[j]
      k += 1
  if (LR == 'L'):
    return np.dot(Psi.T,Psi)
  else:
    return np.dot(Psi,Psi.T)

def Project(v,l,HB,SzB,SpB,SmB):
  HB[l+1] = v * HB[l+1].todense() * v.T
  x,u = eigh(HB[l+1])
  HB[l+1] = u * HB[l+1] * u.T
  SzB[l+1] = u * v * SzB[l+1] * v.T * u.T
  SpB[l+1] = u * v * SpB[l+1] * v.T * u.T
  SmB[l+1] = u * v * SmB[l+1] * v.T * u.T

  return (HB,SzB,SpB,SmB)

def RightDMRGAdd(f,(HB,SzB,SpB,SmB),Sz,Sp,Sm,mnew,lB,lU):

  # Form Superblock
  (mB,mU,HBU) = BuildSuperBlock(HB[lB],SzB[lB],SpB[lB],SmB[lB],HB[lU],SzB[lU],SpB[lU],SmB[lU],Sz,Sp,Sm)
  m = mB

  # Diagonalize Superblock
  w,psi = eigsh(HBU,k=1)
  E = w[0] # Energy

  # Form Rho
  rho = FormRho(psi,mB,mU,'R')

  # Diagonalize Rho
  w,v = eigh(rho)
  w = w[-mnew:]
  v = v.T[-mnew:]

  # Measure Things
  SiSj = kron(SzB[lB],Sz) + 0.5*(kron(SmB[lB],Sp) + kron(SpB[lB],Sm)) # Local Bond Strength
  TruncErr = 1 - sum(w) # Truncation Error
  Svn = 0. # Entanglement Entropy
  for i in range(0,mnew):
    if (w[i] > 0.):
      Svn -= w[i]*log(w[i])
  f.write(str(lB+lU+4)+' '+str(E/(lB+lU+4))+' '+str(lB)+' '+str(np.trace(rho*SiSj))+' '+str(np.trace(rho*kron(SzB[lB],I(2))))+' '+str(TruncErr)+' '+str(lB+1)+' '+str(Svn)+'\n')

  # Add New Site
  HB.append(kron(HB[lB],I(2)) + SiSj)
  SzB.append(kron(I(m),Sz))
  SpB.append(kron(I(m),Sp))
  SmB.append(kron(I(m),Sm))

  # Project Out New Operators
  return Project(v,lB,HB,SzB,SpB,SmB)

def LeftDMRGStep(f,(HB,SzB,SpB,SmB),Sz,Sp,Sm,mnew,lB,lU):

  # Form Superblock
  (mB,mU,HBU) = BuildSuperBlock(HB[lB],SzB[lB],SpB[lB],SmB[lB],HB[lU],SzB[lU],SpB[lU],SmB[lU],Sz,Sp,Sm)
  m = mU

  # Diagonalize Superblock
  w,psi = eigsh(HBU,k=1)
  E = w[0] # Energy

  # Form Rho
  rho = FormRho(psi,mB,mU,'L')

  # Diagonalize Rho
  w,v = eigh(rho)
  w = w[-mnew:]
  v = v.T[-mnew:]

  # Measure Things
  SiSj = kron(Sz,SzB[lU]) + 0.5*(kron(Sp,SmB[lU]) + kron(Sm,SpB[lU])) # Local Bond Strength
  TruncErr = 1 - sum(w) # Truncation Error
  Svn = 0. # Entanglement Entropy
  for i in range(0,mnew):
    if (w[i] > 0.):
      Svn -= w[i]*log(w[i])
  f.write(str(lB+lU+4)+' '+str(E/(lB+lU+4))+' '+str(lB+lU+2-lU)+' '+str(np.trace(rho*SiSj))+' '+str(np.trace(rho*kron(I(2),SzB[lU])))+' '+str(TruncErr)+' '+str(lU+1)+' '+str(Svn)+'\n')

  # Add New Site
  HB[lU+1] = kron(I(2),HB[lU]) + SiSj
  SzB[lU+1] = kron(Sz,I(m))
  SpB[lU+1] = kron(Sp,I(m))
  SmB[lU+1] = kron(Sm,I(m))

  # Project Out New Operators
  return Project(v,lU,HB,SzB,SpB,SmB)

def RightDMRGStep(f,(HB,SzB,SpB,SmB),Sz,Sp,Sm,mnew,lB,lU):

  # Form Superblock
  (mB,mU,HBU) = BuildSuperBlock(HB[lB],SzB[lB],SpB[lB],SmB[lB],HB[lU],SzB[lU],SpB[lU],SmB[lU],Sz,Sp,Sm)
  m = mB

  # Diagonalize Superblock
  w,psi = eigsh(HBU,k=1)
  E = w[0] # Energy

  # Form Rho
  rho = FormRho(psi,mB,mU,'R')

  # Diagonalize Rho
  w,v = eigh(rho)
  w = w[-mnew:]
  v = v.T[-mnew:]

  # Measure Things
  SiSj = kron(SzB[lB],Sz) + 0.5*(kron(SmB[lB],Sp) + kron(SpB[lB],Sm)) # Local Bond Strength
  TruncErr = 1 - sum(w) # Truncation Error
  Svn = 0. # Entanglement Entropy
  for i in range(0,mnew):
    if (w[i] > 0.):
      Svn -= w[i]*log(w[i])
  f.write(str(lB+lU+4)+' '+str(E/(lB+lU+4))+' '+str(lB)+' '+str(np.trace(rho*SiSj))+' '+str(np.trace(rho*kron(SzB[lB],I(2))))+' '+str(TruncErr)+' '+str(lB+1)+' '+str(Svn)+'\n')

  # Add New Site
  HB[lB+1] = kron(HB[lB],I(2)) + SiSj
  SzB[lB+1] = kron(I(m),Sz)
  SpB[lB+1] = kron(I(m),Sp)
  SmB[lB+1] = kron(I(m),Sm)

  # Project Out New Operators
  return Project(v,lB,HB,SzB,SpB,SmB)

def main():

  # Get Inputs
  L = int(sys.argv[1])
  m = int(sys.argv[2])
  N = int(sys.argv[3])

  # Data File
  f = open('data/dmrg-'+str(L)+'-'+str(m),'w')

  # Original Operators
  Sz = np.zeros((2,2))
  Sz[0,0] = 0.5
  Sz[1,1] = -0.5
  SzB = []
  SzB.append(Sz)
  Sp = np.zeros((2,2))
  Sp[0,1] = 1
  SpB = []
  SpB.append(Sp)
  Sm = np.zeros((2,2))
  Sm[1,0] = 1
  SmB = []
  SmB.append(Sm)
  HB = []
  HB.append(np.zeros((2,2)))
  Ops = (HB,SzB,SpB,SmB)

  # Infinite DMRG Loop
  pastm = 0
  for l in range(0,L/2-1):
    # Get m
    if (pastm):
      mnew = m
    else:
      mnew = 1 << (l+2)
    if (mnew > m):
      pastm = 1
      mnew = m
    # Infinite DMRG Step
    Ops = RightDMRGAdd(f,Ops,Sz,Sp,Sm,mnew,l,l)

  # First Half Finite DMRG Loop
  for l in range(L/2-1,L-4):
    Ops = RightDMRGAdd(f,Ops,Sz,Sp,Sm,m,l,L-l-4)

  # Finite DMRG Sweeps
  for n in range(0,N):

    # Left DMRG Sweep
    pastm = 0
    for l in range(0,L-4):
      # Get m
      if (pastm):
        mnew = m
      else:
        mnew = 1 << (l+2)
      if (mnew >= m-1):
        pastm = 1
        mnew = m
      Ops = LeftDMRGStep(f,Ops,Sz,Sp,Sm,mnew,L-l-4,l)

    # Right DMRG Sweep
    pastm = 0
    for l in range(0,L-4):
      # Get m
      if (pastm):
        mnew = m
      else:
        mnew = 1 << (l+2)
      if (mnew >= m-1):
        pastm = 1
        mnew = m
      Ops = RightDMRGStep(f,Ops,Sz,Sp,Sm,mnew,l,L-l-4)

  f.close()
  return 0

if __name__ == "__main__":
  main()
