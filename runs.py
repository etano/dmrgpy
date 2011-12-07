import subprocess
import plots
from math import *

L = 10
nm = 5
m = 2
for im in range(0,nm):
  m = m << 1
  print L, m
  subprocess.call(['python','dmrg.py',str(L),str(m)])
plots.makePlots(L,nm,-0.425803520728,10)

L = 100
nm = 5
m = 2
for im in range(0,nm):
  m = m << 1
  print L, m
  subprocess.call(['python','dmrg.py',str(L),str(m)])
plots.makePlots(100,5,0.25 - log(2),96)
