import numpy as np

def thselect(x, tptr):
  x = np.transpose(x).reshape(-1)
  n = x.size
  thr = 0
  if tptr == 'rigrsure':
    sx2 = np.power(np.sort(np.absolute(x)),2)
    risks = (n - (2 * np.arange(1,n+1))) + (np.cumsum(sx2) + np.multiply(np.arange(n-1,-1,-1), sx2))/n
    risk = np.amin(risks)
    best = np.argmin(risks)
    thr = np.sqrt(sx2[best])
  elif tptr == 'heursure':
    hthr = np.sqrt(2*np.log(n))
    eta = (np.power(np.linalg.norm(x),2) - n)/n
    crit = np.power((np.log(n)/np.log(2)), 1.5)/np.sqrt(n)
    if eta < crit:
      thr = hthr
    else:
      thr = min(thselect(x,'rigrsure'), hthr) 
  elif tptr == 'sqrwolog':
    thr = np.sqrt(2*np.log(n))
  elif tptr == 'minimaxi':
    if n <= 32:
      thr = 0
    else:
      thr = 0.3936 + 0.1829*(np.log(n)/np.log(2))
  return(thr)


