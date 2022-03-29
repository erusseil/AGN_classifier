import numpy as np

MINIMUM_POINT = 4
BAND = ['g','r']

#---------------------------------------------------------FITTING FUNCTIONS---------------------------------------------------------------------  

#----------------------------------------------------------------------------
## BASIC FUNCTIONS 
#------------------------------------------------------------------------------


def protected_exponent(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 10, np.exp(x1), np.exp(10))


def sig(x):
    return 1/(1+protected_exponent(-x))


#------------------------------------------------------------------------------
## BUMP
#------------------------------------------------------------------------------


def bump(x, p1, p2, p3, p4):
    return sig(p1*x + p2 - protected_exponent(p3*x)) + p4

guess_bump = [0.225, -2.5, 0.038, 0]
original_shift_bump = 40