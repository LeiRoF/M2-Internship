import sys
sys.path.append('/home/rabnaeb/Documents/COURS/Master2/stage/SOC/')
from SOC_aux import *
import DustLib
from DustLib import *

# Choose a dust model from DustEM
DUST = 'MC10'
DUSTEM_DIR = '/home/rabnaeb/Documents/COURS/Master2/stage/dustem4.2_web'
DUSTEM_FILE = 'GRAIN_MC10.DAT'

FREQ = loadtxt('freq.dat')

dname = DUST+'.dust'
dscfile = DUST+'.dsc'

DustLib.DUSTEM_DIR = DUSTEM_DIR

NEWNAME = write_DUSTEM_files(DUSTEM_FILE)
DEDUST = []
METHOD = 'CRT'
for name in NEWNAME:
    DDUST = DustemDustO('dustem_%s.dust' % name, force_nsize=200)
    DEDUST.append(DDUST)
    write_simple_dust([DDUST,], FREQ, filename='%s_simple.dust' % name, dscfile='%s.dsc' % name)

write_simple_dust(DEDUST, FREQ, filename=dname, dscfile=dscfile)

if (0):
    DustLib.METHOD = 'CRT'
    write_A2E_dustfiles(NEWNAME, DEDUST, NE_ARAAY=128*ones(len(DEDUST)), prefix='')