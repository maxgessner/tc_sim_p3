import sys
import platform
import os
import numpy as np
# from numpy.polynomial.chebyshev import chebfit
# from numpy.polynomial.chebyshev import chebval
# import numdifftools as nd
import pandas as pd
from scipy.signal import savgol_filter as sgf
from scipy.signal import wiener
from scipy.signal import medfilt2d as med2d
from scipy.signal import medfilt as med
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import pyprind
# from scipy.optimize import curve_fit as cfit
# plt.xkcd()
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import axes3d
# from mpl_toolkits.mplot3d.axes3d import Axes3D
# from mpl_toolkits import axes3d
# import mpl_toolkits.mplot3d
# from mpl_toolkits import mplot3d
# import matplotlib.projections as proj
# from mpl_toolkits.mplot3d.axes3d import Axes3D
# from mpl_toolkits import mplot3d
# from mpl_toolkits.mplot3d import axes3d
# loading modules for fitting
from scipy.optimize import curve_fit
# from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.special import erf
from scipy.optimize import minimize
from scipy.optimize import fmin
# from matplotlib import cm
# from scipy import signal
from scipy import interpolate
from scipy.interpolate import interp1d as int1d
# from window_function import window_function
# from functions import window_function
# from functions_py3 import getdata
# from functions_py3 import delta_cp_and_e_ht
from functions_py3 import sim_delta_cp_and_e_ht
from functions_py3 import sim_e_ht
from functions_py3 import sim_e_ht2
from functions_py3 import sim_e_ht3
from functions_py3 import sim_plot_temp_history
# from functions_py3 import cut_current_no_zero
# from functions_py3 import dt_to_dx
from functions_py3 import temp_to_intensity
from functions_py3 import temp_to_intensity2
from functions_py3 import intensity_to_temp
from functions_py3 import intensity_to_temp2
from functions_py3 import cal_cp_from_current
from functions_py3 import cal_cp_from_current2
from functions_py3 import cal_cp_from_current3
from functions_py3 import cal_cp_from_current4
from functions_py3 import cal_cp_from_current5
from functions_py3 import cal_cp_from_current6
from functions_py3 import load_reference
from functions_py3 import mean_diff2
from functions_py3 import mean_diff3
from functions_py3 import mean_diff_all
from itertools import cycle
from tkinter import Tk as tk
from tkinter import filedialog as fd
import progressbar
import warnings
# import matplotlib
# matplotlib.use('Agg')
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', np.RankWarning)
# warnings.simplefilter('ignore', curve_fit.OptimizeWarning)
# import tkFileDialog as fd

# plt.xkcd()
plt.ion()
# plt.style.use(['dark_background', 'presentation'])
plt.style.use('presentation')
block = 0

# set to False if Simulation is in Temperature
is_in_itensity = True

# export evaluation data to csv for further investigations
export_to_csv = False

# smooth data? (None, "sgf", "wiener", "med", "med2d")
smooth_data = "wiener"
ksize = 15

# skip data choice?
skip_data_choice = True

# use the span selector?
select_span = True

# verbose mode?
verbose = False

# material data

lit_cp = 265        # specific heat capacity J / (kg * K)
lit_delta = 8570    # density kg / m**3
lit_epsht = 0.3     # emissivity
lit_lambda = 54     # thermal conductivity W / (m * K)

# choose what to skip: 'calc', 'input', 'none'
# SKIP = 'input'
# SKIP = 'calc'
if len(sys.argv) > 1:
    SKIP = sys.argv[1]
else:
    SKIP = 'none'

# if SKIP == 'calc' or SKIP == 'none':
if verbose is True:
    print('Import data ...')

# platform dependent, where to search for datafiles
if platform.system() == 'Windows':
    file = 'I:\THERMISCHE ANALYSE\Messmethoden\PISA\PISA_Labor' \
        '\\demodata1.txt'
# changed for testing on Linux Laptop which is not connected to ZAEgoto
elif platform.system() == 'Linux':
    file = 'demodata1_5col_pulse_noise_py3.txt'

    path = '/home/mgessner/vm_share/max_pisa/working/'
    # file_scan = '/home/mgessner/vm_share/max_pisa/working/4test/DatenKleinEins/combined_pisa_scan.txt'
    # file_bbhole = '/home/mgessner/vm_share/max_pisa/working/4test/DatenKleinEins/combined_pisa_bbhole.txt'
    # file_scan = '/home/mgessner/vm_share/max_pisa/working/4test/Daten/combined_pisa_scan.txt'
    # file_bbhole = '/home/mgessner/vm_share/max_pisa/working/4test/Daten/combined_pisa_bbhole.txt'
    file_scan = path + '/5test/Daten/combined_pisa_scan_from_150.txt'
    file_bbhole = path + '/5test/Daten/combined_pisa_bbhole_from_150.txt'
    # file_scan = path + '/7test/Daten/combined_pisa_scan_from_101.txt'
    # file_bbhole = path + '/7test/Daten/combined_pisa_bbhole_from_101.txt'
    # print(file_bbhole)
    # print(file_scan)
    # exit()
    # file = 'demodata1.txt'
else:
    sys.exit('It is recommened to use Linux! (or Windows if you have to)')


# is commented for test reasons,
# just to load same file every time, is faster
# # ask user to set path for data
# skip_data_choice = False

if skip_data_choice is True or SKIP != 'none':
    # file = '/home/mgessner/simulations/34test/1_to_4001/combined_bbhole_from_1_to_4001.txt'
    # file = '/home/mgessner/simulations/34test/1_to_4001_noise_2percent/combined_bbhole_from_1_to_4001.txt'
    # file = '/home/mgessner/simulations/35test/1_to_4001/combined_bbhole_from_1_to_4001.txt'
    # file = '/home/mgessner/simulations/35test/1_to_4001_noise_2percent/combined_bbhole_from_1_to_4001.txt'
    # file = '/home/mgessner/simulations/36test/1_to_4001/combined_bbhole_from_1_to_4001.txt'
    # file = '/home/mgessner/simulations/36test/1_to_4001_noise_2percent/combined_bbhole_from_1_to_4001.txt'
    # file = '/home/mgessner/simulations/37test/1_to_4001/combined_bbhole_from_1_to_4001.txt'
    # file = '/home/mgessner/simulations/37test/1_to_4001_noise_2percent/combined_bbhole_from_1_to_4001.txt'
    # file = '/home/mgessner/simulations/45test/1_to_5001/combined_bbhole_from_1_to_5001.txt'
    # file = '/home/mgessner/simulations/45test/1_to_5001_noise_2percent/combined_bbhole_from_1_to_5001.txt'
    # file = '/home/mgessner/simulations/46test/1_to_1001/combined_bbhole_from_1_to_1001.txt'
    # file = '/home/mgessner/simulations/46test/1_to_1001_noise_2percent/combined_bbhole_from_1_to_1001.txt'
    # file = '/home/mgessner/simulations/04Niob/1_to_401/combined_bbhole_from_1_to_401.txt'
    # file = '/home/mgessner/simulations/04Niob/1_to_401_noise_2percent/combined_bbhole_from_1_to_401.txt'
    # file = '/home/mgessner/simulations/04Niob/1_to_401_noise_5percent/combined_bbhole_from_1_to_401.txt'
    # file = '/home/mgessner/simulations/05Niob/1_to_401/combined_bbhole_from_1_to_401.txt'
    # file = '/home/mgessner/simulations/05Niob/1_to_401_noise_2percent/combined_bbhole_from_1_to_401.txt'
    # file = '/home/mgessner/simulations/05Niob/1_to_401_noise_5percent/combined_bbhole_from_1_to_401.txt'
    # file = '/home/mgessner/simulations/06Niob/1_to_401/combined_bbhole_from_1_to_401.txt'
    # file = '/home/mgessner/simulations/06Niob/1_to_401_noise_2percent/combined_bbhole_from_1_to_401.txt'
    # file = '/home/mgessner/simulations/06Niob/1_to_401_noise_5percent/combined_bbhole_from_1_to_401.txt'
    # file = '/home/mgessner/simulations/07Niob/1_to_401/combined_bbhole_from_1_to_401.txt'
    # file = '/home/mgessner/simulations/07Niob/1_to_401_noise_2percent/combined_bbhole_from_1_to_401.txt'
    # file = '/home/mgessner/simulations/07Niob/1_to_401_noise_5percent/combined_bbhole_from_1_to_401.txt'
    # file = '/home/mgessner/simulations/08Niob/1_to_2001/combined_bbhole_from_1_to_2001.txt'
    # file = '/home/mgessner/simulations/08Niob/1_to_2001_noise_2percent/combined_bbhole_from_1_to_2001.txt'
    # file = '/home/mgessner/simulations/08Niob/1_to_2001_noise_5percent/combined_bbhole_from_1_to_2001.txt'
    # file = '/home/mgessner/simulations/09Niob/1_to_2001/combined_bbhole_from_1_to_2001.txt'
    # file = '/home/mgessner/simulations/09Niob/1_to_2001_noise_2percent/combined_bbhole_from_1_to_2001.txt'
    # file = '/home/mgessner/simulations/09Niob/1_to_2001_noise_5percent/combined_bbhole_from_1_to_2001.txt'
    # file = '/home/mgessner/simulations/12Niob/1_to_3001/combined_bbhole_from_1_to_3001.txt'
    # file = '/home/mgessner/simulations/12Niob/1_to_3001_noise_2percent/combined_bbhole_from_1_to_3001.txt'
    # file = '/home/mgessner/simulations/12Niob/1_to_3001_noise_5percent/combined_bbhole_from_1_to_3001.txt'
    # file = '/home/mgessner/simulations/12Niob/1_to_1001/combined_bbhole_from_1_to_1001.txt'
    # file = '/home/mgessner/simulations/12Niob/1_to_1001_noise_2percent/combined_bbhole_from_1_to_1001.txt'
    file = '/home/mgessner/simulations/12Niob/1_to_501/combined_bbhole_from_1_to_501.txt'
    # file = '/home/mgessner/simulations/12Niob/1_to_501_noise_2percent/combined_bbhole_from_1_to_501.txt'
    # file = '/home/mgessner/simulations/12Niob/1_to_251/combined_bbhole_from_1_to_251.txt'
    # file = '/home/mgessner/simulations/12Niob/1_to_251_noise_2percent/combined_bbhole_from_1_to_251.txt'
    # file = '/home/mgessner/simulations/07Iron/1_to_1001/combined_bbhole_from_1_to_1001.txt'
    # file = '/home/mgessner/simulations/07Iron/1_to_1001_noise_2percent/combined_bbhole_from_1_to_1001.txt'
elif skip_data_choice is False or SKIP != 'input':
    window = tk()
    window.withdraw()

    file = fd.askopenfilename(filetypes=[('Text files', 'combined*.txt')],
                              initialdir='/home/mgessner/vm_share/max_pisa/working/')

    window.destroy()

    window = tk()
    window.withdraw()


# if ref_file != '':
#     plt.plot(eps_ref_x, eps_ref_y)
# plt.show()
# input()
# exit()


# file = '/home/mgessner/vm_share/max_pisa/working/16test/combined_bbhole_from_1_to_1001.txt'
# file = '/home/mgessner/vm_share/max_pisa/working/16test/combined_bbhole_from_1_to_4307.txt'
# file = '/home/mgessner/vm_share/max_pisa/working//12test/workingtest/combined_scan_from_200.txt'
# file = '/home/mgessner/vm_share/max_pisa/working/12test/' \
#         'for_simulation_extra_small/combined_bbhole_from_250.txt'
# exit()


# reftype = 'Iron'
reftype = 'Niob'

if reftype.lower() in file.lower():
    reftype = 'Niob'
    print(reftype)
else:
    reftype = 'Iron'
    print(reftype)

# check if user aborted file selection
if file == '' or file == ():
    sys.exit('no file to load data from!')
else:
    if skip_data_choice is True or SKIP == 'input':
        if reftype == 'Iron':
            ref_file = '/home/mgessner/vm_share/max_pisa/working/refdata/Iron/'
        elif reftype == 'Niob':
            ref_file = '/home/mgessner/vm_share/max_pisa/working/refdata/Niob/'

    elif skip_data_choice is False:
        if reftype == 'Iron':
            ref_file = fd.askdirectory(initialdir='/home/mgessner/vm_share/max_pisa/working/refdata/Iron/')
        elif reftype == 'Niob':
            ref_file = fd.askdirectory(initialdir='/home/mgessner/vm_share/max_pisa/working/refdata/Niob/')

    if ref_file != '':
        if reftype == 'Iron':
            eps_ref_x, eps_ref_y = load_reference(ref_file + '/Temperature__vs_Spectral_nor_87.npy.npz')
            hc_ref_x, hc_ref_y = load_reference(ref_file + '/Temperature__vs_Heat_capacit_119.npy.npz')
            tc_ref_x, tc_ref_y = load_reference(ref_file + '/Temperature__vs_Thermal_cond_261.npy.npz')
            sigma_ref_x, sigma_ref_y = load_reference(ref_file + '/Temperature__vs_Electrical_c_51.npy.npz')
            rho_ref_x, rho_ref_y = load_reference(ref_file + '/Temperature__vs_Electrical_r_226.npy.npz')
            den_ref_x, den_ref_y = load_reference(ref_file + '/Temperature__vs_Specific_den_135.npy.npz')
        elif reftype == 'Niob':
            eps_ref_x, eps_ref_y = load_reference(ref_file + '/Temperature__vs_Spectral_nor_98.npy.npz')
            hc_ref_x, hc_ref_y = load_reference(ref_file + '/Temperature__vs_Heat_capacit_168.npy.npz')
            tc_ref_x, tc_ref_y = load_reference(ref_file + '/Temperature__vs_Thermal_cond_48.npy.npz')
            sigma_ref_x, sigma_ref_y = load_reference(ref_file + '/Temperature__vs_Electrical_c_90.npy.npz')
            rho_ref_x, rho_ref_y = load_reference(ref_file + '/Temperature__vs_Electrical_r_90.npy.npz')
            den_ref_x, den_ref_y = load_reference(ref_file + '/Temperature__vs_Specific_den_366.npy.npz')
        # den_ref = 7874  # kg/m**3
        if verbose is True:
            print('reference loaded ...\n')
            print('... complete\n')
            print('directory of reference:\n')
            print(ref_file)
            print('\n')
    elif ref_file == '':
        den_ref = 7874  # kg/m**3
        den_ref = 8570
        print('no reference loaded!')

    if verbose is True:
        print('... complete!\n')
    print('loaded data:')
    print(file)
    print('\n')

# set the path name from selected file
path = os.path.dirname(file) + '/'

# load all three files 'bbhole', 'scan' and 'voltage'
file_scan = os.path.basename(file).replace('bbhole', 'scan')
file_scan = file_scan.replace('voltage', 'scan')
file_bbhole = os.path.basename(file).replace('scan', 'bbhole')
file_bbhole = file_bbhole.replace('voltage', 'bbhole')
file_voltage = os.path.basename(file).replace('scan', 'voltage')
file_voltage = file_voltage.replace('bbhole', 'voltage')

# abort if one of the three files cannot be found
if os.path.isfile(path + file_scan) is False or \
  os.path.isfile(path + file_bbhole) is False or \
  os.path.isfile(path + file_voltage) is False:
    print('file missing!')
    sys.exit()
elif os.path.getsize(path + file_scan) <= 16 or \
    os.path.getsize(path + file_bbhole) <= 16 or \
    os.path.getsize(path + file_voltage) <= 16:
    print('file empty!')
    sys.exit()

if SKIP == 'calc' or SKIP == 'none':
    # get data from file
    # important to notice: pandas.read_csv are faster than numpy.genfromtxt
    data_scan = pd.read_csv(path + file_scan, delimiter='\t', header=0,
                            engine='c', decimal='.')
    data_bbhole = pd.read_csv(path + file_bbhole, delimiter='\t', header=0,
                              engine='c', decimal='.')
    data_voltage = pd.read_csv(path + file_voltage, delimiter='\t', header=0,
                               engine='c', decimal='.')

    # initialize matrices for the data to get stored in
    scan = data_scan.as_matrix()

    bbhole = data_bbhole.as_matrix()

    voltage = data_voltage.as_matrix()

    # remove leading and tailing values for the simulation generates
    # same values for following scans
    scan = np.delete(scan,
                     np.where(np.roll(scan[:, 0], 1) == scan[:, 0]), 0)

    bbhole = np.delete(bbhole,
                       np.where(np.roll(bbhole[:, 0], 1) == bbhole[:, 0]), 0)

    voltage = np.delete(voltage,
                        np.where(np.roll(voltage[:, 0], 1) == voltage[:, 0]), 0)

    # remove dublicate values
    change_scan = np.where(np.roll(scan[:, 1], 1) != scan[:, 1])[0]
    change_bbhole = np.where(np.roll(bbhole[:, 1], 1) != bbhole[:, 1])[0]
    change_voltage = np.where(np.roll(voltage[:, 1], 1) != voltage[:, 1])[0]

    fs = 0  # 0.02
    fl = 1

    # how far from the center should be cut off in percent
    # actual 0% off center
    fitlength = np.int(np.round(np.mean(np.diff(change_scan)) * (fl - fs), 0))

    # how far from the side should be cut off in percent
    # actual 00%
    fitstart = np.int(np.round(np.mean(np.diff(change_scan)) * fs, 0))

    # initialize numpy arrays for the data
    np_scan = np.ndarray(())
    np_scan2 = np.ndarray(())
    np_scan3 = np.ndarray(())
    all_np_scan = np.ndarray(())

    np_bbhole = np.ndarray(())
    np_bbhole2 = np.ndarray(())
    all_np_bbhole = np.ndarray(())

    np_voltage = np.ndarray(())
    np_voltage2 = np.ndarray(())
    all_np_voltage = np.ndarray(())

    # bring these numpy arrays to the correct size
    np_scan.resize((len(change_scan) - 1, fitlength, 3))
    np_scan2.resize((len(change_scan) - 1, fitlength, 3))
    np_scan3.resize((len(change_scan) - 1, fitlength, 3))
    all_np_scan.resize((len(change_scan) - 1, max(np.diff(change_scan)), 3))

    c_scan = np.empty(np.shape(np_scan)[0])
    c_all_scan = np.empty(np.shape(np_scan)[0])

    np_bbhole.resize((len(change_bbhole) - 1, fitlength, 3))
    np_bbhole2.resize((len(change_bbhole) - 1, fitlength, 3))
    all_np_bbhole.resize((len(change_bbhole) - 1, max(np.diff(change_bbhole)), 3))

    c_bbhole = np.empty(np.shape(np_scan)[0])
    c_all_bbhole = np.empty(np.shape(np_scan)[0])

    # generate the voltage as difference in voltage on top and on the bottom
    # of the specimen
    np_voltage.resize((len(change_voltage) - 1, fitlength, 3))
    np_voltage2.resize((len(change_voltage) - 1, fitlength, 3))
    all_np_voltage.resize((len(change_voltage) - 1, max(np.diff(change_voltage)), 3))
    c_all_voltage = np.empty(np.shape(np_voltage)[0])
    diff_all_voltage = np.empty(np.shape(np_voltage)[0])
    diff_all_voltage1 = np.empty(np.shape(np_voltage)[0])

    c_all_time = np.empty(np.shape(np_voltage)[0])

    # not used anymore
    sgf_length = np.int(np.round(fitlength / 10))
    if sgf_length % 2 == 0:
        sgf_length += 1
    # print(sgf_length)
    sgf_porder = 5
    if sgf_porder > sgf_length:
        sgf_porder = np.int(np.round(sgf_porder / 2))

    # make start and end point of scan same value
    scan[change_scan, 2] = np.mean(scan[change_scan, 2])
    scan[change_scan - 1, 2] = np.mean(scan[change_scan - 1, 2])

    # function for fitting scans - not used actually
    def scanfunc(x, a, b, c, d, e):
        # x_0 = x + np.max(x) # x from 0
        # x_l = np.max(x) - x # x from l
        # d = np.max(np.scan[:, :, 0])
        # return(- a * 0.5 * ((np.tanh((x - d) * b) + np.tanh((- x - d) * b))))
        # return(-a / ((np.exp(b * (x - d)) + 1) * (np.exp(b * (- x - d))) + 1))
        return(-a * (((np.exp(1 * (x - d) * b) - 1) / (np.exp(1 * (x - d) * b) + 1)) +
               ((np.exp(1 * (-x - d) * b) - 1) / (np.exp(1 * (-x - d) * b) + 1))) - c)
        # return(-a * (((np.exp((x - d) * b) - 1) / (np.exp((x - e) * c) + 1)) +
        #        ((np.exp((-x - e) * c) - 1) / (np.exp((-x - d) * b) + 1))))
        # return(-a * ((erf((-x - b) * c)) + erf((x - b) * c)))
        # return(-a * ((erf((x**2 - b) * c))) + e)
        # return(a * 0.5 * (np.exp(-b * x_0) * erf((x_0 * c) - d) + np.exp(b * x_0) * erf((x_0 * c) + d)))
        # return(a * np.cosh(b * (np.max(x) - x_0)) / np.cosh(b * np.max(x)))
        # return(a * (1 - (np.sinh(b * x_l) + np.sinh(b * (x_0))) / np.sinh(b * e)))
        # print(x + np.max(x)) # x from 0 to l
        # print(np.max(x) - x)
        # return(a * (np.sin(b * (x + c)) * np.cos(d * (x + e))))
        # return((a * (np.sinh(b * x_0) + np.sinh(c * x_l))) / (np.sinh(c * b * np.max(x))))
        # return(a * ((np.sinh(b * (d - x))) + np.sinh(c * x)) / np.sinh(b * c))

    # function for fitting blackbody hole - not used actually
    def bbholefunc(x, a):
        # d = np.max(np.scan[:, :, 0])
        return(a)

    # function for fitting voltagedrop - actually used
    # def voltagefunc(x, a):
    #     # d = np.max(np.scan[:, :, 0])
    #     return(a * x)



    # what fit to be used
    fit_np_scan = False
    fit_np_bbhole = False
    fit_np_voltage = True

    # generate progressbars for loading data
    if fit_np_scan is True:
        # progbar_scan = pyprind.ProgBar(len(change_scan) - 1, width=50, track_time=False)
        progbar_scan = pyprind.ProgPercent(len(change_scan) - 1, track_time=False, title='fit scan')

    # *** NP_SCAN ***

    for i in range(len(change_scan) - 1):
        start = change_scan[i] + fitstart
        end = change_scan[i] + fitstart + fitlength

        np_scan[i, :] = scan[start:end, :]
        all_np_scan[i, :] = scan[change_scan[i]:change_scan[i + 1], :]

        if fit_np_scan is True:

            # splinefit begin scan
            g = np.int(np.round(np.shape(np_scan[i, :, 2])[0] / 10))
            a0 = np.max(np_scan[i, g:-g, 2])
            # a0 = 1
            b0 = 0.1
            c0 = 0.1
            # d0 = np.max(np_scan[i, :, 0])
            d0 = 0.1
            e0 = 0.1
            # e0 = d0
            # e0 = 0
            # print(a0, b0, c0, d0)
            # print("hello")
            # input()
            # if i > 0:
            #     a0 = popt_back[0]
            #     b0 = popt_back[1]
            #     c0 = popt_back[2]
            #     d0 = popt_back[3]
            #     e0 = popt_back[4]
            # print(popt_back)
            # input()
            popt_back = (a0, b0, c0, d0, e0)
            # print(popt_back)
            errorraised = False

            try:
                weight = np.ones(np.shape(np_scan)[1])
                weight[[0, -1]] = 0.01
                popt, pcov = curve_fit(scanfunc, np_scan[i, :, 0], np_scan[i, :, 2], p0=(a0, b0, c0, d0, e0), sigma=weight, maxfev=50)
                sgflength = np.trunc(np.shape(np_scan)[1] / 10)
                if sgflength % 2 == 0:
                    sgflength += 1
                    # scanfunc = np.poly1d(np.polyfit(np_scan[i, :, 0], wiener(np_scan[i, :, 2]), deg=np.int(np.shape(np_scan)[1]/2), w=weight))
                    # scanfunc = chebval(chebfit(np_scan[i, :, 0], np_scan[i, :, 2], deg=np.int(np.shape(np_scan)[1]/2)))
                # popt_back = popt
            except (RuntimeError):
                popt = popt_back
                # print(RuntimeError)
                # print("error")
                errorraised = True

            ### HERE SOMETHING TO DO!!!
            np_scan2[i, :, :] = np.copy(np_scan[i, :, :])
            np_scan3[i, :, :] = np.copy(np_scan[i, :, :])
            # np_scan2[i, :, 2] = wiener(scanfunc(med(np_scan[i, :, 0])))
            np_scan2[i, :, 2] = wiener(scanfunc(med(np_scan[i, :, 0]), *popt))

            # splinefit end scan

            progbar_scan.update()


        if smooth_data == "sgf":
            np_scan[i, :, 2] = sgf(scan[start:end, 2], sgf_length, sgf_porder, mode='nearest')
            all_np_scan[i, :, 2] = sgf(scan[change_scan[i]:change_scan[i + 1], 2], sgf_length, sgf_porder, mode='nearest')
        elif smooth_data == "wiener":
            np_scan[i, :, 2] = wiener(scan[start:end, 2])
            all_np_scan[i, :, 2] = wiener(scan[change_scan[i]:change_scan[i + 1], 2])
        elif smooth_data == "med":
            np_scan[i, :, 2] = med(scan[start:end, 2], kernel_size=ksize)  # , sgf_length, sgf_porder)
            all_np_scan[i, :, 2] = med(scan[change_scan[i]:change_scan[i + 1], 2], kernel_size=ksize)  # , sgf_length, sgf_porder)
        # if scan[i, 1] < split and scan[i, 1] > pulse_stop:
        #     np_scan_for_cp[i, :] = scan[start:end, :]
        #     all_np_scan_for_cp[i, :] = scan[change_scan[i]:change_scan[i + 1], :]
        # if scan[i, 1] >= split:
        #     np_scan_for_tc[i, :] = scan[start:end, :]
        #     all_np_scan_for_tc[i, :] = scan[change_scan[i]:change_scan[i + 1], :]
        # c_scan[i] = np.mean(np_scan[i, :, 2])
        # print(np_scan[i, np.int(np.shape(np_scan[i, :])[0] / 2)])
        if smooth_data != "med2d":
            c_scan[i] = np_scan[i, np.int(np.shape(np_scan[i, :])[0] / 2), 2]
            # c_all_scan[i] = np.mean(all_np_scan[i, :, 2])
            c_all_scan[i] = all_np_scan[i, np.int(np.shape(all_np_scan[i, :])[0] / 2), 2]

    if fit_np_scan is True:

        np_scan[:, :, 2] = np_scan2[:, :, 2]

    if smooth_data == "med2d":
        all_np_scan[:, :, 2] = med2d(all_np_scan[:, :, 2], kernel_size=ksize)
        for i in range(len(change_scan) - 1):
            c_scan[i] = np_scan[i, np.int(np.shape(np_scan[i, :])[0] / 2), 2]
            c_all_scan[i] = all_np_scan[i, np.int(np.shape(all_np_scan[i, :])[0] / 2), 2]

    # *** NP_BBHOLE ***

    if fit_np_bbhole is True:
        # progbar_bbhole = pyprind.ProgBar(len(change_bbhole) - 1, width=100, track_time=False)
        progbar_bbhole = pyprind.ProgPercent(len(change_bbhole) - 1, track_time=False, title='fit bbhole')

    for j in range(len(change_bbhole) - 1):
        start = change_bbhole[j] + fitstart
        end = change_bbhole[j] + fitstart + fitlength

        np_bbhole[j, :] = bbhole[start:end, :]
        all_np_bbhole[j, :] = bbhole[change_bbhole[j]:change_bbhole[j + 1], :]

        if fit_np_bbhole is True:
            # splinefit begin bbhole

            a0 = np.max(np_bbhole[j, :, 2])

            popt_back = [a0]

            try:
                popt, pcov = curve_fit(bbholefunc, np_bbhole[j, :, 0], np_bbhole[j, :, 2], p0=(a0))
            except (RuntimeError):
                popt = popt_back

            np_bbhole2[j, :, :] = np_bbhole[j, :, :]
            np_bbhole2[j, :, 2] = bbholefunc(np_bbhole[j, :, 0], *popt)
            # splinefit end bbhole

            progbar_bbhole.update()

        # smooth data using sgf, wiener oder med
        if smooth_data == "sgf":
            np_bbhole[j, :, 2] = sgf(bbhole[start:end, 2], sgf_length, sgf_porder, mode='nearest')
            all_np_bbhole[j, :, 2] = sgf(bbhole[change_bbhole[j]:change_bbhole[j + 1], 2], sgf_length, sgf_porder, mode='nearest')

        elif smooth_data == "wiener":
            np_bbhole[i, :, 2] = wiener(bbhole[start:end, 2])  # , sgf_length, sgf_porder)
            all_np_bbhole[j, :, 2] = wiener(bbhole[change_bbhole[j]:change_bbhole[j + 1], 2])  # , sgf_length, sgf_porder)

        elif smooth_data == "med":
            np_bbhole[i, :, 2] = med(bbhole[start:end, 2], kernel_size=ksize)  # , sgf_length, sgf_porder)
            all_np_bbhole[j, :, 2] = med(bbhole[change_bbhole[j]:change_bbhole[j + 1], 2], kernel_size=ksize)  # , sgf_length, sgf_porder)

        if smooth_data != "med2d":
            c_bbhole[j] = np_bbhole[j, np.int(np.shape(np_bbhole[j, :])[0] / 2), 2]
            c_all_bbhole[j] = all_np_bbhole[j, np.int(np.shape(np_bbhole[j, :])[0] / 2), 2]

    if fit_np_bbhole is True:

        np_bbhole[:, :, 2] = np_bbhole2[:, :, 2]

    if smooth_data == "med2d":
        all_np_bbhole[:, :, 2] = med2d(all_np_bbhole[:, :, 2], kernel_size=ksize)
        for j in range(len(change_bbhole) - 1):
            c_bbhole[j] = np_bbhole[j, np.int(np.shape(np_bbhole[j, :])[0] / 2), 2]
            c_all_bbhole[j] = all_np_bbhole[j, np.int(np.shape(np_bbhole[j, :])[0] / 2), 2]

    # *** NP_VOLTAGE ***
    if fit_np_voltage is True:
        # progbar_bbhole = pyprind.ProgBar(len(change_voltage) - 1, width=100, track_time=False)
        progbar_voltage = pyprind.ProgPercent(len(change_voltage) - 1, track_time=False, title='fit voltage')

    for k in range(len(change_voltage) - 1):
        start = change_voltage[k] + fitstart
        end = change_voltage[k] + fitstart + fitlength

        np_voltage[k, :] = voltage[start:end, :]
        all_np_voltage[k, :] = voltage[change_voltage[k]:change_voltage[k + 1], :]

        if fit_np_voltage is True:

            # splinefit begin voltage
            # if k == 0:
            # a0 = 1  # np.max(np_bbhole[i, :, 2])
            # popt_back = [a0]
            # if k > 0:
            #     a0 = popt[0]
            #     popt_back = popt

            # try:
            #     popt, pcov = curve_fit(voltagefunc, np_voltage[k, :, 0], np_voltage[k, :, 2], p0=(a0))
            # except (RuntimeError):
            #     popt = popt_back

            popt = np.polyfit(np_voltage[k, :, 0], np_voltage[k, :, 2], 2)

            np_voltage2[k, :, :] = np_voltage[k, :, :]
            # print(np_voltage2)
            # np_voltage2[k, :, 2] = voltagefunc(np_voltage[k, :, 0], *popt)
            np_voltage2[k, :, 2] = np.poly1d(popt)(np_voltage[k, :, 0])
            # np_voltage2[k, :, 2][np.abs(np_voltage2[k, :, 2] < 1e-100)] = 0.0
            # print(np_voltage2)
            # exit()

            progbar_voltage.update()

        elif fit_np_voltage is False:
            np_voltage2 = np_voltage

        if smooth_data == "sgf":
            np_voltage[k, :, 2] = sgf(voltage[start:end, 2], sgf_length, sgf_porder, mode='nearest')
            all_np_voltage[k, :, 2] = sgf(voltage[change_voltage[k]:change_voltage[k + 1], 2], sgf_length, sgf_porder, mode='nearest')

        elif smooth_data == "wiener":
            np_voltage[k, :, 2] = wiener(voltage[start:end, 2])
            all_np_voltage[k, :, 2] = wiener(voltage[change_voltage[k]:change_voltage[k + 1], 2])  # , sgf_length, sgf_porder, mode='nearest')

        elif smooth_data == "med":
            np_voltage[k, :, 2] = med(voltage[start:end, 2], kernel_size=ksize)
            all_np_voltage[k, :, 2] = med(voltage[change_voltage[k]:change_voltage[k + 1], 2], kernel_size=ksize)  # , sgf_length, sgf_porder, mode='nearest')

        diff_all_voltage[k] = np.abs(np.max(np_voltage2[k, :, 2])) + np.abs(np.min(np_voltage2[k, :, 2]))
        # diff_all_voltage[k] = np.abs(np.max(np_voltage[k, :, 2])) + np.abs(np.min(np_voltage[k, :, 2]))

        # if k > 20:
        #     print(np_voltage[k, :, 2])
        #     print(diff_all_voltage[k])
        #     print(np_voltage2[k, :, 2])
        #     print(diff_all_voltage1[k])
        #     input()

        if diff_all_voltage[k] < 0.0001:
            diff_all_voltage[k] = 0

        c_all_time[k] = np.mean(all_np_voltage[k, :, 1])
        c_all_voltage[k] = np.max(all_np_voltage[k, :, 2])

    # plt.figure('voltage test')
    # # plt.plot(voltage, label='voltage')
    # plt.plot(diff_all_voltage * 2, label='diff_all_voltage')
    # plt.plot(diff_all_voltage1, label='diff_all_voltage1')
    # plt.legend()
    # input()
    # exit()
    # # print(voltage)
    # print(np_voltage)
    # print(np_voltage2)
    # exit()

        # if smooth_data != "med2d":
            # c_all_voltage[k] = np.mean(all_np_voltage[k, :, 2])
            # c_all_time[k] = np.mean(all_np_voltage[k, :, 1])
            # c_voltage = np_voltage[:, np.int(np.shape(np_voltage[k, :])[0] / 2), 2]
            # c_all_voltage = all_np_voltage[:, np.int(np.shape(np_voltage[k, :])[0] / 2), 2]
            # c_all_voltage[k] = np.max(all_np_voltage[k, :, 2])

    if fit_np_voltage is True:

        np_voltage[:, :, 2] = np_voltage2[:, :, 2]

    if smooth_data == "med2d":
        all_np_voltage[:, :, 2] = med2d(all_np_voltage[:, :, 2], kernel_size=ksize)

        for k in range(np.shape(np_voltage)[0]):
            diff_all_voltage[k] = np.abs(np.max(np_voltage2[k, :, 2])) + np.abs(np.min(np_voltage2[k, :, 2]))
            
            if diff_all_voltage[k] < 0.0001:
                diff_all_voltage[k] = 0

    # ###RESTLICHE VARIABLEN AUCH SPEICHERN!!!
    if SKIP == 'calc' or SKIP != 'none':
        print('saving variables ...')
        np.save('variables/all_np_bbhole.npy', all_np_bbhole)
        np.save('variables/all_np_scan.npy', all_np_scan)
        np.save('variables/all_np_voltage.npy', all_np_voltage)
        np.save('variables/c_all_bbhole.npy', c_all_bbhole)
        np.save('variables/c_all_scan.npy', c_all_scan)
        np.save('variables/c_all_time.npy', c_all_time)
        np.save('variables/c_all_voltage.npy', c_all_voltage)
        np.save('variables/c_bbhole.npy', c_bbhole)
        np.save('variables/c_scan.npy', c_scan)
        np.save('variables/change_scan.npy', change_scan)
        np.save('variables/change_bbhole.npy', change_bbhole)
        np.save('variables/change_voltage.npy', change_voltage)
        np.save('variables/np_bbhole.npy', np_bbhole)
        np.save('variables/np_bbhole2.npy', np_bbhole2)
        np.save('variables/np_scan.npy', np_scan)
        np.save('variables/np_scan2.npy', np_scan2)
        np.save('variables/np_scan3.npy', np_scan3)
        np.save('variables/np_voltage.npy', np_voltage)
        np.save('variables/np_voltage2.npy', np_voltage2)
        np.save('variables/diff_all_voltage.npy', diff_all_voltage)
        np.save('variables/fitstart.npy', fitstart)
        np.save('variables/fitlength.npy', fitlength)
        print('... variables saved!')


if SKIP == 'input' or SKIP != 'none':
    all_np_bbhole = np.load('variables/all_np_bbhole.npy')
    all_np_scan = np.load('variables/all_np_scan.npy')
    all_np_voltage = np.load('variables/all_np_voltage.npy')
    c_all_bbhole = np.load('variables/c_all_bbhole.npy')
    c_all_scan = np.load('variables/c_all_scan.npy')
    c_all_time = np.load('variables/c_all_time.npy')
    c_all_voltage = np.load('variables/c_all_voltage.npy')
    c_bbhole = np.load('variables/c_bbhole.npy')
    c_scan = np.load('variables/c_scan.npy')
    change_scan = np.load('variables/change_scan.npy')
    change_bbhole = np.load('variables/change_bbhole.npy')
    change_voltage = np.load('variables/change_voltage.npy')
    np_bbhole = np.load('variables/np_bbhole.npy')
    np_bbhole2 = np.load('variables/np_bbhole2.npy')
    np_scan = np.load('variables/np_scan.npy')
    np_scan2 = np.load('variables/np_scan2.npy')
    np_scan3 = np.load('variables/np_scan3.npy')
    np_voltage = np.load('variables/np_voltage.npy')
    np_voltage2 = np.load('variables/np_voltage2.npy')
    diff_all_voltage = np.load('variables/diff_all_voltage.npy')
    fitstart = np.load('variables/fitstart.npy')
    fitlength = np.load('variables/fitlength.npy')
elif SKIP == 'calc':
    exit()

if verbose is True:
    print('Calculate emissivity ...')

# (epsht, f_epsht, temp_epsht) = \
#     sim_e_ht2(bbhole[change_bbhole + np.int(np.mean(np.diff(change_bbhole)) / 2)],
#               scan[change_scan + np.int(np.mean(np.diff(change_scan)) / 2)],
#               voltage[change_voltage + np.int(np.mean(np.diff(change_voltage)) - 1)],
#               litdata=[lit_delta, lit_cp, lit_epsht], plotresult=True)
(epsht, f_epsht, temp_epsht, f_epsht_int, f_epsht_int_int) = \
    sim_e_ht3(c_all_bbhole[:],
              c_all_scan[:],
              diff_all_voltage[:],
              litdata=[lit_delta, lit_cp, lit_epsht], plotresult=True)

# input()
# exit()

if ref_file != '':
    plt.plot(eps_ref_x, eps_ref_y, label='eps lit.')

if verbose is True:
    print('... complete!\n')


# progbar_scan2 = pyprind.ProgBar(np.shape(np_scan)[0], width=100)

# convert intensity to temperature for scan
if is_in_itensity is True:
    # print(all_np_scan[:, :, 2])
    np_scan[:, :, 2] = intensity_to_temp2(np_scan[:, :, 2], 900e-9,
                                          f_epsht_int)
    all_np_scan[:, :, 2] = intensity_to_temp2(all_np_scan[:, :, 2], 900e-9,
                                              f_epsht_int)
    c_scan = intensity_to_temp2(c_scan, 900e-9,
                                f_epsht_int)
    c_all_scan = intensity_to_temp2(c_all_scan, 900e-9,
                                    f_epsht_int)


# progbar_bbhole2 = pyprind.ProgBar(np.shape(np_bbhole)[0], width=100)

# convert intensity to temperature for blackbody hole
if is_in_itensity is True:
    np_bbhole[:, :, 2] = intensity_to_temp(np_bbhole[:, :, 2], 900e-9, 1.)
    all_np_bbhole[:, :, 2] = intensity_to_temp(all_np_bbhole[:, :, 2], 900e-9, 1.)
    c_bbhole = intensity_to_temp(c_bbhole, 900e-9, 1.)
    c_all_bbhole = intensity_to_temp(c_all_bbhole, 900e-9, 1.)

# find center value of scan for cp calculation
np_scan_center = np_scan[:, np.int(np.shape(np_scan)[1] / 2), 2]
# np_scan_diff_min_val = np.argmin(np.diff(np_scan_center))

# split = np.sign(np.abs(np.diff(np_scan_center)) - np.abs(np_scan_diff_min_val / 2))


# split_last = np_scan_diff_min_val
# split_first = np.int(np.argwhere(np.sign(np.diff(np_scan_center)[0:split_last][::-1]) >= 0)[-1])
# split_number = np.int((split_last))
# split = np_scan[split_number, :, 1][0]
# print(split)
# exit()
# print(split)
# exit()

if verbose is True:
    print('Calculate heat capacity ...')

if ref_file != '':
    rho_lit = int1d(rho_ref_x, rho_ref_y, kind='linear',
                    fill_value='extrapolate')
    den_lit = int1d(den_ref_x, den_ref_y, kind='linear',
                    fill_value='extrapolate')
else:
    rho_lit = int1d(c_all_bbhole, 1 / (6.58e6 * 300 / c_all_bbhole),
                    kind='linear', fill_value='extrapolate')
    den_lit = 7874
    # sigma_lit = int1d(c_all_scan, 1 / (6.58e6 * 300 / c_all_scan),
    #                   kind='linear', fill_value='extrapolate')


# print(1 / sigma_ref_y)
# print(rho_ref_y)
# plt.figure('check')
# plt.plot(c_all_bbhole[:])
# plt.plot((all_np_bbhole[:, np.int(np.shape(all_np_bbhole)[1] / 2), 2]), 'r.')
# input()
# exit()


# (dcp, f_dcp) = cal_cp_from_current2(rho_lit(c_all_bbhole),
#                                     70e-3, c_all_bbhole,
#                                     c_all_time, diff_all_voltage)
# (dcp, f_dcp) = cal_cp_from_current2(rho_lit(c_all_scan),
#                                     70e-3, c_all_scan,
#                                     c_all_time, diff_all_voltage)
# (dcp, f_dcp) = cal_cp_from_current5(rho_lit(c_all_bbhole),
#                                     den_ref_x, den_ref_y,
#                                     70e-3, np.ones_like(temp_epsht), temp_epsht,
#                                     c_bbhole, c_all_time,
#                                     diff_all_voltage)
# (dcp, f_dcp) = cal_cp_from_current5(rho_lit(c_all_scan),
#                                     den_ref_x, den_ref_y,
#                                     70e-3, epsht, temp_epsht,
#                                     c_all_scan, c_all_time,
#                                     diff_all_voltage)
# (dcp, f_dcp) = cal_cp_from_current3(rho_lit(c_all_scan),
#                                     den_ref_x, den_ref_y,
#                                     70e-3, epsht, temp_epsht,
#                                     c_all_scan, c_all_time,
#                                     diff_all_voltage)
# (dcp, f_dcp) = cal_cp_from_current4(rho_lit(c_all_scan),
#                                     den_ref_x, den_ref_y,
#                                     70e-3, epsht, temp_epsht,
#                                     c_all_scan, c_all_time,
#                                     diff_all_voltage)
# (dcp, f_dcp) = cal_cp_from_current4(rho_lit(c_all_bbhole),
#                                     den_ref_x, den_ref_y,
#                                     70e-3, np.ones_like(temp_epsht), temp_epsht,
#                                     c_all_bbhole, c_all_time,
#                                     diff_all_voltage)
# (dcp, f_dcp) = cal_cp_from_current6(rho_ref_x, rho_ref_y,
#                                     den_ref_x, den_ref_y,
#                                     70e-3, epsht, temp_epsht,
#                                     c_all_scan, c_all_time,
#                                     diff_all_voltage)
(dcp, f_dcp) = cal_cp_from_current6(rho_ref_x, rho_ref_y,
                                    den_ref_x, den_ref_y,
                                    70e-3, f_epsht(temp_epsht), temp_epsht,
                                    c_all_scan , c_all_time,
                                    diff_all_voltage)
# (dcp, f_dcp) = cal_cp_from_current6(rho_ref_x, rho_ref_y,
#                                     den_ref_x, den_ref_y,
#                                     70e-3, np.ones_like(temp_epsht), temp_epsht,
#                                     c_all_bbhole , c_all_time,
#                                     diff_all_voltage)

if ref_file != '':
    f_den_ref_y = int1d(den_ref_x, den_ref_y, kind='linear',
                        fill_value='extrapolate')
    f_hc_ref_y = int1d(hc_ref_x, hc_ref_y, kind='linear',
                       fill_value='extrapolate')
    # plt.plot(hc_ref_x, f_den_ref_y(hc_ref_x) * hc_ref_y, label='hc lit.')
    plt.plot(hc_ref_x, f_den_ref_y(hc_ref_x) * f_hc_ref_y(hc_ref_x), label='hc lit.')
    # plt.plot(hc_ref_x, hc_ref_y, label='hc lit.')
    plt.legend()
    # delta*cp einbauen!!!

if verbose is True:
    print('... complete!\n')

# print(dcp)
# plt.figure('check')
# plt.plot(rho_ref_x, rho_ref_y, label='rho lit')
# plt.legend()
# input()
# exit()

voltage_off = np.int((np.argwhere(np.abs(diff_all_voltage) > 0)[-1]) + 1)

# print(voltage_off)

i_np_scan_center = np.int(np.shape(np_scan)[1] / 2)
np_scan_center_mean = np.empty_like(np_scan[0, :, 0])

for o in range(np.shape(np_scan)[1]):
    np_scan_center_mean[o] = np.mean(np_scan[o, i_np_scan_center - 25:i_np_scan_center + 25, 2])
max_np_scan_center = np.max(np_scan_center_mean)

i_np_scan_max = np.argmax(np_scan_center_mean == max_np_scan_center)

if voltage_off > i_np_scan_max:
    i_np_scan_max = voltage_off
# print(i_np_scan_max)

# print(np_scan_center_mean)
# print(i_np_scan_max)
# exit()

np_scan_for_tc = np_scan[i_np_scan_max:, :, :]
all_np_scan_for_tc = all_np_scan[i_np_scan_max:, :, :]
np_bbhole_for_tc = np_bbhole[i_np_scan_max:, :, :]
all_np_bbhole_for_tc = np_bbhole[i_np_scan_max:, :, :]

if verbose is True:
    print('Calculate temperature history ...')

sim_plot_temp_history(raw_puv=all_np_scan[:-2, :, 2],
                      raw_time=all_np_scan[:-2, :, 1],
                      raw_pv=all_np_bbhole[:-2, :, 2],
                      raw_pv_time=all_np_bbhole[:-2, :, 1],
                      litdata=[lit_delta, lit_cp, lit_epsht],
                      plotresult=True, voltage_off=i_np_scan_max)

if verbose is True:
    print('... complete!\n')

# input()
# exit()
# (epsht, delta_cp, f_epsht, f_dcp, c_puv) = \
#     sim_delta_cp_and_e_ht(raw_puv=all_np_scan[:-2, :, 2],
#                           raw_time=all_np_scan[:-2, :, 1],
#                           raw_pv=all_np_bbhole[:-2, :, 2],
#                           # raw_pv=c_bbhole[:],
#                           raw_pv_time=all_np_bbhole[:-2, :, 1],
#                           litdata=[lit_delta, lit_cp, lit_epsht],
#                           plotresult=True)

# print('Calculate emissivity ...')

# # (epsht, f_epsht, c_time) = \
# #     sim_e_ht(raw_puv=all_np_scan[:-2, :, 2],
# #                           raw_time=all_np_scan[:-2, :, 1],
# #                           raw_pv=all_np_bbhole[:-2, :, 2],
# #                           # raw_pv=c_bbhole[:],
# #                           raw_pv_time=all_np_bbhole[:-2, :, 1],
# #                           litdata=[lit_delta, lit_cp, lit_epsht],
# #                           plotresult=True)
# (epsht, f_epsht, c_time) = \
#     sim_e_ht(raw_puv=all_np_scan_for_tc[:-2, :, 2],
#                           raw_time=all_np_scan_for_tc[:-2, :, 1],
#                           raw_pv=all_np_bbhole_for_tc[:-2, :, 2],
#                           # raw_pv=c_bbhole[:],
#                           raw_pv_time=all_np_bbhole_for_tc[:-2, :, 1],
#                           litdata=[lit_delta, lit_cp, lit_epsht],
#                           plotresult=True)

# print('... complete!\n')

# exit()

if verbose is True:
    print('Generate 3D Plots ...')

plot3Dhistory = False

if plot3Dhistory is True:
    yrange = np.arange(1, np.shape(all_np_scan)[0] + 1)
    zrange = np.arange(1, np.shape(all_np_scan)[1] + 1)

    np_scan_3d = np.empty([all_np_scan.shape[0], all_np_scan.shape[1], 3])

    for c in range(all_np_scan.shape[1]):
        for d in range(all_np_scan.shape[0]):
            np_scan_3d[d, c] = (all_np_scan[d, c, 2], d + 1, c + 1)

    X = all_np_scan[:, :, 0]
    Y = all_np_scan[:, :, 1]
    Z = all_np_scan[:, :, 2]

    plt.figure('3D plot')
    ax = plt.subplot(111, projection='3d')

    plt.title('3D plot')

    ax.plot_surface(X, Y, Z, cmap='jet')
    ax.set_xlabel('z (specimen) / m')
    ax.set_ylabel('propagating time / s')
    ax.set_zlabel('temperature / K')

# testwrapper!!!
np_scan_backup = np_scan_for_tc
np_bbhole_backup = np_bbhole_for_tc
# np_voltage_backup = np_voltage_for_tc

# select_span = True

global T_l
global T_r
T_l = 0.0
T_r = 0.0

if select_span is True:
    from matplotlib.widgets import SpanSelector
    fig1, (ax1, ax2) = plt.subplots(2, figsize=(12, 10))
    # ax1 = plt.figure('heat map')

    plt.suptitle('heat map plot')
    plt.title('select span to fit tc from:')
    # plt.pcolormesh(all_np_scan[0, :, 0],
    #                all_np_scan[:, 0, 1],
    #                all_np_scan[:, :, 2],
    #                cmap='jet')
    ax1.pcolormesh(all_np_scan[0, :, 0],
                   all_np_scan[:, 0, 1],
                   all_np_scan[:, :, 2],
                   cmap='jet')
    plt.xlabel('z (specimen) / m')
    plt.ylabel('propagating time / s')
    # plt.colorbar(extend='max', pad=0.1)
    # cbar2.set_label('temperature / K')

    ax2.pcolormesh(all_np_scan[0, :, 0],
                   all_np_scan[:, 0, 1],
                   all_np_scan[:, :, 2],
                   cmap='jet')
    plt.xlabel('z (specimen) / m')
    plt.ylabel('propagating time / s')


    # plt.axvline(all_np_scan[0, fitstart, 0], color='black')
    # # ax1.axvline(all_np_scan[0, fitstart, 0], color='black')
    # plt.axvline(all_np_scan[0, fitstart + fitlength - 1, 0], color='black')
    # # ax1.axvline(all_np_scan[0, fitstart + fitlength - 1, 0], color='black')
    # cbar2 = plt.colorbar(extend='max', pad=0.1)
    # # cbar2 = ax1.colorbar(extend='max', pad=0.1)
    # cbar2.set_label('temperature / K')
    # input()

    def onselect(xmin, xmax):
        indmin, indmax = np.searchsorted(all_np_scan[0, :, 0], (xmin, xmax))
        indmax = min(len(all_np_scan[0, :, 0]) - 1, indmax)

        thisx = all_np_scan[0, :, 0][indmin:indmax]
        if len(thisx) == 0:
            thisx = all_np_scan[0, :, 0][indmin-1:indmax+1]
        # thisy = all_np_scan[0, :, 2]  #[indmin:indmax]
        # line2.set_data(thisx, thisy)
        ax2.set_xlim(thisx[0], thisx[-1])
        # ax2.set_ylim(thisy.min(), thisy.max())
        fig1.canvas.draw()
        # if 'line_l' in locals() and 'line_r' in locals():
        #     ax1.lines.remove(line_l)
        #     ax1.lines.remove(line_r)
        # if 'selected' in locals():
        #     print('found')
        # selected.remove()
        ax1.clear()
        ax1.pcolormesh(all_np_scan[0, :, 0],
                       all_np_scan[:, 0, 1],
                       all_np_scan[:, :, 2],
                       cmap='jet')
        # line_l = ax1.axvline(thisx[0], color='black')
        # line_r = ax1.axvline(thisx[-1], color='black')
        ax1.axvspan(thisx[0], thisx[-1], color='green', alpha=0.5)
        # selected.remove()
        global T_l
        global T_r
        T_l = thisx[0]
        T_r = thisx[-1]
        # return(T_l, T_r)

    span = SpanSelector(ax1, onselect, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red'))

    # print(span)

    print('select span of data to calculate tc from:')
    print('(proceed with [ENTER])')
    input()
    print(T_l, T_r)
    print('\n')
    plt.close(fig1)

ax2 = plt.figure('heat map')

plt.title('heat map plot')
plt.pcolormesh(all_np_scan[0, :, 0],
               all_np_scan[:, 0, 1],
               all_np_scan[:, :, 2],
               cmap='jet')
plt.xlabel('z (specimen) / m')

plt.ylabel('propagating time / s')

cbar2 = plt.colorbar(extend='max', pad=0.1)
cbar2.set_label('temperature / K')

if select_span is True:
    plt.axvspan(T_l, T_r, color='green', alpha=0.5)
    ax2.text(0.95, 0.95,'Borders:$\n$T_l = %.2f$\n$T_r = %.2f ' % (T_l, T_r))
# plt.show(block)
# plt.show(block=False)

# print(T_l)
# print(T_r)


# # # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# # ## just for test reaasons
# # T_l = -0.0200704566
# # # T_r = 0.0200704566
# # T_r = -0.000375
# T_l = -0.0299313483
# T_r = -0.0101908899
# select_span = True
# # ##
# # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


i = 0

# input()
# exit()

auto_set_fs = True
# if true: fs = 0, fl = 1

if auto_set_fs is False:
    while True:
        try:
            print('input for START(0...1):')
            # fs = np.float64(input())
            fs = input()
            if fs == '\n' or fs == '':
                fs = 0
            elif np.float64(fs) < 0 or np.float64(fs) > 1:
                print(fs)
                raise ValueError()
            elif fs == '':
                exit()
            # elif fs == '\n':
            #     fs = 0
        except ValueError:
            print('only numeric float values between 0 and 1')
            # print(fs)
            continue
        else:
            break
elif auto_set_fs is True:
    fs = 0

fs = np.float64(fs)
# if len(sys.argv) > 2:
#     fl = np.float64(sys.argv[2])
# else:
fl = 1 - fs
# while True:
#     try:
#         print('input for LENGTH(0...1):')
#         fl = np.float64(input())
#         if fl < 0 or fl > 1:
#             raise ValueError()
#     except ValueError:
#         print("only numeric float values between 0 and 1")
#         continue
#     else:
#         break
# print('input for LENGTH(0...1):')
# fl = np.float64(input())
i += 1
if i > 1 and verbose is True:
    print('Generate 3d Plots ...')
# print(i)

# fitstart = np.int(np.round(np.mean(np.diff(change_scan)) * fs, 0))
# fitlength = np.int(np.round(np.mean(np.diff(change_scan)) * (fl - fs), 0))

# np_scan_for_tc = np_scan_for_tc[:, fitstart:fitlength, :]
# np_bbhole_for_tc = np_bbhole_for_tc[:, fitstart:fitlength, :]
# np_voltage_for_tc = np_voltage_backup[:, fitstart:fitlength, :]
if i > 1:
    ax2.clear()

if verbose is True:
    print('... complete!\n')
# input()

# static values for calculation
# to be set to variable values if they are provided by experiment
# c_p = 0.265  # J/(g*K)
sigma = 5.670367e-8  # W m-2 K-4
r_o = 7.6e-3
r_i = 6e-3
# p = 20e-3  # m
p = r_o * 2 * np.pi
# delta = 8.57 * 10**6  # g/m**3
# epsilon_ht = 0.02
s = (r_o**2 - r_i**2) * np.pi  # m**2
# s = 8.85 * 10**(-6) # m**2
t_a = 300  # K
d = 100  # distance center of rotary mirror to sample (perpendicular)
l = 70e-3  # sample length

if select_span is True:
    index_T_l = np.where(np_scan_for_tc[0, :, 0] <= T_l)[0][-1]
    index_T_r = np.where(np_scan_for_tc[0, :, 0] >= T_r)[0][0]
    print(index_T_l)
    print(index_T_r)
    # print(np.shape(np_scan_for_tc))
    t_raw = np.array(np_scan_for_tc[:, index_T_l:index_T_r, 2])
    # t_raw = wiener(np.array(np_scan_for_tc[:, index_T_l:index_T_r, 2]))
    x = np.array(np_scan_for_tc[:, index_T_l:index_T_r, 0])
    raw_time = np.array(np_scan_for_tc[:, index_T_l:index_T_r, 1])
else:
    # t_raw = np.array(np_scan_for_tc[:, :, 2])
    t_raw = wiener(np.array(np_scan_for_tc[:, :, 2]))
    x = np.array(np_scan_for_tc[:, :, 0])
    raw_time = np.array(np_scan_for_tc[:, :, 1])

# # calculate 1st and 2nd gradient along each profile (x direction)
# savgol_length = 15  # remember to set this to an ODD value
# savgol_order = 3  # remember to set this to an ODD value

# print(select_span)
# plt.figure('t_raw')
# plt.plot(t_raw[0], label='t_raw')
# plt.legend()
# input()
# print(np.shape(t_raw))
# print(np.shape(x[0]))
# print(np.shape(raw_time[:,0]))
# exit()

# tx = np.gradient(t_raw, axis=1, edge_order=0) / np.gradient(x, axis=1, edge_order=0)
# tx = np.gradient(t_raw, axis=1) / np.gradient(x, axis=1)
# tx = np.gradient(t_raw, axis=0) / np.gradient(x, axis=0)
tx = mean_diff2(t_raw, x, x_axis=1, y_axis=1)
# tx = np.gradient(t_raw, x[0], axis=1, edge_order=2)
# tx = mean_diff3(t_raw, x, x_axis=1, y_axis=1)
# tx = mean_diff_all(t_raw, x, x_axis=1, y_axis=1, num=5)
# print(np.shape(tx))
# print(np.shape(x[:, 1:-1]))

# tx = sgf(tx, savgol_length, savgol_order, mode='nearest')

# t2x = np.gradient(tx, axis=1, edge_order=0) / np.gradient(x, axis=1, edge_order=0)
# t2x = np.gradient(tx, axis=1) / np.gradient(x, axis=1)
# t2x = np.gradient(tx, axis=0) / np.gradient(x, axis=0)
t2x = mean_diff2(tx, x[:, 1:-1], x_axis=1, y_axis=1)
# t2x = np.gradient(tx, x[0], axis=1, edge_order=2)
# t2x = mean_diff3(tx, x[:, 1:-1], x_axis=1, y_axis=1)
# t2x = mean_diff3(tx, x[:, 2:-2], x_axis=1, y_axis=1)
# t2x = mean_diff_all(tx, x[:, 1:-1], x_axis=1, y_axis=1, num=5)

# t2x = sgf(t2x, savgol_length, savgol_order, mode='nearest')

# tt = np.gradient(t_raw, axis=0, edge_order=0) / np.gradient(raw_time, axis=0, edge_order=0)
# tt = np.gradient(t_raw, axis=0) / np.gradient(raw_time, axis=0)
# tt = np.gradient(t_raw, axis=1) / np.gradient(raw_time, axis=1)
tt = mean_diff2(t_raw, raw_time, x_axis=0, y_axis=0)
# tt = np.gradient(t_raw, raw_time[:, 0], axis=0, edge_order=2)
# tt = mean_diff3(t_raw, raw_time, x_axis=0, y_axis=0)
# tt = mean_diff_all(t_raw, raw_time, x_axis=0, y_axis=0, num=5)

# tt = sgf(tt, savgol_length, savgol_order, mode='nearest')

# print(np.shape(tx))
# print(np.shape(t2x))
# print(np.shape(tt))


tx = tx[1:-1, 1:-1]
t2x = t2x[1:-1, :]
tt = tt[:, 2:-2]

# tx = wiener(tx[1:-1, 1:-1])
# t2x = wiener(t2x[1:-1, :])
# tt = wiener(tt[:, 2:-2])

# tx = wiener(tx[2:-2, 2:-2])
# t2x = wiener(t2x[2:-2, :])
# tt = wiener(tt[:, 4:-4])


# print(np.shape(tx))
# print(np.shape(t2x))
# print(np.shape(tt))

t = t_raw[1:-1, 2:-2]
t_time = raw_time[1:-1, 2:-2]

# t = wiener(t_raw[1:-1, 2:-2])
# t_time = wiener(raw_time[1:-1, 2:-2])

# t = wiener(t_raw[2:-2, 4:-4])
# t_time = wiener(raw_time[2:-2, 4:-4])

# t = t_raw
# t_time = raw_time

# print('after conversion')
# for i in range(np.shape(t)[1]):
#     plt.figure('t')
#     plt.plot(t[:, i])
#     print(t[:, i])
#     # plt.plot(t[:, -1])
#     input()
# exit()

# print(np.gradient(raw_time, axis=0, edge_order=1))
# exit()

a0 = np.full(40, 0.0000, dtype=np.double)
# print(a0)

# a0[0] = 54.
# a0[0] = 1
# a0[6] = 0.

# # initiate xdata and ydata as list of the same length as raw_d
# xdata = list(range(len(np_scan[0])))
# ydata = xdata
# input()

def model(a, T):
    '''
    describes the model for heat capacity as a function of the temperature
    which will be used as left hand side for fitting

    with T is the temperature:
    T[0] = t   # temperature profile from scanning method
    T[1] = tx   # first derivative of temperature profile
    T[2] = t2x  # second derivative of temperature profile

    and a is the fitting parameter

    fitting will be to the 15. order of a
    '''
    return \
     (a[0]  *              (       T[2]               ) \
    + a[1]  *              (T[0] * T[2] + 1  * T[1]**2) \
    + a[2]  * T[0]**(1)  * (T[0] * T[2] + 2  * T[1]**2) \
    + a[3]  * T[0]**(2)  * (T[0] * T[2] + 3  * T[1]**2) \
    + a[4]  * T[0]**(3)  * (T[0] * T[2] + 4  * T[1]**2) \
    + a[5]  * T[0]**(4)  * (T[0] * T[2] + 5  * T[1]**2) \
    + a[6]  * T[0]**(5)  * (T[0] * T[2] + 6  * T[1]**2) \
    + a[7]  * T[0]**(6)  * (T[0] * T[2] + 7  * T[1]**2) \
    + a[8]  * T[0]**(7)  * (T[0] * T[2] + 8  * T[1]**2) \
    + a[9]  * T[0]**(8)  * (T[0] * T[2] + 9  * T[1]**2) \
    + a[10] * T[0]**(9)  * (T[0] * T[2] + 10 * T[1]**2) \
    + a[11] * T[0]**(10) * (T[0] * T[2] + 11 * T[1]**2) \
    + a[12] * T[0]**(11) * (T[0] * T[2] + 12 * T[1]**2) \
    + a[13] * T[0]**(12) * (T[0] * T[2] + 13 * T[1]**2) \
    + a[14] * T[0]**(13) * (T[0] * T[2] + 14 * T[1]**2) \
    + a[15] * T[0]**(14) * (T[0] * T[2] + 15 * T[1]**2) \
    + a[16] * T[0]**(15) * (T[0] * T[2] + 16 * T[1]**2) \
    + a[17] * T[0]**(16) * (T[0] * T[2] + 17 * T[1]**2) \
    + a[18] * T[0]**(17) * (T[0] * T[2] + 18 * T[1]**2) \
    + a[19] * T[0]**(18) * (T[0] * T[2] + 19 * T[1]**2) \
    + a[20] * T[0]**(19) * (T[0] * T[2] + 20 * T[1]**2) \
    + a[21] * T[0]**(20) * (T[0] * T[2] + 21 * T[1]**2) \
    + a[22] * T[0]**(21) * (T[0] * T[2] + 22 * T[1]**2) \
    + a[23] * T[0]**(22) * (T[0] * T[2] + 23 * T[1]**2) \
    + a[24] * T[0]**(23) * (T[0] * T[2] + 24 * T[1]**2) \
    + a[25] * T[0]**(24) * (T[0] * T[2] + 25 * T[1]**2) \
    + a[26] * T[0]**(25) * (T[0] * T[2] + 26 * T[1]**2) \
    + a[27] * T[0]**(26) * (T[0] * T[2] + 27 * T[1]**2) \
    + a[28] * T[0]**(27) * (T[0] * T[2] + 28 * T[1]**2) \
    + a[29] * T[0]**(28) * (T[0] * T[2] + 29 * T[1]**2) \
    + a[30] * T[0]**(29) * (T[0] * T[2] + 30 * T[1]**2) \
    + a[31] * T[0]**(30) * (T[0] * T[2] + 31 * T[1]**2) \
    + a[32] * T[0]**(31) * (T[0] * T[2] + 32 * T[1]**2) \
    + a[33] * T[0]**(32) * (T[0] * T[2] + 33 * T[1]**2) \
    + a[34] * T[0]**(33) * (T[0] * T[2] + 34 * T[1]**2) \
    + a[35] * T[0]**(34) * (T[0] * T[2] + 35 * T[1]**2) \
    + a[36] * T[0]**(35) * (T[0] * T[2] + 36 * T[1]**2) \
    + a[37] * T[0]**(36) * (T[0] * T[2] + 37 * T[1]**2) \
    + a[38] * T[0]**(37) * (T[0] * T[2] + 38 * T[1]**2) \
    + a[39] * T[0]**(38) * (T[0] * T[2] + 39 * T[1]**2) \
    )

    # bis a[7] alles gut


def residual(a, rhs, T):
    '''
    combining the model 'model' with the right hand side of the formula
    '''
    return(model(a, T) - rhs)


def residual2(a, rhs, T):
    '''
    combining the model 'model' with the right hand side of the formula
    changed order of arguments for possible other fitting method(?)
    '''
    # T = args[0][1]
    # rhs = args[0][1]
    # print(args[0][1])
    # print(args[0][1])
    # T = [T, Tx, T2x]
    return((model(a, T) - rhs).ravel())
    # return((model(a, T) - rhs).reshape(-1))


# def residual3(a, rhs, t, tx, t2x):
def residual3(a, rhs, T):
    '''
    combining the model 'model' with the right hand side of the formula
    changed order of arguments for possible other fitting method(?)
    '''
    # T = args[0][1]
    # rhs = args[0][1]
    # print(args[0][1])
    # print(args[0][1])
    # T = [t, tx, t2x]
    return(np.sum((model(a, T) - rhs).flatten()))


def residual2_curvefit(T, rhs, a):
    '''
    combining the model 'model' with the right hand side of the formula
    changed order of arguments for possible other fitting method(?)
    '''
    # T = args[0][1]
    # rhs = args[0][1]
    # print(args[0][1])
    # print(args[0][1])
    # T = [T, Tx, T2x]
    # return((model(a, T)).flatten())
    return((model(a, T)))


def heat_cond(T, a):
    '''
    calculate the thermal conductivity as a polynom of the temperature
    for this calculation the best values from the fitting shall be used
    '''
    tc = a[0] \
        + a[1] * T**1 \
        + a[2] * T**2 \
        + a[3] * T**3 \
        + a[4] * T**4 \
        + a[5] * T**5 \
        + a[6] * T**6 \
        + a[7] * T**7 \
        + a[8] * T**8 \
        + a[9] * T**9 \
        + a[10] * T**10 \
        + a[11] * T**11 \
        + a[12] * T**12 \
        + a[13] * T**13 \
        + a[14] * T**14 \
        + a[15] * T**15 \
        + a[16] * T**16 \
        + a[17] * T**17 \
        + a[18] * T**18 \
        + a[19] * T**19 \
        + a[20] * T**20 \
        + a[21] * T**21 \
        + a[22] * T**22 \
        + a[23] * T**23 \
        + a[24] * T**24 \
        + a[25] * T**25 \
        + a[26] * T**26 \
        + a[27] * T**27 \
        + a[28] * T**28 \
        + a[29] * T**29 \
        + a[30] * T**30 \
        + a[31] * T**31 \
        + a[32] * T**32 \
        + a[33] * T**33 \
        + a[34] * T**34 \
        + a[35] * T**35 \
        + a[36] * T**36 \
        + a[37] * T**37 \
        + a[38] * T**38 \
        + a[39] * T**39
    return(tc)


# initiate values
c = 0
rest = []
rhs = []
T = []
Tappend = []
rhs_all = []

# movement of point caused by thermal expansion
# w is the mass per unit length of the specimen
# w = delta * s

initialize_values = False

# initializing values

if initialize_values is True:
    print('Initialize values ...')
    ai = np.array(a0)
    initial_stepwidth = 0.01
    max_iterator = 100

    for i in range(len(a0) - 0):
        # for c in range(0, len(tx)):
        #     tm2 = t[c, :]
        #     txm2 = tx[c, :]
        #     t2xm2 = t2x[c, :]
        #     ttm = tt[c, :]

            # T1 = [tm2, txm2, t2xm2]
        T_all = [t, tx, t2x]

        # rhs_all = (f_dcp(t) * tt + (f_epsht(t) * sigma * p *
                                    # (t**4 - t_a**4)) / s)
        rhs_all = (f_dcp(t) * tt + (f_epsht(t) * sigma * p *
                                    (t**4 - t_a**4)) / s)

        difference0 = residual(a0, rhs_all, T_all)
        # exit()

        takeinitial = False
        # start = True
        iterator = 0
        pos = ''

        while takeinitial is not True and iterator < max_iterator:
            # print(slope)
            # print(takeinitial)
            progress_init = np.int(round(float((i * max_iterator) + (iterator)) /
                                         (max_iterator * len(a0)) * 100, 0))

            print(' initialize: [' + progress_init * '#' +
                  (100 - progress_init) * '.' + ']' + '  ' +
                  '%.1f%%   \r' % (round(float((i * max_iterator) + (iterator)) /
                                   (max_iterator * len(a0)) * 100, 1)),
                  sep=' ', end='', flush=True)

            iterator += 1

            diff_0 = abs(np.mean(residual(ai, rhs_all, T_all)))

            ai_neg = np.array(ai)
            ai_neg[i] = ai[i] - initial_stepwidth

            diff_neg = abs(np.mean(residual(ai_neg, rhs_all, T_all)))

            ai_pos = np.array(ai)
            ai_pos[i] = ai[i] + initial_stepwidth

            diff_pos = abs(np.mean(residual(ai_pos, rhs_all, T_all)))

            # slope = 'up'

            if abs(diff_0 - diff_neg) > abs(diff_pos - diff_0) * 1.00:
                # or diff_pos > diff_0:
                ai[i] -= initial_stepwidth
                if pos is 'down':
                    # iterator = max_iterator
                    takeinitial = True
                    if iterator > 1:
                        ai[i] += initial_stepwidth
                        # print('help')
                    # print('down to up')
                    break
                pos = 'up'
                # print(ai)
                continue
                # ai[i] += initial_stepwidth
                # print(ai)

            if abs(diff_0 - diff_neg) < abs(diff_pos - diff_0) * 1.00:
                # or diff_pos < diff_0:
                ai[i] += initial_stepwidth
                if pos is 'up':
                    # iterator = max_iterator

                    takeinitial = True
                    if iterator > 1:
                        ai[i] -= initial_stepwidth
                        # print('help')
                    # print('up to down')
                    break
                pos = 'down'
                # print(pos)
                continue
                # ai[i] -= initial_stepwidth
                # print(ai)

            # if diff_pos)) > diff_0)):
                # if np.mean(abs(diff2)) - np.mean(abs(diff1)) > \
                #    np.mean(abs(diff3)) - np.mean(abs(diff1)):
                #     diff2 = diff3
                #     slope = 'down'
                # start = False
                # print('help')
                # print(np.mean(abs(diff2)))
            # exit()
            if abs(diff_0 - diff_neg) <= abs(diff_pos - diff_0) * 1.0000 \
                or abs(diff_0 - diff_neg) >= abs(diff_pos - diff_0) * 1 \
               or diff_pos == diff_neg:
                # or iterator >= 1000:
                # ai[i] -= initial_stepwidth
                # print('initial condition satisfied')
                pos = ''
                # a0[i] = ai[i]
                takeinitial = True
                # print(a0)
                # print(ai)
                # exit()
                break
            if iterator >= max_iterator:
                # print('iteration max reached')
                pos = ''
                # a0[i] = ai[i]
                takeinitial = True
                # print(ai)
                break

            # if np.mean(diff2) < np.mean(diff1):
            #     # this is slope = 'down'
            #     diff1 = residual(ai, rhs_all, T_all)
            #     ai[i] = ai[i] + initial_stepwidth
            #     diff2 = residual(ai, rhs_all, T_all)
            #     print('hello')
            #     if slope == 'up':
            #         iterator = max_iterator
            #         print(slope)
            #     slope = 'down'

            # if np.mean(diff2) > np.mean(diff1):
            #     # this is slope = 'up'
            #     diff1 = residual(ai, rhs_all, T_all)
            #     ai[i] = ai[i] - initial_stepwidth
            #     diff2 = residual(ai, rhs_all, T_all)
            #     if slope == 'down':
            #         iterator = max_iterator
            #         print(slope)
            #     slope = 'up'
            # print('diff2')
            # print(np.mean(abs(diff2)))
            # print(np.mean(diff2))
            # print('diff1')
            # print(np.mean(abs(diff1)))
            # print(np.mean(diff1))
            # print(ai)
            # print(a0 + 1000 * initial_stepwidth)

        # print(ai)
    a0 = np.array(ai)
    print('... complete!')

# print(a0)
# exit()
# print(a0)

if verbose is True:
    print('Fitting thermal conductivity ...')

fitmethod = 'lm'

order = '3d'

if ref_file != '':
    den_lit = int1d(den_ref_x, den_ref_y, kind='linear',
                    fill_value='extrapolate')
else:
    den_lit = 7874

if order is 'dx':
    # first dx than dt
    # progbar_dx_fit = pyprind.ProgBar(np.shape(tx)[0] - 1, width=100)
    progbar_dx_fit = pyprind.ProgPercent(np.shape(tx)[0] - 1, track_time=False, title='fit scan')

    for c in range(0, len(tx)):
        '''
        for every profile
        '''
        tm2 = t[c, :]
        txm2 = tx[c, :]
        t2xm2 = t2x[c, :]
        ttm = tt[c, :]

        # print(f_dcp(tm2))
        # print(f_epsht(tm2))
        # print(ttm)
        # print(t2xm2)
        # exit()

        # append values for right hand side to rhs

        # rhs.append(
        #     lit_delta * lit_cp * ttm + (f_epsht(tm2) * sigma * p *
        #                                 (tm2**4 - t_a**4)) / s)
        # rhs1 = (lit_delta * lit_cp * ttm + (f_epsht(tm2) * sigma * p *
        #                                     (tm2**4 - t_a**4)) / s)
        # rhs1 = (f_dcp(tm2) * ttm + (0.3 * sigma * p *
        #                                  (tm2**4 - t_a**4)) / s)

        # progress = np.int(round(float(np.shape(rhs)[0]) / len(tx) * 100, 0))

        use_cal_hc = True

        if use_cal_hc is True:
            # ## reccent working with CALCULATED hc
            rhs.append(
                f_dcp(tm2) * ttm + (f_epsht(tm2) * sigma * p *
                                    (tm2**4 - t_a**4)) / s)
            # ## recent working with CALCULATED hc
        elif use_cal_hc is False:
            # ## reccent working with LITERATURE hc
            if ref_file != '':
                rhs.append(
                    f_hc_ref_y(tm2) * den_lit(tm2) * ttm + (f_epsht(tm2) * sigma * p *
                                                            (tm2**4 - t_a**4)) / s)
            elif ref_file == '':
                rhs.append(
                    lit_delta * lit_cp * ttm + (f_epsht(tm2) * sigma * p *
                                                (tm2**4 - t_a**4)) / s)

        if use_cal_hc is True:
            # ## reccent working with CALCULATED hc
            rhs1 = (f_dcp(tm2) * ttm + (f_epsht(tm2) * sigma * p *
                                        (tm2**4 - t_a**4)) / s)
            # ## reccent working with CALCULATED hc
        elif use_cal_hc is False:
            # ## reccent working with LITERATURE hc
            if ref_file != '':
                rhs1 = (f_hc_ref_y(tm2) * den_lit(tm2) * ttm + (f_epsht(tm2) * sigma * p *
                                                                (tm2**4 - t_a**4)) / s)
            elif ref_file == '':
                rhs1 = (lit_cp * lit_delta * ttm + (f_epsht(tm2) * sigma * p *
                                                    (tm2**4 - t_a**4)) / s)


        # print(rhs1)
        # exit()

        # print('calculating: [' + progress * '#' + \
        #       (100 - progress) * '.' + ']' + '  ' +
        #       '%.1f%%   \r' % (round(float(np.shape(rhs)[0]) /
        #                        len(tx) * 100, 1)),
        #       sep=' ', end='', flush=True)

        # append derivatives of temperature (x direction)
        # to one big temperature variable
        T.append([tm2, txm2, t2xm2])
        T1 = [tm2, txm2, t2xm2]
        # arguments = [rhs1, T1]
        # print(type(rhs1))
        # print(type(T1))
        # print(type(tm2))
        # print(arguments[0][0])
        # residual2(a0, arguments)

        # jacobian = np.tri(len(a0), len(a0), 1, dtype=float)

        # res = minimize(residual, a0,
        #                args=(T1, rhs1),
        #                method='least_squares'
        #                )

        res = least_squares(residual, a0, args=(rhs1, T1),
                            method=fitmethod,
                            # bounds=(53.999999999, 54.0000000001),
                            verbose=2,
                            # jac=callable,
                            # x_scale='jac',
                            # x_scale=1000,
                            # f_scale=10,
                            # max_nfev=10000,
                            # xtol=2.22044604926e-16,
                            # ftol=2.22044604926e-16,
                            # gtol=2.22044604926e-16,
                            # loss='cauchy',
                            # tr_solver='exact'
                            # tr_options=
                            )

        # append the results to variable
        rest.append(res['x'].tolist())

        progbar_dx_fit.update()

        # the trick to get a working leastsq is to set
        # the arguments in 'args' in the correct order
        # same order as in definition of residualction 'residual'

        # up to this point everything ist straight forward
        # now calculate the thermalconductivity from the fit values

        # somehow resetting the calculated values in the given residualction
        # returns values around 10**13 instead of 0

if order is 'dt':
    # first dt than dx
    # progbar_dt_fit = pyprind.ProgBar(np.shape(tt)[1] - 1, width=100)
    progbar_dt_fit = pyprind.ProgPercent(np.shape(tt)[1] - 1, track_time=False, title='fit scan')

    for c in range(0, np.shape(tt)[1]):
        '''
        for every profile
        '''
        tm2 = t[:, c]
        txm2 = tx[:, c]
        t2xm2 = t2x[:, c]
        ttm = tt[:, c]

        # print(f_dcp(tm2))
        # print(f_epsht(tm2))
        # print(ttm)
        # print(t2xm2)
        # exit()

        # append values for right hand side to rhs

        # rhs.append(
        #     lit_delta * lit_cp * ttm + (f_epsht(tm2) * sigma * p *
        #                                 (tm2**4 - t_a**4)) / s)
        # rhs1 = (lit_delta * lit_cp * ttm + (f_epsht(tm2) * sigma * p *
        #                                     (tm2**4 - t_a**4)) / s)
        # rhs1 = (f_dcp(tm2) * ttm + (0.3 * sigma * p *
        #                                  (tm2**4 - t_a**4)) / s)

        use_cal_hc = True

        if use_cal_hc is True:
            # ## reccent working with CALCULATED hc
            rhs.append(
                f_dcp(tm2) * ttm + (f_epsht(tm2) * sigma * p *
                                    (tm2**4 - t_a**4)) / s)
            # ## recent working with CALCULATED hc
        elif use_cal_hc is False:
            # ## reccent working with LITERATURE hc
            if ref_file != '':
                rhs.append(
                    f_hc_ref_y(tm2) * den_lit(tm2) * ttm + (f_epsht(tm2) * sigma * p *
                                                            (tm2**4 - t_a**4)) / s)
            elif ref_file == '':
                rhs.append(
                    lit_delta * lit_cp * ttm + (f_epsht(tm2) * sigma * p *
                                                (tm2**4 - t_a**4)) / s)

        # progress = np.int(round(float(np.shape(rhs)[0]) /
        #                   np.shape(tt)[1] * 100, 0))

        if use_cal_hc is True:
            # ## reccent working with CALCULATED hc
            rhs1 = (f_dcp(tm2) * ttm + (f_epsht(tm2) * sigma * p *
                                        (tm2**4 - t_a**4)) / s)
            # ## reccent working with CALCULATED hc
        elif use_cal_hc is False:
            # ## reccent working with LITERATURE hc
            if ref_file != '':
                rhs1 = (f_hc_ref_y(tm2) * den_lit(tm2) * ttm + (f_epsht(tm2) * sigma * p *
                                                                (tm2**4 - t_a**4)) / s)
            elif ref_file == '':
                rhs1 = (lit_cp * lit_delta * ttm + (f_epsht(tm2) * sigma * p *
                                                    (tm2**4 - t_a**4)) / s)

        # print(rhs1)
        # exit()

        # print('calculating: [' + progress * '#' + \
        #       (100 - progress) * '.' + ']' + '  ' +
        #       '%.1f%%   \r' % (round(float(np.shape(rhs)[0]) /
        #                        np.shape(tt)[1] * 100, 1)),
        #       sep=' ', end='', flush=True)

        # append derivatives of temperature (x direction)
        # to one big temperature variable
        T.append([tm2, txm2, t2xm2])
        T1 = [tm2, txm2, t2xm2]
        # arguments = [rhs1, T1]
        # print(type(rhs1))
        # print(type(T1))
        # print(type(tm2))
        # print(arguments[0][0])
        # residual2(a0, arguments)

        # jacobian = np.tri(len(a0), len(a0), 1, dtype=float)

        # res = minimize(residual, a0,
        #                args=(T1, rhs1),
        #                method='least_squares'
        #                )

        res = least_squares(residual, a0, args=(rhs1, T1),
                            method=fitmethod,
                            # bounds=(53.999999999, 54.0000000001),
                            verbose=2,
                            # jac=callable,
                            # x_scale='jac',
                            # x_scale=0.001,
                            # f_scale=10,
                            max_nfev=10000,
                            # xtol=1e100,
                            # xtol=2.22044604926e-16,
                            # ftol=2.22044604926e-16,
                            # gtol=2.22044604926e-16,
                            # loss='cauchy',
                            # tr_solver='lsmr'
                            # tr_options=
                            )

        # append the results to variable
        rest.append(res['x'].tolist())

        progbar_dt_fit.update()

        # the trick to get a working leastsq is to set
        # the arguments in 'args' in the correct order
        # same order as in definition of residualction 'residual'

        # up to this point everything ist straight forward
        # now calculate the thermalconductivity from the fit values

        # somehow resetting the calculated values in the given residualction
        # returns values around 10**13 instead of 0

if order is '3d':
    # fit in 3D
    # for c in range(0, np.shape(tt)[1]):
    # tm2 = t[:, :]
    # txm2 = tx[:, :]
    # t2xm2 = t2x[:, :]
    # ttm = tt[:, :]

    # append values for right hand side to rhs

    # rhs.append(
    #     f_dcp(tm2) * ttm + (f_epsht(tm2) * sigma * p *
    #                              (tm2**4 - t_a**4)) / s)
    use_cal_hc = True

    if use_cal_hc is True:
        # ## reccent working with CALCULATED hc
        rhs.append(
            f_dcp(t) * tt + (f_epsht(t) * sigma * p *
                             (t**4 - t_a**4)) / s)
        # ## recent working with CALCULATED hc
    elif use_cal_hc is False:
        # ## reccent working with LITERATURE hc
        if ref_file != '':
            rhs.append(
                f_hc_ref_y(t) * den_lit(t) * tt + (f_epsht(t) * sigma * p *
                                                   (t**4 - t_a**4)) / s)
        elif ref_file == '':
            rhs.append(
                lit_delta * lit_cp * tt + (f_epsht(t) * sigma * p *
                                           (t**4 - t_a**4)) / s)
        ## reccent working with LITERATURE hc

    # rhs.append(
    #     lit_delta * lit_cp * tt + (f_epsht(t) * sigma * p *
    #                                (t**4 - t_a**4)) / s)
    # rhs1 = (f_dcp(tm2) * ttm + (f_epsht(tm2) * sigma * p *
    #                                  (tm2**4 - t_a**4)) / s)

    if use_cal_hc is True:
        # ## reccent working with CALCULATED hc
        rhs1 = (f_dcp(t) * tt + (f_epsht(t) * sigma * p *
                                 (t**4 - t_a**4)) / s)
        # ## reccent working with CALCULATED hc
    elif use_cal_hc is False:
        # ## reccent working with LITERATURE hc
        if ref_file != '':
            rhs1 = (f_hc_ref_y(t) * den_lit(t) * tt + (f_epsht(t) * sigma * p *
                                                       (t**4 - t_a**4)) / s)
        elif ref_file == '':
            rhs1 = (lit_cp * lit_delta * tt + (f_epsht(t) * sigma * p *
                                               (t**4 - t_a**4)) / s)
        # ## reccent working with LITERATURE hc

    T1 = [t, tx, t2x]

    res = least_squares(residual2, a0, args=(rhs1, T1),
                        method=fitmethod,
                        # bounds=(0, +np.inf),
                        verbose=0,
                        # jac=callable,
                        # x_scale='jac',
                        # x_scale=100,
                        # f_scale=10,
                        # max_nfev=100000,
                        # xtol=1e100,
                        # xtol=2.22044604926e-16,
                        # ftol=2.22044604926e-16,
                        # gtol=2.22044604926e-16,
                        # loss='linear',
                        # loss='soft_l1',
                        # loss='huber',
                        # loss='cauchy',
                        # loss='arctan',
                        # tr_solver='exact'
                        # tr_options=
                        )

    # res = minimize(residual3, a0, args=(rhs1, t, tx, t2x), method='CG')
    # res = minimize(residual3, a0, args=(rhs1, T1), method='CG')

    # res = fmin(residual3, a0, args=(rhs1, T1))
    print(res)

    # res = curve_fit(residual2_curvefit, T1, rhs1, a0[0:2])

    # bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    # bar = progressbar.ProgressBar()
    # bar(res.nfev)

    # append the results to variable
    rest.append(res['x'].tolist())

    # the trick to get a working leastsq is to set
    # the arguments in 'args' in the correct order
    # same order as in definition of residualction 'residual'

    # up to this point everything ist straight forward
    # now calculate the thermalconductivity from the fit values

    # somehow resetting the calculated values in the given residualction
    # returns values around 10**13 instead of 0



rest2 = [a for a in rest]

# import pprint
# pp = pprint.PrettyPrinter(width=320)
# pp.pprint(np.array(rest2[:][6]))

# print(rest)

# exit()

if verbose is True:
    print('... complete!\n')

    print('Plotting results ...')

temp_range = np.linspace(np.amin(t), np.amax(t) * 0.95, num=len(t))
# plt.show(block)
# plt.plot(temp_range)
# plt.show(block)
# exit()

# print(np.shape(rest2))
# print(rest2)
# print(rest2[0])
# print(np.shape(model(rest2[0], T1)))
# print(np.shape(rhs1))
# print(np.shape(t))
# print(np.shape(T1))
# for u in range(np.shape(T1)[1]):
#     plt.figure('t test')
#     plt.plot(t[u, :])

#     plt.figure('model rhs1')
#     plt.plot(t[u, :], model(rest2[0], [t[u, :], tx[u, :], t2x[u, :]]), label='model')
#     plt.plot(t[u, :], rhs1[u, :], label='rhs1')
#     plt.legend()

#     plt.figure('residuals')
#     plt.plot(t[u, :], residual(rest2[0], rhs1[u, :], [t[u, :], tx[u, :], t2x[u, :]]), 'b-', label='residual')
#     plt.legend()
#     # plt.figure('residual3')

#     plt.figure('tc')
#     plt.plot(t[u, :], heat_cond(t[u, :], rest2[0]), label='tc')
#     if ref_file != '':
#         plt.plot(tc_ref_x, tc_ref_y, label='tc lit.')
#     plt.legend()
#     # if u > 10:
#     #     input()
#     input()
# plt.figure('compare model rhs1')
# plt.plot(t.flatten('C'), model(rest2[0], T1).flatten('C'), 'b-', label='model')
# plt.plot(t.flatten('C'), rhs1.flatten('C'), 'r.', label='rhs1')
# plt.legend()
# # print(np.shape(residual(rest2, rhs1, T1)))
# # print(np.shape(residual2(rest2, rhs1, T1)))
# plt.figure('residuals')
# plt.plot(t.ravel(), residual2(rest2[0], rhs1, T1), 'b-', label='residual2')
# # plt.legend()
# # plt.figure('residual3')
# plt.plot(t.flatten(), residual3(rest2[0], rhs1, T1), 'r-', label='residual3')
# plt.legend()

# input()
# print(t[0])
# plt.figure('compare model rhs1')
# plt.plot(t[0], model(rest2[0], T1)[0], 'b-', label='model')
# plt.plot(t[0], rhs1[0], 'g-', label='rhs1')
# plt.legend()
# print(np.shape(residual(rest2, rhs1, T1)))
# print(np.shape(residual2(rest2, rhs1, T1)))
# plt.figure('residuals')
# plt.plot(t.ravel(), residual2(rest2[0], rhs1, T1), 'b-', label='residual2')
# # plt.legend()
# # plt.figure('residual3')
# plt.plot(t.flatten(), residual3(rest2[0], rhs1, T1), 'r-', label='residual3')
# plt.legend()

# plt.figure('tc')
# plt.plot(temp_range, heat_cond(temp_range, rest2[0]), label='tc')
# if ref_file != '':
#     plt.plot(tc_ref_x, tc_ref_y, label='tc lit.')
# plt.legend()
# input()
# exit()

tc = []

for c in range(1, np.shape(rest2)[0]):
    tc.append(heat_cond(temp_range, rest[c]))

tc_a0 = heat_cond(temp_range, a0)

# plt.close('all')
lines = ['-', '--', '-.', ':']
linecycler = cycle(lines)

plotdata = False

if plotdata is True:
    if i > 1:
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

    fig, ((ax1, ax4), (ax3, ax2)) = \
        plt.subplots(2, 2, sharex=True, sharey=False, figsize=(15, 10))

    # if order is 'dt':
    #     ende = np.shape(rhs)[0]
    # if order is 'dx':
    #     ende = np.shape(rhs)[0]
    #     # print(ende)
    # if order is '3d':
    #     print('help')
    #     ende = np.shape(rhs)[0]
    ende = np.shape(rhs)[0]
    # print(ende)

    # print(np.shape(T1))
    # print(np.shape(t))
    # exit()

    for f in range(ende):
        # print(f)
        plot1_progress = np.int(round(float(f + 1) /
                                      ende * 100, 0))

        print('   plotting: [' + plot1_progress * '#' +
              (100 - plot1_progress) * '.' + ']' +
              '  ' + '%.1f%%   \r' % (round(float(f + 1) /
                                      ende * 100, 1)),
              sep=' ', end='', flush=True)

        style = next(linecycler)

        if order is 'dt':
            print(np.shape(t_raw))
            ax1.plot(t[:, f], t[:, f],
                     style,
                     label=f)
            ax2.plot(T[f][0], rhs[f].T,
                     style,
                     label=f)
            ax3.plot(T[f][0], model(rest[f], T[f]),
                     style,
                     label=f)
            ax4.plot(T[f][0], np.add(model(rest[f], T[f]), - rhs[f]),
                     style,
                     label=f)

        if order is '3d':
            # ax1.plot(T1[0][f], t[f],
            #          style,
            #          label=f)
            # ax1.legend()
            # ax2.plot(T1[0][f], rhs[f][1],
            #          style,
            #          label=f)
            # ax2.legend()
            ax1.plot(T1[0], t,
                     style,
                     label=f)
            ax2.plot(T1[0], rhs[:][0],
                     style,
                     label=f)
            # print('\n')
            # print(np.shape(T1))
            # print(np.shape(rest2[f]))
            # print(np.shape(T1[0][f]))
            # print(np.shape(model(rest2[f], T1[:][f])))
            # ax3.plot(T1[0][f], model(rest2[f], T1[:][f]),
            #          style,
            #          label=f)
            # print(np.shape(T1))
            # print(np.shape(rest2[0]))
            ax3.plot(T1[0], model(rest2[0], T1),
                     style,
                     label=f)
            # ax3.plot(T1[0][f], np.add(model(T1[0][:][f], rest2[f]), - rhs[f] * 0)[1],
            #          style,
            #          label=f)
            # ax4.plot(T1[0][f], model(rest2[f], T1[:][f]) - rhs[f][0],
            #          style,
            #          label=f)
            ax4.plot(T1[0], model(rest2[0], T1[:]) - rhs[:][0],
                     style,
                     label=f)

        if order is 'dx':

            print(np.shape(t))
            print(np.shape(t_raw))
            print(np.shape(rhs))

            ax1.plot(t[f, :], t[f, :],
                     style,
                     label=f)
            ax2.plot(t[f, :], rhs[f],
                     style,
                     label=f)
            ax3.plot(t[f, :], model(rest[f], T[f]),
                     style,
                     label=f)
            ax4.plot(t[f, :], np.add(model(rest[f], T[f]), - rhs[f]),
                     style,
                     label=f)

    ax4.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax1.set_title('filt_d')

    ax2.set_title('rhs')
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))

    ax3.set_title('lhs')
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))

    ax4.set_title('Ende')
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))

    # plt.show(block)


plot_tc = True

if plot_tc is True:
    ax_tc = plt.figure('thermal conductivity')
    all_heat_cond = np.empty([len(temp_range), len(rest)])

    for j in range(len(rest)):
        tmax = np.amax(t)
        all_heat_cond[:, j] = np.piecewise(temp_range,
                                           [temp_range <= tmax,
                                            temp_range > tmax],
                                           [1, np.nan]) * \
            heat_cond(temp_range, rest[j])
        all_heat_cond[:, j][all_heat_cond[:, j] < 0] = 0.0

        # plot2_progress = np.int(round(float(j + 1) / len(rest) * 100, 0))

        # print('   plotting: [' + plot2_progress * '#' +
        #       (100 - plot2_progress) * '.' + ']' +
        #       '  ' + '%.1f%%   \r' % (round(float(j + 1) /
        #                               len(rest) * 100, 1)),
        #       sep=' ', end='', flush=True)

    m_heat_cond = np.nanmean(all_heat_cond, axis=1)
    max_heat_cond = np.nanmax(all_heat_cond, axis=1)
    min_heat_cond = np.nanmin(all_heat_cond, axis=1)

    # print(m_heat_cond)
    if use_cal_hc is True:
        bhc = 'cal_hc'
    elif use_cal_hc is False:
        bhc = 'lit_hc'

    plt.plot(temp_range, m_heat_cond, linewidth=2.0,
             label='data; fs=' + str(fs) + '; fl=' + str(fl) + '\n' + bhc +
                   '\nleft:   ' + str(round(T_l, 2)) + '\nright: ' + str(round(T_r, 2)))
    # plt.plot(temp_range, 3*m_heat_cond, 'y', linewidth=2.0, label='data')
    # plt.plot(temp_range, heat_cond(temp_range, rest[0]), 'g', linewidth=1.0, label='data')
    # plt.fill_between(temp_range, min_heat_cond, max_heat_cond,
    #                  facecolor='black', alpha=0.5)
    plt.fill_between(temp_range, m_heat_cond * 1.05, m_heat_cond * 0.95,
                     facecolor='black', alpha=0.5)
    # plt.ylim(ymin=0)
    # plt.axhline(y=lit_lambda, color='black')  # , label='sim. param.')
    # plt.plot(temp_range, lit_lambda / (t_a / temp_range), color='black')  # , label='sim. param.')
    # plt.plot(temp_range, lit_lambda * (t_a / temp_range), color='black')  # , label='sim. param.')

    # plt.plot(temp_range, tc_a0, 'y', label='initial_guess')
    # plt.plot(temp_range, heat_cond(temp_range, np.mean(rest, axis=0)),
    #          label='mean rest')

    plt.title(r'thermal conductivity $k$ / $W \cdot m^{-1} \cdot K^{-1}$' + ' \n' + file)
    plt.xlabel('temperature / K')
    plt.ylabel(r'thermal conductivity $k$ / W $\cdot m^{-1} \cdot K^{-1}$')
    plt.legend()

    plt.grid(True)

    if ref_file != '':
        plt.plot(tc_ref_x, tc_ref_y, label='tc lit.')

if verbose is True:
    print("... plots shown ...")
    # plt.show()
    print('... complete!\n')

print(np.mean(rest, axis=0))

if export_to_csv is True:
    print("... export to csv ...")

    print(" export comlete!\n")

input('Press [ENTER] to continue: ')
