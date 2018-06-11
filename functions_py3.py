def getcalibration(pyroname):
    import platform
    import sys
    import pandas as pd
    import numpy as np

    # value = 0
    # err = 0
    pyroname = pyroname.upper()
    directory = ''
    filename = ''

    # specify directory and filename to get the calibration data from

    if platform.system() == 'Windows':
        directory = 'i:\THERMISCHE ANALYSE\Messmethoden\PISA\ '\
                    'PISA_Labor\Kalibrationsmessungen'
        # changed directory for testing on Linux Laptop
        # which is not connected to ZAE
        filename = directory + '\_Kalibration_Pyrometer.txt'
    elif platform.system() == 'Linux':
        # directory = ''
        # use directory to set path where "_Kalibration_Pyrometer.txt"
        # can be found, if same folder -> directory = ''
        directory = '/home/mgessner/PythonCode/'
        filename = directory + '_Kalibration_Pyrometer.txt'
    else:
        sys.exit('It is recommened to use Linux! (or Windows if you have to)')
    # print(filename)
    # read the data using pandas
    data = pd.read_csv(filename, delimiter='\t', decimal='.', engine='c')

    value = np.array(data['value/mV'][data['pyrometer'] == pyroname])
    error = np.array(data['error/mV'][data['pyrometer'] == pyroname])
    # pyrometers = data['pyrometer'].index[pyroname.upper()]
    # print(pyrometers[pyrometers == pyroname])

    value /= 1000   # change from mV to V
    error /= 1000

    # print(type(pyrometers))

    if pyroname in data['pyrometer']:
        print(data['value/mV'])

    return (value, error)
# passt


def cal_pyrotemp(rawdata, pyrometer):
    import numpy as np

    C_2 = 0.014388  # m*K      constant
    T_90_Cu = 1357.77  # K     melting plateau of pure copper
    pyrometer = pyrometer.upper()

    rawdata = abs(rawdata)

    # each pyrometer has its own effective wavelenth
    # which is needed for the calculation later on
    if pyrometer == 'PVM':
        Lambda = 925.2  # nm
    elif pyrometer == 'PV':
        Lambda = 903.1  # nm
    elif pyrometer == 'PUV':
        Lambda = 904.4  # nm
    else:
        Lambda = 900.0  # nm

    # get value at the melding plateau for choosen
    # pyromter from calibration file
    # #calibration data is in mV measured data in V so "/ 1000"
    (U_T90_Cu, sdU_T90_Cu) = getcalibration(pyrometer)
    # print(U_T90_Cu)
    # U_T90_Cu = U_T90_Cu / 1000
    # sdU_T90_Cu = sdU_T90_Cu / 1000

    Lambda = Lambda * 1e-9  # m
    # convert wavelenth from nm to m

    q = rawdata / U_T90_Cu   # scalar division to get simplified variable q

    # calculate real temperature from comparison with calibrated values
    # and convert it to numpy array type: float64
    T_90 = (C_2 / Lambda) / np.log(np.array(abs(((np.exp(C_2 /
                                            (Lambda * T_90_Cu)) + q - 1) / q)),
                                            dtype='float64'))

    return T_90
# passt


def cal_Cp(temp_pv, rho, time, current, s, delta, p, t_a):
    #             T = T_PV;
    #           rho = R_SPEC;
    #             t = time;
    #    epsilon_ht = 1 for Blackbody;
    #       current = CURRENT;

    #    int i = 1;
    #    double sigma;

    # initialize values for temperature, time and heatcapacity
    # T_a = T_a + 23; #ambient temperature at 23 C -> 296.15 K

    # as the pv looks into the blackbody hole in the specimen
    import numpy as np
    epsilon_ht = 1

    sigma = 5.670367e-8  # W m^-2 K^-4

    # [S,delta,p,T_a]=dialog2();

    # calculate discrete derivative of temperature
    dtemp_pv = np.diff(temp_pv)
    dtemp_pv[dtemp_pv == 0] = 0.0000000001
    dtemp_pv = np.append(dtemp_pv, dtemp_pv[-1])
    # dtemp(abs(dT)>=100) = 0;

    dtime = np.diff(time)  # and of time
    dtime[dtime == 0] = 0.0000000001
    dtime = np.append(dtime, dtime[-1])

    dtimedtemp = np.divide(dtime, dtemp_pv)
    # calculate the inverse of the first derivative of temperature over time
    # print(len(dtime))

    # calculate the heatcapacity
    Cp = dtimedtemp / delta * (((rho * (current**2)) /
                                (s**2)) - (epsilon_ht * sigma * p *
                                           ((temp_pv**4) - (t_a**4)) / s))

    return Cp
# obsolete as delta_cp_and_e_ht is now in order


def getmaterial():
    # get materialdata from file
    import sys
    # import os
    from Tkinter import Tk as tk
    import tkFileDialog as fd
    from Tkinter import Message
    from Tkinter import mainloop

    # name = ''

    def Msg(name):
        master = tk()
        master.title('Error message')
        msg = Message(master, text='unable to load ' + name + ' from file')
        # msg.config()
        msg.pack()
        # close_button = Button(master, text='OK', command=master.destroy)
        mainloop()

    window = tk()
    # window.title()
    window.withdraw()
    filename = fd.askopenfilename(filetypes=[('Text files', '*.txt')],
                                  initialdir='/home/mgessner/'
                                             'PythonCode/Materialien',
                                  title='Get materialdata from file:')
    window.destroy()

    if filename == ():
        exit()

    file = open(filename, 'r')
    # materialname = file.read(1)
    data = file.readlines()
    file.close()

    if data == []:
        sys.exit('File to load was empty, please choose another file!')

    data = list(map(str.strip, data))

    # print(len(data))

    materialname = data[0]

    if data[3].find('Length') != -1:
        length = float(data[3].split('\t')[0].replace('D', 'E'))
    else:
        length = float('nan')
        Msg('Length')
    if data[4].find('Crosssection') != -1:
        crosssection = float(data[4].split('\t')[0].replace('D', 'E'))
    else:
        crosssection = float('nan')
        Msg('Crosssection')
    if data[5].find('Density') != -1:
        density = float(data[5].split('\t')[0].replace('D', 'E'))
    else:
        density = float('nan')
        Msg('Density')
    if data[6].find('Perimeter') != -1:
        perimeter = float(data[6].split('\t')[0].replace('D', 'E'))
    else:
        perimeter = float('nan')
        Msg('Perimeter')
    if data[7].find('ambient Temperature') != -1:
        ambient_temperature = float(data[7].split('\t')[0].replace('D', 'E'))
    else:
        ambient_temperature = float('nan')
        Msg('ambient Temperature')

    return [materialname, length, crosssection, density, perimeter,
            ambient_temperature]
# passt


def epsht(time, t_pv, t_puv):
    # calculate total hemispherical emissivity
    # by comparing pv and puv data
    # eps_ht_pv = 1
    import numpy as np
    sigma = 5.670367 * 10**-8  # W m^-2 K**-4
    sigma = sigma * 10**-3  # W mm^-2 K**-4
    [name, length, crosssection, density, perimeter, t_a] = getmaterial()
    # t_pv_t = np.diff(t_pv)
    t_pv = np.array(t_pv[:-1])
    # t_puv_t = np.diff(t_puv)
    t_puv = np.array(t_puv[:-1])

    '''
    assumptions:

    current = 0

    we are only focusing on one point
    in the middle of the specimen -> dT/dx = 0

    we are only focusing on one point in time -> dT/dt = 0

    specimen has fixed perimeter, cross section

    the only thing that changes is \varepsilon_ht
    '''
    epsht = (t_pv**4 - t_a**4) / (t_puv**4 - t_a**4)

    return epsht
# obsolete as delta_cp_and_e_ht is now in order


def defaultplot(title, xlabel, ylabel, xvalues, legend=''):
    import matplotlib.pyplot as plt
    # int figurenumber
    # if figurenumber != 0:
    #     plt.figure(figurenumber)
    # else:
    if 'legend' in locals():
        if legend != '':
            plt.legend([legend])
    # plt.figure()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xlim(min(xvalues), max(xvalues))
# passt


def delta_cp_and_e_ht(raw_puv=[], raw_time=[], raw_pv=[],
                      raw_pv_time=[], plotresult=False):
    import numpy as np
    import matplotlib.pyplot as plt
    # import sys
    # import platform
    from scipy import interpolate
    # from functions import getcalibration
    # from functions import cal_pyrotemp

    if raw_puv == []:
        raw_puv = np.load('raw_dat.npy')
    if raw_time == []:
        raw_time = np.load('raw_time.npy')
    if raw_pv == []:
        raw_pv = np.load('raw_pv.npy')
    if raw_pv_time == []:
        raw_pv_time = np.load('raw_pv_time.npy')

    sigma = 5.670367 * 10**(-8)  # W m-2 K-4
    # p = 20 * 10**(-3)  # m
    # s = p**2 * np.pi  # m**2

    puv_means = []
    cpt = []

    c_puv = []
    c_pv = []
    c_time = []
    c_pv_time = []

    # print(np.shape(raw_time))

    for i in range(np.shape(raw_time)[1]):
        # print(np.shape(raw_time[:,i]))
        # puv_means = np.append(puv_means, (np.shape(raw_time[:, i])[0] / 2))
        puv_means = np.shape(raw_time[:, i])[0] / 2
        cpt = np.append(cpt, raw_time[int(puv_means), i])
        c_pv_time = np.append(c_pv_time, raw_pv_time[cpt[i] == raw_pv_time])
        c_pv = np.append(c_pv, raw_pv[cpt[i] == raw_pv_time])
        c_puv = np.append(c_puv, raw_puv[cpt[i] == raw_time])
        c_time = np.append(c_time, raw_time[cpt[i] == raw_time])

    # if real values from pyrometers are achieved
    # they have to be converted to kelvin!!!
    # c_pv_90 = cal_pyrotemp(c_pv, 'pv')
    # c_puv_90 = cal_pyrotemp(c_puv, 'puv')
    # for demovalues no convertion is required
    c_puv_90 = c_puv
    c_pv_90 = c_pv
    # print(c_puv_90)
    # print(cpt)
    # # print(c_pv_90)
    # print('bla')
    # print(c_pv_time)
    # print(c_pv)
    # print(c_puv)
    # print(c_time)
    # print('blub')
    # print(c_time)
    # print(np.diff(c_time))
    # print(c_pv_time)

    # print(np.shape(c_pv_time))
    # print(np.shape(c_pv))
    # print(np.shape(c_puv))
    # print(np.shape(c_time))

    # plt.plot(c_time, c_puv_90)
    # plt.plot(c_time, c_pv_90)
    # plt.show()

    # t_a = 300  # K

    def cal_delta_cp(puv_90, pv_90, time, t_a=300, p=20 * 10 ^ (-3)):
        if t_a not in locals():
            t_a = 300

        if p not in locals():
            p = 20 * 10**(-3)  # m

        s = p**2 * np.pi  # m**2

        dt_puv_90 = np.diff(puv_90) / np.diff(time)
        puv_90_s = puv_90[:len(np.diff(puv_90))]
        # print(np.shape(np.diff(pv_90)))
        # print(np.shape(np.diff(time)))
        dt_pv_90 = np.diff(pv_90) / np.diff(time)
        pv_90_s = pv_90[:len(np.diff(pv_90))]

        I_pv = temp_to_intensity(pv_90_s, 900)
        I_puv = temp_to_intensity(puv_90_s, 900)

        epsht = I_puv / I_pv

        # epsht = (pv_90_s**4 - t_a**4) / (puv_90_s**4 - t_a**4) * \
        #     dt_puv_90 / dt_pv_90
        # print(dt_puv_90)
        # print(dt_pv_90)

        f_epsht = interpolate.interp1d(puv_90_s, epsht, kind='linear',
                                       fill_value='extrapolate')

        delta_cp = (-1) * (dt_pv_90)**(-1) * 1 * sigma * p * \
                   (pv_90_s**4 - t_a**4) / s

        f_delta_cp = interpolate.interp1d(puv_90_s, delta_cp, kind='linear',
                                          fill_value='extrapolate')

        # calculate epsht from delta_cp values!!!

        # delta_cp = (-1) * (dt_puv_90)**(-1) * f_epsht(puv_90_s) * sigma
        #             * p * (puv_90_s**4 - t_a**4) / s

        # f_delta_cp = interpolate.interp1d(puv_90_s, delta_cp, kind='linear',
        #                                   fill_value='extrapolate')

        return(epsht, delta_cp, f_epsht, f_delta_cp)

    (epsht, delta_cp, f_epsht, f_delta_cp) = cal_delta_cp(c_puv_90, c_pv_90,
                                                          c_time,
                                                          t_a=300, p=0.02)

    # print(delta_cp)
    # print(dir(interpolate.interp1d))
    # print(f_epsht.item())
    # print(c_puv_90)
    # print(f_epsht(1800), f_epsht(1800.0001), f_epsht(1801))

    # plt.plot(c_puv_90, fun_epsht(c_puv_90), 'r.')
    # plt.show()
    # exit()

    # # print(min(c_puv_90))

    # temp_epsht = np.arange(np.ceil(min(c_puv_90))
    #                        np.floor(max(c_puv_90)), 0.1)
    # y_epsht = fun_epsht(temp_epsht)
    # print(fun_epsht(2000))

    # # print(type(epsht_all))
    # plt.figure('epsht_all')
    if plotresult is True:
        newrange = np.linspace(min(c_puv_90), max(c_puv_90))
        plt.figure('total hemispherical emissivity')
        # plt.plot(newrange, f_epsht(newrange), label='epsht')
        plt.plot(c_puv_90[:-1], epsht, 'r.', label='data')
        plt.legend()

        plt.figure('volumetric heat capacity')
        # plt.plot(newrange, f_delta_cp(newrange), label='delta_cp')
        plt.plot(c_puv_90[:-1], delta_cp, 'r.', label='data')
        plt.legend()
        # plt.show()
        plt.show(block=False)

    return(epsht, delta_cp, f_epsht, f_delta_cp, c_time)

# (a, b, c, d) = delta_cp_and_e_ht()
# print(a,b,c(1400),c(2300),d)
# passt


def choose_cutoff(data, time):
    import matplotlib.pyplot as plt
    import numpy as np
    # import Tkinter as tk
    import sys
    from tkinter import Spinbox, Tk, Button, LEFT  # , mainloop
    # from Tkinter import *

    global choose
    choose = True
    # i = 0
    global cut
    cut = 115
    # global l_line
    # l_line = plt.axvline(cut)

    def combine_funcs(*funcs):
        def combined_func(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)
        return combined_func

    def destroywindows(**kwargs):
        for m in kwargs:
            m.destroy()
        for f in kwargs:
            f.close()
        # master.destroy()

    def setvalues(var, value):
        var = value
        return var

    def choosefalse():
        global choose
        choose = False
        # print(choose)

    def spintocut():
        global cut
        cut = int(cutspin.get())
        # print(cut)

    # def callback():
        # print("something")
        # global l_line
        # l_line(cut)
        # l_line = plt.axvline(cut)
        # plt.draw()
        # plt.ion()
        # plt.pause(0.05)

    # plt.figure('data')
    # plt.plot(data)
    # plt.show(block=False)
    # # plt.draw()

    while choose is True:
        plt.figure('data')
        plt.plot(data)
        plt.axvline(cut)
        plt.axvline(np.shape(data)[0] - cut)
        # plt.draw()
        plt.show(block=False)
        # plt.ion()
        # plt.pause(0.05)

        # print(choose)
        # print(cut)

        master = Tk()

        # print(type(data[0]))

        cutspin = Spinbox(master, from_=0, to=np.shape(data)[0],
                          textvariable=cut)  # , command=callback)
        cutspin.delete(0)
        cutspin.insert(0, cut)

        cutspin.pack()
        # print(type(cutspin))
        # print(cutspin.get())
        applyButton = Button(master, text='apply',
                             command=combine_funcs(spintocut,
                                                   plt.close,
                                                   master.destroy))
        applyButton.pack(side=LEFT)
        goonButton = Button(master, text='go on',
                            command=combine_funcs(master.destroy,
                                                  plt.close,
                                                  choosefalse))
        goonButton.pack(side=LEFT)
        quitButton = Button(master, text='quit',
                            command=sys.exit)
        quitButton.pack(side=LEFT)
        goonButton.focus_set()

        master.mainloop()
        # Tk.update()
        # plt.update()
        # plt.show(block=False)

        # l_left.remove()
        # l_right.remove()
        # plt.draw()

    cutoff = cut

    return cutoff

# import numpy as np
# filt_dat = np.load("filt_dat.npy")
# print(type(filt_dat[0,0]))
# print(np.shape(filt_dat))
# time = np.load("raw_time.npy")
# blub = choose_cutoff(filt_dat, time)
# print(blub)


def getdata(rawdata):
    import numpy as np
    from functions_py3 import window_function
    from scipy.signal import savgol_filter
    from functions_py3 import choose_cutoff
    from functions_py3 import window_function_2

    # lowercase all column names
    rawdata.columns = map(str.lower, rawdata.columns.str.strip())
    # print(rawdata.columns)

    # ln_all = np.arange(len(rawdata))
    puv_all = np.array(rawdata['puv'])
    pv_all = np.array(rawdata['pv'])
    time_all = np.array(rawdata['time'])
    if 'current' in rawdata:
        current_all = np.array(rawdata['current'])
    else:
        current_all = []
    if 'r_spec' in rawdata:
        r_spec_all = np.array(rawdata['r_spec'])
    else:
        r_spec_all = []


    # (nsplit, desc, high, low) = window_function_2(puv_all, 310, 360,
    #                                             plotdata=True,
    #                                             printlist=False)

    # exit()

    (nsplit, desc, high, low) = window_function(puv_all, 310, 360,
                                                plotdata=False,
                                                printlist=False)
    nsplit = np.insert(nsplit, 0, 0)
    nsplit = np.append(nsplit, len(puv_all))

    maxnsplit = np.int(max(np.diff(high)))

    # initiate lists to fill with splitted profiles
    raw_dat = list(range(len(high[:]) - 1))
    raw_time = list(range(len(high[:]) - 1))
    filt_dat = list(range(len(high[:]) - 1))

    # define the cutoff from each side
    # cutoff = 50
    # print(np.shape(puv_all))

    # define ambient temperature
    T_a = 300  # K

    for i in range(0, len(high) - 1):
        #
        raw_dat[i] = np.array([puv_all[high[i][0]:high[i][1]]])
        raw_time[i] = np.array([time_all[high[i][0]:high[i][1]]])

        # flatten the splitted data
        raw_dat[i] = raw_dat[i].flatten()
        raw_time[i] = raw_time[i].flatten()

        # make it the same length, all of them
        lendiff = maxnsplit - len(raw_dat[i])
        raw_dat[i] = np.insert(raw_dat[i], 0,
                               np.ones(np.int(lendiff / 2)) * T_a)
        raw_dat[i] = np.append(raw_dat[i], np.ones(np.int(lendiff / 2)) * T_a)
        raw_dat[i] = np.resize(raw_dat[i], (maxnsplit))

        raw_time[i] = np.lib.pad(raw_time[i],
                                 (int(lendiff / 2), int(lendiff / 2)),
                                 'reflect', reflect_type='odd')
        raw_time[i] = np.resize(raw_time[i], (maxnsplit))
        # print(np.shape(raw_time))
        # print(np.shape(raw_dat))
        # cut of several values in the beginning and in the end

        # smooth the data by using filter on them
        filt_dat[i] = savgol_filter(raw_dat[i], 15, 3, axis=0, mode='mirror')
        # link: https://en.wikipedia.org/wiki/Savitzky-Golay_filter
        # uncomment next line to get rid of the filter
        # filt_d[i] = raw_d[i]

    raw_dat = np.asarray(raw_dat).T
    raw_time = np.asarray(raw_time).T
    raw_pv = pv_all
    raw_pv_time = time_all
    filt_dat = np.asarray(filt_dat).T

    # print(np.shape(filt_dat))
    # # print(filt_dat[0][:])
    # # print(filt_dat[:][0])
    # print(type(filt_dat[0]))
    # plt.plot(filt_dat)
    # # plt.plot(filt_dat[:][1])
    # plt.show()
    # exit()
    cutoff = choose_cutoff(filt_dat, raw_time)

    raw_dat = raw_dat[cutoff:-cutoff]
    raw_time = raw_time[cutoff:-cutoff]
    filt_dat = filt_dat[cutoff:-cutoff]

    # for i in range(0, len(high) - 1):
    #     raw_dat[i] = raw_dat[i][cutoff:-cutoff]
    #     raw_time[i] = raw_time[i][cutoff:-cutoff]
    #     filt_dat[i] = filt_dat[i][cutoff:-cutoff]

    # raw_dat = np.asarray(raw_dat).T
    # raw_time = np.asarray(raw_time).T
    # raw_pv = pv_all
    # raw_pv_time = time_all
    # filt_dat = np.asarray(filt_dat).T
    # print(np.shape(raw_time))
    # print(np.shape(filt_dat))

    return(raw_dat, raw_time, raw_pv, raw_pv_time, filt_dat,
           current_all, r_spec_all)
# passt


def window_function(rawdata, lowerbound, upperbound,
                    printlist=False, plotdata=False):
    # printlist = False
    # plotdata = False
    # assumption: data starts low with peaks to higher values
    import numpy as np
    import matplotlib.pyplot as plt
    # import scipy

    # lowerindex = np.where(a[i:] > lowerbound)[0]
    # output = np.array(())
    firstupper = 0
    firstlower = 0
    lastupper = 0
    lastlower = 0
    # global middle
    middle = np.array(())
    middle = np.append(middle, 0)
    high = []
    low = []

    # condlist_l = [rawdata >= lowerbound]
    # condlist_u = [rawdata <= upperbound]

    # output_l = np.select(condlist_l, [rawdata])
    # output_u = np.select(condlist_u, [rawdata])

    # output = output_l + output_u - rawdata

    # # print(output)

    # lineoutput = np.where(rawdata == output)
    # print(lineoutput)

    slope = 'up'
    # global start
    start = 0
    step = 3
    stop = False
    description = {}
    # exit()
    i = 0
    j = 0

    if rawdata[0] <= lowerbound:
        slope = 'up'
        description['0 start at'] = 0
    elif rawdata[0] >= upperbound:
        slope = 'down'
        description['0 start'] = 0
    elif rawdata[0] > lowerbound and rawdata[0] < upperbound:
        middle = np.append(middle, 0)
        if (np.where(rawdata >= upperbound)[0][0] >
           np.where(rawdata <= lowerbound)[0][0]):
            slope = 'up'
        elif (np.where(rawdata >= upperbound)[0][0] <
              np.where(rawdata <= lowerbound)[0][0]):
            slope = 'down'

    # while i <= len(rawdata)

    while stop is False:
        if slope == 'up':
            # print(np.where(rawdata[start:] >= upperbound))

            try:
                firstupper = np.where(rawdata[start:] >=
                                      upperbound)[0][0] + start
            except IndexError:
                stop = True
                break
            # print(firstupper)

            try:
                lastlower = np.where(rawdata[start:firstupper] <=
                                     lowerbound)[0][-1] + start
            except IndexError:
                stop = True
                break
            # print(lastlower)

            middle = np.append(middle, (firstupper + lastlower) / 2)
            # print(type(middle[0]))
            start = firstupper + step
            slope = 'down'
            i += 1
            description[str(i) + '. ' + slope + ' to'] = np.int64(middle[-1])
            # low = np.append(low, [(middle[-2], middle[-1])])
            low.append((np.int(middle[-2]), np.int(middle[-1])))
            # print((firstupper + lastlower) / 2)
            # print(middle)
            # exit()

        elif slope == 'down':
            # print(start)
            try:
                firstlower = np.where(rawdata[start:] <=
                                      lowerbound)[0][0] + start
            except IndexError:
                stop = True
                break
            # print(firstlower)

            try:
                lastupper = np.where(rawdata[start:firstlower] >=
                                     upperbound)[0][-1] + start
            except IndexError:
                stop = True
                break
            # print(lastupper)
            # print(start, firstlower)

            middle = np.append(middle, (firstlower + lastupper) / 2)
            start = firstlower + step
            slope = 'up'
            j += 1
            description[str(j) + '.  ' + slope + ' to '] = np.int64(middle[-1])
            # high = np.append(high, (middle[-2], middle[-1]))
            high.append((np.int(middle[-2]), np.int(middle[-1])))
            # print(middle)
            # exit()

    # middle = (firstupper + lastlower) / 2
    # print(middle)
    # choicelist = [rawdata, rawdata]

    # output = np.select(condlist, choicelist)

    # print(output)

    middle = middle.astype(np.int64)
    # print(type(middle[0]))

    # print('high', high[1])
    # print('low', low)

    # printlist = False

    if printlist is True:
        for x in sorted(description.items()):
            print(str(x) + '\n')

    # plotdata = False

    if plotdata is True:
        plt.plot(rawdata)
        # plt.plot((0, len(rawdata)), (lower, lower))
        plt.axhline(y=lowerbound, color='g')
        plt.axhline(y=upperbound, color='r')
        for i in range(np.shape(middle)[0]):
            plt.axvline(x=middle[i], color='b')
        plt.show()

    return(middle, description, high, low)
# passt


def bb_cavity(e_w, l, d):
    import numpy as np
    e_z = np.longdouble()
    e_z = 1. - (1. - e_w) / e_w * 1. / (1. + (2. * l / d)**2)

    return e_z


def special_bb_cavity(e_w, l, d):
    import numpy as np
    e_z = np.longdouble()
    # l = np.longdouble()
    # d = np.longdouble()
    # e_w = np.longdouble()
    # e_z = (e_w + 1. - (1. - e_w) / e_w * 1. / (1. + (2. * l / d))) / 2.
    e_z = (1. - (1. - e_w) / e_w * 1. /
           (1. + (2. * l / d)**2)) * (1. - (d / l)) \
        + e_w * (d / l)

    return e_z


def special_bb_cavity_d_l(e_w, d_l):
    import numpy as np
    e_z = np.array(())
    # l = np.longdouble()
    # d = np.longdouble()
    # e_w = np.longdouble()
    # e_z = (e_w + 1. - (1. - e_w) / e_w * 1. / (1. + (2. * l / d))) / 2.
    e_z = (1. - (1. - e_w) / e_w * 1. /
           (1. + (2. * (d_l)**(-1))**2)) * (1. - d_l) \
        + e_w * d_l

    return e_z
# passt


def cal_thermocouple(voltage, typ):
    ''' calculate temperatures from voltages in thermocouples

    IMPORTANT: values to be calculated in uV (micro volt)!!! '''
    import numpy as np
    import pandas as pd
    from scipy import interpolate
    import sys

    # if type == 'C':
    #     file = 'Thermoelement_Typ_C_Reference_2.txt'
    # elif type == 'J':
    #     file = ''
    # print(type(typ))
    # typ = 'T'

    file2 = 'Referenzdaten.txt'

    maxlines = 0

    with open(file2) as f1:
        maxlines = sum(1 for _ in f1)

    with open(file2) as f:
        for start, line in enumerate(f, 1):
            if typ in line:
                # print(start)
                break
        for stop, line in enumerate(f, start):
            # print(line.strip())
            if line.strip() == '':
                # print(stop)
                break

    if start == '' or start == maxlines:
        print('Please check type of thermocouple entered!')
        sys.exit()

    data = pd.read_csv(file2, delimiter='\t', header=None,
                       engine='c', decimal=',',
                       skiprows=start, nrows=stop - start)

    value = np.array(())
    temp_C = np.array(())

    data = np.array(data)

    for r in range(1, np.shape(data)[0]):
        for c in range(1, np.shape(data)[1]):
            value = np.append(value, data[r, c])
            temp_C = np.append(temp_C, data[0, c] + int(data[r, 0]))

    temp_C = temp_C[~np.isnan(value)]
    value = value[~np.isnan(value)]

    temp = temp_C  # + 273.15  # in Kelvin

    f_temp = interpolate.interp1d(value, temp, bounds_error=False)

    return f_temp(voltage)


def cut_current_no_zero(rawdata):
    import pandas as pd
    import numpy as np

    if 'current' in rawdata:
        # print(rawdata.index(current < 0))
        split_at_off = rawdata['current'].diff().idxmin()
        split_at_on = rawdata['current'].diff().idxmax()

        # print(split_at_on)
        # print(split_at_off)

        before_pulse = rawdata[:split_at_on]
        current_no_zero = rawdata[split_at_on:split_at_off]
        after_pulse = rawdata[split_at_off:]

        # print(current_no_zero.tail())
        # print(after_pulse.head())

        # exit()

        return(before_pulse, current_no_zero, after_pulse)

    else:
        return(0, 0, rawdata)


def dt_to_dx(t, l, d, plotit=False):
    import numpy as np
    import matplotlib.pyplot as plt

    # assumption as 't' is at least a 2D array separated peaks

    # print(np.shape(t))
    # print(t[0,1])
    # print(t[2,0])
    # print(t)

    # exit()

    # max_alpha = np.degrees(np.arctan(l / (2 * d)))
    max_alpha = np.tan(l / (2 * d))
    # max_x = np.tan(max_alpha / 180 * np.pi) * d
    # dt = np.diff(t)
    x_out = np.empty_like(t, dtype=np.double)
    t_mean = np.array(())
    t_start = np.array(())
    t_stop = np.array(())

    # print(np.ndim(t))
    # print(t[0][17])

    for c in range(np.shape(t)[0]):
        t_mean = np.append(t_mean, (t[c, -1] - t[c, 0]) / 2)
        # print(t[c, -1])
        # t_start = np.append(t_start, t[c, 0])
        # t_center = np.append(t_center)
        # t_stop = np.append(t_stop, t[c, -1])

    t_start = t[0, :]
    # print(t_start)
    t_center = np.mean(t_mean)
    t_number = np.mean(np.diff(t_start))  / np.mean(np.diff(t, axis=0))
    # print(t_number)

    # print(np.mean(np.diff(t, axis=0)))
    # d_alpha = 360 / t_number  # d_alpha in degrees
    d_alpha = 2. * np.pi / t_number  # d_alpha in rad
    # print(d_alpha)

    # alpha = np.linspace

    # print(np.mean(np.diff(t_start)))
    # print(t_start)
    # print(t_number)
    # print(d)
    # print(t)
    for r in range(np.shape(x_out)[0]):
        actual_alpha = np.double(- max_alpha / 2. + max_alpha
                                 * np.divide(r, np.shape(x_out)[0]))
        # print(np.tan(actual_alpha))
        # print(np.divide(r, t_number))
        x_out[r, :] = np.double(2. * d * np.tan(actual_alpha))
        # x_out[r] = d * np.tan(- max_alpha / 2. + max_alpha * r / np.shape(x_out)[0])
        # print(x_out[:,r])
        # print(d * np.tan(actual_alpha))
    # print(x_out)
        # print(np.tan(actual_alpha))
    # print(x_out)

    # print(np.tan(- max_alpha / 2.) * 2. * d)
    # print(np.tan(max_alpha / 2.) * 2. * d)

    # if np.ndim(t) == 1:
    #     alpha = np.linspace(-max_alpha, max_alpha, num=len(t) , endpoint=True)
    #     # x = d * np.tan(alpha / 180 * np.pi)
    #     x_out = d * np.tan(alpha / 180 * np.pi)
    #     print('one')

    # elif np.ndim(t) > 1:
    #     for c in range(np.shape(t)[1]):
    #         # t_mean = np.append(t_mean, t[c,-1] - t[c,0])
    #         # print(c)
    #         # print(t[-1,c])
    #         # exit()
    #         # print(t[-1:,c])
    #         # print(t[0,c])

    #         for r in range(np.shape(t)[0]):
    #             # alpha = np.linspace(0, max_alpha, num=len(t) / 2, endpoint=True)
    #             alpha = np.linspace(-max_alpha, max_alpha, num=len(t) , endpoint=True)
    #             # print(alpha[0])
    #             # print(alpha[1])
    #             # x[c] = d * np.tan(alpha / 180 * np.pi)
    #             # x_out[r,c] = d * np.tan(alpha[r] / 180 * np.pi)
    #             # x_r = d * np.tan((180 + alpha) / 180 * np.pi)
    #             x_out = np.append(x_out, d * np.tan(alpha / 180 * np.pi))
    #             # x_out = np.append(x, l - x[::-1][::])
    #             # print(x_out[:,c])
    # print(t_center)
    # print(t_number)
    # print(len(x_out))
    # print(len(t))
    # print(t_start)
    # exit()
    # print(x_out)
    x_out = (l / 2.) + x_out
    # print(l)
    # print(x_out)

    if plotit is True:
        plt.plot(x_out)
        plt.show()

    return(x_out)


def window_function_2(rawdata, lowerbound, upperbound,
                    printlist=False, plotdata=False):
    # printlist = False
    # plotdata = False
    # assumption: data starts low with peaks to higher values
    import numpy as np
    import matplotlib.pyplot as plt
    # import scipy
    from scipy.signal import argrelextrema as arelext

    # lowerindex = np.where(a[i:] > lowerbound)[0]
    # output = np.array(())
    firstupper = 0
    firstlower = 0
    lastupper = 0
    lastlower = 0
    # global middle
    middle = np.array(())
    middle = np.append(middle, 0)
    high = []
    low = []


    extrema = arelext(rawdata, np.greater, order=1000)
    print(extrema)

    # condlist_l = [rawdata >= lowerbound]
    # condlist_u = [rawdata <= upperbound]

    # output_l = np.select(condlist_l, [rawdata])
    # output_u = np.select(condlist_u, [rawdata])

    # output = output_l + output_u - rawdata

    # # print(output)

    # lineoutput = np.where(rawdata == output)
    # print(lineoutput)

    slope = 'up'
    # global start
    start = 0
    step = 3
    stop = False
    description = {}
    # exit()
    i = 0
    j = 0

    if rawdata[0] <= lowerbound:
        slope = 'up'
        description['0 start at'] = 0
    elif rawdata[0] >= upperbound:
        slope = 'down'
        description['0 start'] = 0
    elif rawdata[0] > lowerbound and rawdata[0] < upperbound:
        middle = np.append(middle, 0)
        if (np.where(rawdata >= upperbound)[0][0] >
           np.where(rawdata <= lowerbound)[0][0]):
            slope = 'up'
        elif (np.where(rawdata >= upperbound)[0][0] <
              np.where(rawdata <= lowerbound)[0][0]):
            slope = 'down'

    # while i <= len(rawdata)

    while stop is False:
        if slope == 'up':
            # print(np.where(rawdata[start:] >= upperbound))

            try:
                firstupper = np.where(rawdata[start:] >=
                                      upperbound)[0][0] + start
            except IndexError:
                stop = True
                break
            # print(firstupper)

            try:
                lastlower = np.where(rawdata[start:firstupper] <=
                                     lowerbound)[0][-1] + start
            except IndexError:
                stop = True
                break
            # print(lastlower)

            middle = np.append(middle, (firstupper + lastlower) / 2)
            # print(type(middle[0]))
            start = firstupper + step
            slope = 'down'
            i += 1
            description[str(i) + '. ' + slope + ' to'] = np.int64(middle[-1])
            # low = np.append(low, [(middle[-2], middle[-1])])
            low.append((np.int(middle[-2]), np.int(middle[-1])))
            # print((firstupper + lastlower) / 2)
            # print(middle)
            # exit()

        elif slope == 'down':
            # print(start)
            try:
                firstlower = np.where(rawdata[start:] <=
                                      lowerbound)[0][0] + start
            except IndexError:
                stop = True
                break
            # print(firstlower)

            try:
                lastupper = np.where(rawdata[start:firstlower] >=
                                     upperbound)[0][-1] + start
            except IndexError:
                stop = True
                break
            # print(lastupper)
            # print(start, firstlower)

            middle = np.append(middle, (firstlower + lastupper) / 2)
            start = firstlower + step
            slope = 'up'
            j += 1
            description[str(j) + '.  ' + slope + ' to '] = np.int64(middle[-1])
            # high = np.append(high, (middle[-2], middle[-1]))
            high.append((np.int(middle[-2]), np.int(middle[-1])))
            # print(middle)
            # exit()

    # middle = (firstupper + lastlower) / 2
    # print(middle)
    # choicelist = [rawdata, rawdata]

    # output = np.select(condlist, choicelist)

    # print(output)

    middle = middle.astype(np.int64)
    # print(type(middle[0]))

    # print('high', high[1])
    # print('low', low)

    # printlist = False

    print(middle)
    print(start)

    if printlist is True:
        for x in sorted(description.items()):
            print(str(x) + '\n')

    # plotdata = False

    if plotdata is True:
        plt.plot(rawdata)
        # plt.plot((0, len(rawdata)), (lower, lower))
        plt.axhline(y=lowerbound, color='g')
        plt.axhline(y=upperbound, color='r')
        for i in range(np.shape(middle)[0]):
            plt.axvline(x=middle[i], color='b')
        plt.show()




    # plt.plot(rawdata)
    # plt.show()

    return(middle, description, high, low)
# passt


def data_to_function(*data, xlabel='X', ylabel='Y', plotit=True):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tkinter import Tk as tk
    from tkinter import filedialog as fd
    from scipy.signal import savgol_filter
    # from scipy import interpolate
    from scipy.interpolate import interp1d as intp

    # print(data == ())

    # headernames = (xlabel, ylabel)

    if data == ():
        window = tk()
        window. withdraw()
        file = fd.askopenfilename(filetypes=[('CSV files', '*.csv'),
                                             ('Text files', '*.txt')],
                                  initialdir='/home/mgessner/Bilder/WebPlots')
        window.destroy()
        if file == ():
            exit()
        data = pd.read_csv(file, delimiter='\t', header=None, engine='c',
                           decimal='.')

        data.rename(columns={data.columns[0]: xlabel}, inplace=True)
        data.sort_values([xlabel], ascending=True, inplace=True)
        # data.columns[0].rename = 'c'
        # print(data.columns.values[0])

        f_data_array = []

        for i in range(1, len(data.columns)):
            # print(ylabel + str(i))
            # data.columns.values[i] = str(ylabel + str(i))
            data.rename(columns={data.columns[i]: ylabel + str(i)}, inplace=True)

            data[ylabel + str(i)] = savgol_filter(data[ylabel + str(i)], 21, 5,
                                                  mode='nearest')
            X = data[xlabel].values
            Y = data[ylabel + str(i)].values

            f_data = intp(Y, X, bounds_error=False)
            
            print(f_data(100))

            f_data_array.append(f_data)

            if plotit is True:
                plt.plot(data[xlabel], data[ylabel + str(i)])
            # plt.show
        # print(data.columns)

        if plotit is True:
            plt.show()

    return(f_data)


    # print(data)


def sim_delta_cp_and_e_ht(raw_puv=[], raw_time=[], raw_pv=[],
                          raw_pv_time=[], litdata=[], plotresult=False):
    import numpy as np
    import matplotlib.pyplot as plt
    # import sys
    # import platform
    from scipy import interpolate
    # from functions import getcalibration
    # from functions import cal_pyrotemp

    # if raw_puv == []:
    #     raw_puv = np.load('raw_dat.npy')
    # if raw_time == []:
    #     raw_time = np.load('raw_time.npy')
    # if raw_pv == []:
    #     raw_pv = np.load('raw_pv.npy')
    # if raw_pv_time == []:
    #     raw_pv_time = np.load('raw_pv_time.npy')

    sigma = 5.670367e-8  # W m-2 K-4

    lit_delta = litdata[0]
    lit_cp = litdata[1]
    lit_epsht = litdata[2]
    # p = 20 * 10**(-3)  # m
    # s = p**2 * np.pi  # m**2

    puv_means = []
    cpt = []

    c_puv = []
    c_pv = []
    c_time = []
    c_pv_time = []

    c_puv = raw_puv[:, np.int(np.shape(raw_puv)[1] / 2)]
    c_pv = raw_pv[:, np.int(np.shape(raw_pv)[1] / 2)]
    c_time = raw_time[:, np.int(np.shape(raw_time)[1] / 2)]
    c_pv_time = raw_pv_time[:, np.int(np.shape(raw_pv_time)[1] / 2)]

    # if real values from pyrometers are achieved
    # they have to be converted to kelvin!!!
    # c_pv_90 = cal_pyrotemp(c_pv, 'pv')
    # c_puv_90 = cal_pyrotemp(c_puv, 'puv')
    # for demovalues no convertion is required
    c_puv_90 = c_puv
    c_pv_90 = c_pv

    split = 1.0
    pulse_stop = 0.4

    # print(c_puv)

    # c_puv = c_puv[raw_time[0] <= split]
    # c_puv = c_puv[raw_time[0] >= pulse_stop]
    # c_pv = c_pv[raw_time[0] <= split]
    # c_pv = c_pv[raw_time[0] >= pulse_stop]
    # c_time = c_time[raw_time[0] <= split]
    # c_time = c_time[raw_time[0] >= pulse_stop]
    # c_pv_time = c_pv_time[raw_time[0] <= split]
    # c_pv_time = c_pv_time[raw_time[0] >= pulse_stop]

    # print(np.shape(c_puv))

    # t_a = 300  # K

    def sim_cal_delta_cp(puv_90, pv_90, time, t_a=300, p=7.6e-3 * 2 * np.pi):
        if t_a not in locals():
            t_a = 300

        if p not in locals():
            p = 7.6e-3 * 2 * np.pi  # m

        s = ((7.6e-3)**2 - (6e-3)**2) * np.pi  # m**2
        sigma = 5.670367e-8  # W m-2 K-4

        # dt_puv_90 = np.diff(puv_90) / np.diff(time)
        # print(np.shape(puv_90))
        dt_puv_90 = np.gradient(puv_90, edge_order=1) / np.gradient(time, edge_order=1)
        # print(time)
        # print(np.diff(puv_90))
        # puv_90_s = puv_90[:len(np.diff(puv_90))]
        puv_90_s = puv_90[:len(np.gradient(puv_90, edge_order=1))]
        # center_puv = np.int(len(puv_90_s) / 2)
        # puv_90_c = puv_90[center_puv]
        # print(puv_90_c)
        # print(np.shape(np.diff(pv_90)))
        # exit()
        # print(np.shape(np.diff(time)))
        # dt_pv_90 = np.diff(pv_90) / np.diff(time)
        # pv_90_s = pv_90[:len(np.diff(pv_90))]
        dt_pv_90 = np.gradient(pv_90, edge_order=1) / np.gradient(time, edge_order=1)
        pv_90_s = pv_90[:len(np.gradient(pv_90, edge_order=1))]
        # center_pv = np.int(len(pv_90_s) / 2)
        # pv_90_c = pv_90[center_pv]
        # print(np.diff(pv_90))
        # print(np.diff(time))
        # print(dt_pv_90**(-1))
        # print(np.diff(time) / np.diff(pv_90))
        # exit()
        # print(np.shape(pv_90_s))

        I_pv = temp_to_intensity(pv_90_s, 900e-9, 1.0)
        I_puv = temp_to_intensity(puv_90_s, 900e-9, lit_epsht)

        epsht = I_puv / I_pv
        # print(pv_90_s)
        # print(puv_90_s[1:5])
        # print(puv_90_s[1:5]**4)
        # exit()

        # #to generate plot over whole range set split in "tc_sim_p3_V2.py" to 10 or very high
        # plt.plot(time, pv_90_s, 'r', label=r'black body cavity $\varepsilon = 1$')
        # plt.plot(time, puv_90_s, 'b', label=r'back of specimen $\varepsilon = 0.3$')
        # # plt.plot(puv_90_s)
        # plt.grid(True)
        # plt.xlabel('time / s')
        # plt.ylabel('temperature / K')
        # plt.title('temperature history')
        # # plt.xlim([0, 0.6])
        # plt.legend()
        # # plt.grid(True)
        # plt.show()
        # exit()

        # import matplotlib.pyplot as plt

        # plt.plot(time, dt_pv_90**(-1))
        # plt.plot(time, 1 / dt_pv_90)
        # plt.show()
        # exit()

        # epsht = (pv_90_s**4 - t_a**4) / (puv_90_s**4 - t_a**4) * \
        #     dt_puv_90 / dt_pv_90

        # print(dt_puv_90 / dt_pv_90)
        # print(dt_pv_90**(-1))

        f_epsht = interpolate.interp1d(puv_90_s, epsht, kind='linear',
                                       fill_value='extrapolate')

        delta_cp = (-1) * (dt_pv_90)**(-1) * 1 * \
            sigma * p * (pv_90_s**4 - t_a**4) / s

        delta_cp[delta_cp < 0.] = 0.
        # print(pv_90_s)
        # print(delta_cp)
        # print(lit_delta * lit_cp)
        # exit()
        delta_cp_puv = (-1) * (dt_puv_90)**(-1) * f_epsht(puv_90_s) * \
            sigma * p * (puv_90_s**4 - t_a**4) / s

        delta_cp_puv[delta_cp_puv < 0.] = 0.

        # print(puv_90_s)
        # print(delta_cp_puv)
        # print((dt_puv_90)**(-1))
        # print(np.diff(time))
        # plt.plot(pv_90_s, pv_90_s**4)
        # plt.plot(time, -dt_pv_90**(-1) * pv_90_s**4 * sigma)

        # plt.axhline(y=lit_delta * lit_cp)
        # plt.show()
        # exit()

        # print(delta_cp)

        f_delta_cp = interpolate.interp1d(pv_90_s, delta_cp,
                                          kind='linear',
                                          fill_value='extrapolate'
                                          # fill_value=(min(delta_cp), max(delta_cp))
                                          )

        f_delta_cp_puv = interpolate.interp1d(puv_90_s, delta_cp,
                                              kind='linear',
                                              fill_value='extrapolate'
                                              # fill_value=(min(delta_cp_puv), max(delta_cp_puv))
                                              )

        # calculate epsht from delta_cp values!!!
        
        # plt.plot(pv_90_s, delta_cp)
        # plt.figure()
        # plt.plot(pv_90_s, f_delta_cp(pv_90_s))
        # plt.show()
        # exit()


        # delta_cp = (-1) * (dt_puv_90)**(-1) * f_epsht(puv_90_s) * sigma
        #             * p * (puv_90_s**4 - t_a**4) / s

        # f_delta_cp = interpolate.interp1d(puv_90_s, delta_cp, kind='linear',
        #                                   fill_value='extrapolate')

        return(epsht, delta_cp, f_epsht, f_delta_cp,
               delta_cp_puv, f_delta_cp_puv)

    (epsht, delta_cp, f_epsht, f_delta_cp, delta_cp_puv, f_delta_cp_puv) = \
        sim_cal_delta_cp(c_puv_90, c_pv_90, c_time, t_a=300,
                         p=7.6e-3 * 2 * np.pi)

    # print(delta_cp)
    # print(dir(interpolate.interp1d))
    # print(f_epsht.item())
    # print(c_puv_90)
    # print(f_epsht(1800), f_epsht(1800.0001), f_epsht(1801))

    # plt.plot(c_puv_90, fun_epsht(c_puv_90), 'r.')
    # plt.show()
    # exit()

    # # print(min(c_puv_90))

    # temp_epsht = np.arange(np.ceil(min(c_puv_90))
    #                        np.floor(max(c_puv_90)), 0.1)
    # y_epsht = fun_epsht(temp_epsht)
    # print(fun_epsht(2000))

    # print(min(c_puv))
    # exit()
    # # print(type(epsht_all))
    # plt.figure('epsht_all')
    # print(min(c_pv_90))
    # print(max(c_pv_90))
    # print(plotresult)

    if plotresult is True:
        # newrange = np.linspace(min(c_puv_90), max(c_pv_90))
        plt.figure('normal emissivity')
        plt.title(r'normal emissivity $\varepsilon_{n}$')
        plt.plot(c_puv_90, epsht, 'r.', label='data')
        plt.fill_between(c_puv_90[:],
                         epsht * 0.95,
                         epsht * 1.05,
                         facecolor='black', alpha=0.5)
        # plt.plot(newrange, f_epsht(newrange), label='lit.')
        # plt.axhline(y=lit_epsht, label='sim. param.')
        plt.ylim([lit_epsht - 0.1, lit_epsht + 0.1])
        # plt.axhline(y=1)
        plt.xlabel('temperature / K')
        plt.ylabel(r'normal emissivity $\varepsilon_{n}$ / a.u.')
        plt.legend()
        plt.grid(True)

        # # newrange1 = np.linspace(min(c_puv_90), max(c_pv_90))
        # plt.figure('volumetric heat capacity')
        # # plt.title(r'volumetric heat capacity $C_p$')
        # plt.plot(c_pv_90, delta_cp, 'g.', label='data')
        # # plt.plot(c_puv_90[:], delta_cp_puv, 'r.', label='data_puv')
        # # plt.plot(newrange, f_delta_cp(newrange), label='sim. param.')
        # plt.fill_between(c_pv_90,
        #                  delta_cp * 0.95,
        #                  delta_cp * 1.05,
        #                  facecolor='black', alpha=0.5)
        # # plt.fill_between(c_puv_90[:],
        # #                  delta_cp_puv * 0.95,
        # #                  delta_cp_puv * 1.05,
        # #                  facecolor='black', alpha=0.5)
        # plt.axhline(y=lit_delta * lit_cp, label='sim. param.')
        # # print(lit_delta * lit_cp)
        # # plt.ylim([0, lit_delta * lit_cp * 1.05])
        # plt.legend()
        # # plt.ylim(ymin=0)
        # plt.grid(True)
        # plt.xlabel('temperature / K')
        # plt.ylabel(r'volumetric heat capacity $C_p$ / $J \cdot m^{-3} \cdot K^{-1}$')
        # plt.show(0)
        # plt.show(block=False)

        # plt.show()
        # exit()
    return(epsht, delta_cp, f_epsht, f_delta_cp, c_time)


def sim_plot_temp_history(raw_puv=[], raw_time=[], raw_pv=[],
                          raw_pv_time=[], litdata=[], plotresult=False,
                          voltage_off=0):
    import numpy as np
    import matplotlib.pyplot as plt
    # import sys
    # import platform
    from scipy import interpolate
    # from functions import getcalibration
    # from functions import cal_pyrotemp

    # if raw_puv == []:
    #     raw_puv = np.load('raw_dat.npy')
    # if raw_time == []:
    #     raw_time = np.load('raw_time.npy')
    # if raw_pv == []:
    #     raw_pv = np.load('raw_pv.npy')
    # if raw_pv_time == []:
    #     raw_pv_time = np.load('raw_pv_time.npy')

    sigma = 5.670367e-8  # W m-2 K-4

    lit_delta = litdata[0]
    lit_cp = litdata[1]
    lit_epsht = litdata[2]
    # p = 20 * 10**(-3)  # m
    # s = p**2 * np.pi  # m**2

    puv_means = []
    cpt = []

    c_puv = []
    c_pv = []
    c_time = []
    c_pv_time = []

    c_puv = raw_puv[:, np.int(np.shape(raw_puv)[1] / 2)]
    c_pv = raw_pv[:, np.int(np.shape(raw_pv)[1] / 2)]
    c_time = raw_time[:, np.int(np.shape(raw_time)[1] / 2)]
    c_pv_time = raw_pv_time[:, np.int(np.shape(raw_pv_time)[1] / 2)]

    # if real values from pyrometers are achieved
    # they have to be converted to kelvin!!!
    # c_pv_90 = cal_pyrotemp(c_pv, 'pv')
    # c_puv_90 = cal_pyrotemp(c_puv, 'puv')
    # for demovalues no convertion is required
    c_puv_90 = c_puv
    c_pv_90 = c_pv

    if plotresult is True:
        # to generate plot over whole range set split in "tc_sim_p3_V2.py" to 10 or very high
        plt.figure('temp history')
        plt.title('temperature history')
        plt.plot(c_time, c_pv_90, 'r', label=r'black body cavity $\varepsilon = 1$')
        plt.plot(c_time, c_puv_90, 'b', label=r'back of specimen $\varepsilon = 0.3$')
        # plt.plot(puv_90_s)
        plt.grid(True)
        plt.xlabel('time / s')
        plt.ylabel('temperature / K')
        # print(voltage_off)
        if voltage_off > 0:
            plt.axvline(c_time[voltage_off])
        # plt.title('temperature history')
        # plt.xlim([0, 0.6])
        # plt.ylim(0, np.inf)
        # plt.legend()
        # plt.grid(True)
        # plt.show(0)
        # exit()

    return 0


def temp_to_intensity(temp, wl, e_ht, n=1):
    import numpy as np
    # description
    if wl not in locals():
        wl = 900e-9  # nm            wavelength
    # constants
    h = 6.6261e-34  # J*s        Planck constant
    k_B = 1.3807e-23  # J*K**-1  Boltzman constant
    c_0 = 2.9979e8  # m*s**-1    lightspeed in vacuum

    intensity = np.empty_like(temp)
    intensity = (2 * h * c_0**2) / (n**2 * wl**5) * \
        (1 / (np.exp((h * c_0) / (n * k_B * wl * temp)) - 1)) * e_ht

    return(intensity)


def temp_to_intensity2(temp, wl, f_e_ht, n=1):
    import numpy as np
    # description
    if wl not in locals():
        wl = 900e-9  # nm            wavelength
    # constants
    h = 6.6261e-34  # J*s        Planck constant
    k_B = 1.3807e-23  # J*K**-1  Boltzman constant
    c_0 = 2.9979e8  # m*s**-1    lightspeed in vacuum

    intensity = np.empty_like(temp)
    intensity = (2 * h * c_0**2) / (n**2 * wl**5) * \
        (1 / (np.exp((h * c_0) / (n * k_B * wl * temp)) - 1)) * f_e_ht(temp)
    
    np.nan_to_num(intensity, copy=False)
    if np.isnan(intensity).any() == True:
        print(intensity)
        input()

    return(intensity)


def intensity_to_temp(intensity, wl, e_ht, n=1):
    import numpy as np
    # description
    if wl not in locals():
        wl = 900e-9  # nm            wavelength
    # constants
    h = 6.6261e-34  # J*s        Planck constant
    k_B = 1.3807e-23  # J*K**-1  Boltzman constant
    c_0 = 2.9979e8  # m*s**-1    lightspeed in vacuum

    # print(e_ht)

    temp = np.empty_like(intensity)
    temp = h * c_0 / (wl * k_B * n * np.log((2 * h * c_0**2 * e_ht /
                                            (intensity * wl**5 * n**2)) + 1))

    return(temp)


def intensity_to_temp2(intensity, wl, f_e_ht_int, n=1):
    import numpy as np
    from scipy import interpolate
    # description
    if wl not in locals():
        wl = 900e-9  # nm            wavelength
    # constants
    h = 6.6261e-34  # J*s        Planck constant
    k_B = 1.3807e-23  # J*K**-1  Boltzman constant
    c_0 = 2.9979e8  # m*s**-1    lightspeed in vacuum

    # f_epsht = interpolate.interp1d(f_e_ht_int_int, f_e_ht_int, kind='linear',
    #                                fill_value='extrapolate')
    # print(f_e_ht_int(intensity))
    # print(f_epsht(f_e_ht_int_int))
    # exit()
    temp = np.empty_like(intensity)
    temp = h * c_0 / (wl * k_B * n * np.log((2 * h * c_0**2 * f_e_ht_int(intensity) /
                                            (intensity * wl**5 * n**2)) + 1))
    # print(temp)
    np.nan_to_num(temp, copy=False)
    if np.isnan(temp).any() == True:
        print(np.argwhere(np.isnan(temp)))
        print('temp')
        input()

    return(temp)


def cal_cp_from_current(rho, length, raw_puv=[], raw_time=[], raw_voltage=[]):
    import numpy as np
    from scipy import interpolate
    from scipy.interpolate import UnivariateSpline
    # current =
    # DTdt1 = np.gradient(raw_puv, edge_order=1) / \
    #     np.gradient(raw_time, edge_order=1)

    max_temp = np.max(raw_puv)
    voltage_off = (np.argwhere(np.abs(raw_voltage) > 0)[-1]) + 1
    # print(max_temp)

    temp_heat = np.split(raw_puv, voltage_off)[0]
    time_heat = np.split(raw_time, voltage_off)[0]
    temp_cool = np.split(raw_puv, voltage_off)[1]
    time_cool = np.split(raw_time, voltage_off)[1]
    voltage_heat = np.split(raw_voltage, voltage_off)[0]
    rho_heat = np.split(rho, voltage_off)[0]


    f_temp_heat = interpolate.interp1d(temp_heat, temp_heat, kind='linear',
                                       fill_value='extrapolate')
    f_temp_cool = interpolate.interp1d(temp_cool, temp_cool, kind='linear',
                                       fill_value='extrapolate')
    f_time_heat = interpolate.interp1d(time_heat, time_heat, kind='linear',
                                       fill_value='extrapolate')
    f_time_cool = interpolate.interp1d(time_cool, time_cool, kind='linear',
                                       fill_value='extrapolate')
    f_voltage_heat = interpolate.interp1d(temp_heat, voltage_heat,
                                          kind='linear',
                                          fill_value='extrapolate')
    f_rho = interpolate.interp1d(temp_heat, rho_heat, kind='linear',
                                 fill_value='extrapolate')
    # print(time_heat)
    # print(temp_cool)
    import matplotlib.pyplot as plt

    # plt.plot(temp_cool, temp_cool, 'b.')
    # plt.plot(temp_cool, f_temp_cool(temp_cool), 'g-')
    # plt.plot(temp_heat, temp_heat, 'y.')
    # plt.plot(temp_heat, f_temp_heat(temp_heat), 'r-')
    # plt.show()
    # exit()

    # exit()
    # DTdtheat = UnivariateSpline(time_heat, temp_heat).derivative()
    # DTdtcool = UnivariateSpline(time_cool, temp_cool).derivative()
    # DTdtcool = UnivariateSpline(time_cool, temp_cool).derivative()
    DTdtheat = interpolate.interp1d(temp_heat,
                                    (np.gradient(temp_heat, edge_order=1) /
                                     np.gradient(time_heat, edge_order=1)),
                                    kind='quadratic')
    DTdtcool = interpolate.interp1d(temp_cool,
                                    (np.gradient(temp_cool, edge_order=1) /
                                     np.gradient(time_cool, edge_order=1)),
                                    kind='quadratic')
    # DTdtcool = np.gradient(f_temp_cool, edge_order=1) / \
    #     np.gradient(f_time_cool, edge_order=1)

    # print(f_temp_heat(1000))
    # exit()


    # plt.plot(temp_heat, (np.gradient(temp_heat, edge_order=1) /
    #     np.gradient(time_heat, edge_order=1)), 'b.')
    # plt.plot(temp_heat, DTdtheat(temp_heat), 'g-')
    # # plt.plot(temp_heat, temp_heat, 'y.')
    # # plt.plot(temp_heat, f_temp_heat(temp_heat), 'r-')
    # plt.show()
    # exit()

    # print(max_temp)
    # print(DTdtheat(max_temp-1))
    # exit()
    # print(f_voltage_heat(0.99 * max_temp))

    tempcp = np.linspace(0.95 * max_temp, 0.99 * max_temp)
    # print(np.shape(temp_to_cal))
    # print(type(f_voltage_heat(0.99 * max_temp)))

    # delta_cp_new = np.empty_like(np.linspace(0.9 * max_temp, 0.99 * max_temp))
    delta_cp_new = np.array(()) #np.empty_like(temp_to_cal)
    dcpnew = np.array(())

    # for tempcp in range(temp_to_cal):
    for i in range(len(tempcp)):
        # print(tempcp[i])
        # DTdtheat = UnivariateSpline(f_time_heat(tempcp), f_temp_heat(tempcp)).derivative()
        # DTdtcool = UnivariateSpline(f_time_cool(tempcp), f_temp_cool(tempcp)).derivative()
        #here scalar products and derivatives!!!!
        v = f_voltage_heat(tempcp[i])
        dh = DTdtheat(tempcp[i])
        # print(DTdtheat(tempcp[i]))
        dc = DTdtcool(tempcp[i])
        r = f_rho(tempcp[i])
        dcpnew = np.append(dcpnew, - v**2 / (r * length**2) / (dh - dc))
        delta_cp_new = np.append(delta_cp_new, f_voltage_heat(tempcp[i])**2 / (f_rho(tempcp[i]) * length**2) * 1 / \
                   (DTdtheat(tempcp[i]) - DTdtcool(tempcp[i])))
                   # (np.gradient(f_temp_cool(tempcp)) / np.gradient(f_time_cool(tempcp)) \
                   # - np.gradient(f_temp_heat(tempcp)) / np.gradient(f_time_heat(tempcp)))
    # print(f_voltage_heat(tempcp))
    # print(DTdtheat(tempcp))
    # print(DTdtcool(tempcp))
    # print(np.shape(delta_cp_new))
    # print(dcpnew)
    # exit()

    # print(dh)
    # print(dc)

    # print(np.shape(dcpnew))
    # print(np.shape(tempcp))

    plt.plot(tempcp, dcpnew)
    lit_cp = 265        # specific heat capacity J / (kg * K)
    lit_delta = 8570    # density kg / m**3
    plt.axhline(lit_cp * lit_delta)
    plt.show()
    exit()

    # print(DTdtheat)
    # print(DTdtcool)


def cal_cp_from_current2(rho, length, raw_pyro=[], raw_time=[], raw_voltage=[]):
    import numpy as np
    from scipy import interpolate
    from scipy.interpolate import interp1d as int1d
    # from scipy.interpolate import UnivariateSpline as US
    # from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    import matplotlib.pyplot as plt
    # from scipy.signal import savgol_filter as sgf
    # from scipy.optimize import curve_fit

    import warnings
    warnings.simplefilter('ignore', np.RankWarning)
    # current =
    # DTdt1 = np.gradient(raw_pyro, edge_order=1) / \
    #     np.gradient(raw_time, edge_order=1)
    # T_a = 300  # K

    max_temp = np.max(raw_pyro)
    # min_temp = np.min(raw_pyro)
    max_temp_point = (np.argwhere(raw_pyro == np.max(raw_pyro)))[0]
    # print(max_temp)

    # print(raw_voltage)
    # exit()
    voltage_off = (np.argwhere(np.abs(raw_voltage) > 0)[-1])
    voltage_on = (np.argwhere(np.abs(raw_voltage) > 0)[1]) - 1

    # if max_temp_point > voltage_off:
    #     voltage_off = max_temp_point

    # print(voltage_on)
    # print(voltage_off)
    # print(raw_voltage)
    # print(raw_voltage[voltage_on])
    # print(raw_voltage[voltage_off])
    # print(np.abs(raw_voltage[voltage_off]))
    # exit()
    # voltage_off = np.where(np.roll(raw_voltage[:, 1], 1) != raw_voltage[:, 1])[0]

    DTdt = np.gradient(raw_pyro, edge_order=0) / \
        np.gradient(raw_time, edge_order=0)

    # plt.figure("temp")
    # plt.plot(DTdt,'.')
    # input()
    # exit()

    temp_heat = np.split(raw_pyro, voltage_off)[0]
    temp_curr = np.split(temp_heat, voltage_on)[1]
    temp_cool = np.split(raw_pyro, voltage_off)[1]

    min_temp = np.min(temp_cool)

    time_heat = np.split(raw_time, voltage_off)[0]
    time_curr = np.split(time_heat, voltage_on)[1]
    time_cool = np.split(raw_time, voltage_off)[1]

    voltage_heat = np.split(np.abs(raw_voltage), voltage_off)[0]
    voltage_curr = np.split(voltage_heat, voltage_on)[1]

    rho_heat = np.split(rho, voltage_off)[0]
    rho_curr = np.split(rho_heat, voltage_on)[1]

    DTdt_heat = np.split(DTdt, voltage_off)[0]
    DTdt_curr = np.split(DTdt_heat, voltage_on)[1]
    DTdt_cool = np.split(DTdt, voltage_off)[1]

    # print(temp_curr)
    # print(rho_curr)
    # print(np.split(raw_voltage, voltage_off)[0])
    # exit()

    # temp_heat = np.split(raw_pyro, np.where(raw_puv == max_temp)[0])[0]
    # time_heat = np.split(raw_time, np.where(raw_puv == max_temp)[0])[0]
    # temp_cool = np.split(raw_puv, np.where(raw_puv == max_temp)[0])[1]
    # time_cool = np.split(raw_time, np.where(raw_puv == max_temp)[0])[1]
    # voltage_heat = np.split(raw_voltage, np.where(raw_puv == max_temp)[0])[0]
    # rho_heat = np.split(rho, np.where(raw_puv == max_temp)[0])[0]
    # DTdt_heat = np.split(DTdt, np.where(raw_puv == max_temp)[0])[0]
    # DTdt_cool = np.split(DTdt, np.where(raw_puv == max_temp)[0])[1]

    # f_voltage_heat = int1d(temp_heat, voltage_heat, kind='quadratic')
    #                                       # fill_value='extrapolate')
    # f_rho_heat = int1d(temp_heat, rho_heat, kind='quadratic')
    #                                   # fill_value='extrapolate')
    f_DTdt_curr = int1d(temp_curr, DTdt_curr, kind='linear',
                        fill_value='extrapolate')
    f_DTdt_cool = int1d(temp_cool, DTdt_cool, kind='linear',
                        fill_value='extrapolate')

    # f_voltage_heat = int1d(temp_heat, voltage_heat, kind='linear',
    #                                       fill_value='extrapolate')
    # f_rho_heat = int1d(temp_heat, rho_heat, kind='linear',
    #                                   fill_value='extrapolate')
    degree = 21

    # f_temp_heat = int1d(time_heat, temp_heat, kind='linear',
    #                                    fill_value='extrapolate')
    # f_temp_cool = int1d(time_cool, temp_cool, kind='linear',
    #                                    fill_value='extrapolate')
    p_temp_cool = np.poly1d(np.polyfit(time_cool, temp_cool, deg=degree))

    # f_DTdt_heat = int1d(temp_heat, DTdt_heat, kind='linear',
    #                                    fill_value='extrapolate')
    # p_DTdt_heat = np.poly1d(np.polyfit(temp_heat, DTdt_heat, deg=degree))

    # f_DTdt_heat_curr = int1d(temp_curr, DTdt_heat_curr,
    #                                         kind='linear',
    #                                         fill_value='extrapolate')
    # p_DTdt_heat_curr = np.poly1d(np.polyfit(temp_curr, DTdt_heat_curr, deg=degree))

    # f_DTdt_cool = int1d(temp_cool, DTdt_cool, kind='linear',
    #                     fill_value='extrapolate')
    p_DTdt_cool = np.poly1d(np.polyfit(temp_cool, DTdt_cool, deg=degree))

    # print(temp_heat)

    # f_voltage_heat = IUS(temp_heat, voltage_heat, ext=0)
    # f_voltage_heat = int1d(temp_heat, voltage_heat,
    #                                       kind='linear',
    #                                       fill_value='extrapolate')

    f_voltage_curr = int1d(temp_curr, voltage_curr, kind='linear',
                           fill_value='extrapolate')
    p_voltage_curr = np.poly1d(np.polyfit(temp_curr, voltage_curr, deg=degree))

    # f_rho_heat = IUS(temp_heat, rho_heat, ext=0)
    # f_rho_heat = int1d(temp_heat, rho_heat, kind='linear',
    #                                     fill_value='extrapolate')
    f_rho_curr = int1d(temp_curr, rho_curr, kind='linear',
                       fill_value='extrapolate')
    p_rho_curr = np.poly1d(np.polyfit(temp_curr, rho_curr, deg=degree))


    time_cool_new = np.linspace(min(time_cool), max(time_cool), 1000)
    time_curr_new = np.linspace(min(time_curr), max(time_curr), 1000)
    temp_cool_new = np.linspace(300, max(temp_cool), 1000)

    # print(time_curr_new)

    # plt.figure('heat')
    # # plt.plot(time_heat, temp_heat, 'r.')
    # plt.plot(temp_cool_new, f_temp_heat(temp_cool_new))

    # time_cool_new = np.linspace(min(time_cool), 20, 1000)
    temp_cool_new = np.linspace(300, max(temp_cool), 1000)
    temp_curr_new = np.linspace(300, max(temp_curr), 1000)

    # plt.plot(time_cool, temp_cool, 'r.')
    # plt.plot(time_cool_new, p_temp_cool(time_cool_new))

    # input()
    # exit()

    # def exponenial_func(x, a, b, c):
    #     return(a * np.exp(-b * x) + c)


    # popt, pcov = curve_fit(exponenial_func, time_cool, temp_cool, p0=(1, 1e-6, 1))
    ep_temp_cool = np.poly1d(np.polyfit(time_cool, np.log(temp_cool), deg=3))  # , w=np.sqrt(temp_cool)))
    ep_temp_curr = np.poly1d(np.polyfit(time_curr, np.log(temp_curr), deg=3))  # , w=np.sqrt(temp_cool)))

    DTdt_cool_3 = np.gradient(ep_temp_cool(time_cool_new)) / \
                   np.gradient(time_cool_new)
    DTdt_curr_3 = np.gradient(ep_temp_curr(time_curr_new)) / \
                   np.gradient(time_curr_new)

    p_DTdt_cool_3 = np.poly1d(np.polyfit(temp_cool_new, DTdt_cool_3, deg=degree))
    f_DTdt_cool_3 = int1d(temp_cool_new, DTdt_cool_3, kind='linear',
                          fill_value='extrapolate')
    p_DTdt_curr_3 = np.poly1d(np.polyfit(temp_curr_new, DTdt_curr_3, deg=degree))
    f_DTdt_curr_3 = int1d(temp_curr_new, DTdt_curr_3, kind='linear',
                          fill_value='extrapolate')

    # print(DTdt_curr)


    p_DTdt_cool_2 = np.poly1d(np.polyfit(temp_cool, DTdt_cool, deg=degree))
    f_DTdt_cool_2 = int1d(temp_cool, DTdt_cool, kind='linear',
                          fill_value='extrapolate')
    p_DTdt_curr_2 = np.poly1d(np.polyfit(temp_curr, DTdt_curr, deg=degree))
    f_DTdt_curr_2 = int1d(temp_curr, DTdt_curr, kind='linear',
                          fill_value='extrapolate')

    # print(min(temp_cool))
    # print(min(temp_curr))

    # tempcp = np.linspace(max_temp - 50, 1 * max_temp, 1000)
    # tempcp = np.linspace( 1 * max_temp2, u_temp_cool(time_cool[idx]), num=100)
    tempcp = np.linspace(max_temp * 0.95, min_temp, num=10000)

    # p_DTdt_heat_curr_2 = np.poly1d(np.polyfit(tempcp, f_DTdt_heat_curr(tempcp), deg=degree))
    # p_DTdt_cool_2 = np.poly1d(np.polyfit(tempcp, f_DTdt_cool(tempcp), deg=degree))

    # plt.figure('curr')
    # plt.plot(temp_curr_new, p_DTdt_curr_3(temp_curr_new))


    dcp2 = p_voltage_curr(tempcp)**2 / \
        (p_rho_curr(tempcp) * length**2) * \
        1 / (f_DTdt_curr_2(tempcp) - f_DTdt_cool_2(tempcp))
    dcp3 = p_voltage_curr(tempcp)**2 / \
        (p_rho_curr(tempcp) * length**2) * \
        1 / (p_DTdt_curr_3(tempcp) - p_DTdt_cool_3(tempcp))
    dcp1 = f_voltage_curr(tempcp)**2 / \
        (f_rho_curr(tempcp) * length**2) * \
        1 / (f_DTdt_curr(tempcp) - f_DTdt_cool(tempcp))

    # print('check please:')
    # print(f_DTdt_curr_2(tempcp))
    # print(rho)
    plt.figure('dcp formula check')
    plt.plot(tempcp, f_voltage_curr(tempcp), label='f_voltage_curr')
    plt.plot(tempcp, 1 / f_rho_curr(tempcp), label='f_rho_curr')
    plt.plot(tempcp, 1 / f_DTdt_curr(tempcp), label='f_DTdt_curr')
    plt.plot(tempcp, 1 / f_DTdt_cool(tempcp), label='f_DTdt_cool')
    # plt.plot(tempcp, f_DTdt_cool_2(tempcp), label='int1d_2')
    # plt.plot(tempcp, f_DTdt_cool_2(tempcp), label='int1d_3')
    # # plt.plot(tempcp, p_voltage_curr(tempcp), label='poly')
    plt.legend()
    # plt.show()
    # input()
    # exit()
    # plt.plot(rho)
    # dcp1 = f_voltage_curr(tempcp)**2 / \
    #     (p_rho_curr(tempcp) * length**2) * \
    #     1 / (p_DTdt_heat_curr_2(tempcp) - p_DTdt_cool_3(tempcp))

    # ## recent working kind of
    # dcp1 = p_voltage_curr(tempcp)**2 / \
    #     (p_rho_curr(tempcp) * length**2) * \
    #     1 / (p_DTdt_curr_2(tempcp) - p_DTdt_cool_2(tempcp))
    # ## recent working kind of

    # dcp3 = f_voltage_heat(tempcp)**2 / \
    #     (f_rho_heat(tempcp) * length**2) * \
    #     1 / (u_DTdt_heat_2(tempcp) - u_DTdt_cool_2(tempcp))

    f_dcp1 = int1d(tempcp, dcp1, kind='linear',
                   fill_value='extrapolate')
    # f_dcp2 = interpolate.interp1d(tempcp, dcp2, kind='linear',
    #                               fill_value='extrapolate')

    # for i in range(len(tempcp)):
    #     print(f_DTdt_cool(tempcp[i]))
    #     dcp[i] = f_voltage_heat(tempcp[i])**2 / \
    #              (f_rho_heat(tempcp[i]) * length**2) * \
    #              1 / (f_DTdt_heat(tempcp[i]) - f_DTdt_cool(tempcp[i]))
        # print((f_rho_heat(tempcp[i]) * length**2))
        # print(f_DTdt_heat(tempcp[i]) - f_DTdt_cool(tempcp[i]))
    # exit()
    # dcp1_sgf = sgf(dcp1, 51, 5)

    # print(dcp1)
    # print(dcp2)
    # print("%f , %f" % (dcp1, dcp2))
    # print(dcp3)
    # plt.figure('dcp-check')
    # plt.plot(tempcp, dcp1, label='dcp1')
    # plt.plot(tempcp, dcp2, label='dcp2')
    # plt.plot(tempcp, dcp3, label='dcp3')
    # plt.legend()
    # plt.show()
    # input()
    # exit()
    # plt.show()
    # exit()
    # import matplotlib.ticker as mtick
    plt.figure('dcp')
    # plt.ticklabel_format(useOffset=False)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title(r'volumetric heat capacity $\delta C_p$')
    plt.xlabel(r'temperature / $K$')
    # plt.ylabel(r'volumetric heat capacity / $J/K \cdot m 3$')
    plt.ylabel('volumetric heat capacity')
    plt.plot(tempcp, dcp1, label=r'$\delta C_p$ dcp1')
    # plt.plot(tempcp, dcp2, label=r'$\delta C_p$ dcp2')
    # plt.plot(tempcp, dcp3, label=r'$\delta C_p$ d_3')
    # plt.plot(tempcp, f_dcp1(tempcp), label=r'func $\delta C_p$ (T)')
    # plt.plot(tempcp, dcp1_sgf, label=r'$\delta C_p - smooth$')
    # plt.plot(tempcp, dcp2)
    # plt.plot(tempcp, dcp3)
    lit_cp = 265        # specific heat capacity J / (kg * K)
    lit_delta = 8570    # density kg / m**3
    # print(np.float64(lit_cp * lit_delta))
    # plt.axhline(lit_cp * lit_delta, label='sim. param.', color='black')
    plt.fill_between(tempcp, 0.95 * dcp1, 1.05 * dcp1, color='gray', alpha=0.5)
    # plt.fill_between(tempcp, 0.95 * dcp2, 1.05 * dcp2, color='gray', alpha=0.5)
    # plt.fill_between(tempcp, 0.95 * dcp3, 1.05 * dcp3, color='gray', alpha=0.5)
    # plt.plot(raw_puv, raw_puv + lit_cp * lit_delta, '.')
    plt.legend()
    plt.grid(True)

    # plt.show()
    # # plt.draw()
    # input()
    # exit()
    return(dcp1, f_dcp1)


def cal_cp_from_current3(rho, temp_den, den, length, epsht, temp_epsht,
                         raw_pyro=[], raw_time=[], raw_voltage=[]):
    import numpy as np
    from scipy import interpolate
    from scipy.interpolate import interp1d as int1d
    # from scipy.interpolate import UnivariateSpline as US
    # from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    import matplotlib.pyplot as plt
    # from scipy.signal import savgol_filter as sgf
    # from scipy.optimize import curve_fit

    import warnings
    warnings.simplefilter('ignore', np.RankWarning)
    # current =
    # DTdt1 = np.gradient(raw_pyro, edge_order=1) / \
    #     np.gradient(raw_time, edge_order=1)
    # T_a = 300  # K
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

    f_epsht = int1d(temp_epsht, epsht, kind='linear',
                    fill_value='extrapolate')
    f_den = int1d(temp_den, den, kind='linear',
                  fill_value='extrapolate')

    max_temp = np.max(raw_pyro)
    # min_temp = np.min(raw_pyro)
    max_temp_point = (np.argwhere(raw_pyro == np.max(raw_pyro)))[0]
    # print(max_temp)

    # print(raw_voltage)
    # exit()
    voltage_off = (np.argwhere(np.abs(raw_voltage) > 0)[-1]) + 1
    voltage_on = (np.argwhere(np.abs(raw_voltage) > 0)[1])

    # print(voltage_off)
    # print(voltage_on)
    # print(np.shape(raw_voltage))
    # exit()

    # plt.figure("raw_voltage")
    # plt.plot(raw_voltage)
    # print(voltage_off)
    # input()
    # exit()

    if max_temp_point > voltage_off:
        voltage_off = max_temp_point

    # print(voltage_on)
    # print(voltage_off)
    # print(raw_voltage)
    # print(raw_voltage[voltage_on])
    # print(raw_voltage[voltage_off])
    # print(np.abs(raw_voltage[voltage_off]))
    # exit()
    # voltage_off = np.where(np.roll(raw_voltage[:, 1], 1) != raw_voltage[:, 1])[0]

    DTdt = np.gradient(raw_pyro, edge_order=0) / \
        np.gradient(raw_time, edge_order=0)

    # plt.figure("temp")
    # plt.plot(DTdt,'.')
    # input()
    # exit()

    temp_heat = np.split(raw_pyro, voltage_off)[0]
    temp_curr = np.split(temp_heat, voltage_on)[1]
    temp_cool = np.split(raw_pyro, voltage_off)[1]


    min_temp = np.min(temp_cool)

    time_heat = np.split(raw_time, voltage_off)[0]
    time_curr = np.split(time_heat, voltage_on)[1]
    time_cool = np.split(raw_time, voltage_off)[1]

    voltage_heat = np.split(np.abs(raw_voltage), voltage_off)[0]
    voltage_curr = np.split(voltage_heat, voltage_on)[1]

    rho_heat = np.split(rho, voltage_off)[0]
    rho_curr = np.split(rho_heat, voltage_on)[1]

    DTdt_heat = np.split(DTdt, voltage_off)[0]
    DTdt_curr = np.split(DTdt_heat, voltage_on)[1]
    DTdt_cool = np.split(DTdt, voltage_off)[1]

    # print(np.gradient(raw_pyro, edge_order=0))
    # print(np.gradient(raw_time, edge_order=0))
    # print(raw_time)
    # plt.figure('check')
    # plt.plot(DTdt_heat)
    # plt.plot(DTdt_curr, '.')
    # plt.plot(DTdt_cool)
    # input()
    # exit()

    # print(DTdt)
    # exit()
    # print(temp_curr)

    # print(rho_curr)
    # print(np.split(raw_voltage, voltage_off)[0])
    # exit()

    # temp_heat = np.split(raw_pyro, np.where(raw_puv == max_temp)[0])[0]
    # time_heat = np.split(raw_time, np.where(raw_puv == max_temp)[0])[0]
    # temp_cool = np.split(raw_puv, np.where(raw_puv == max_temp)[0])[1]
    # time_cool = np.split(raw_time, np.where(raw_puv == max_temp)[0])[1]
    # voltage_heat = np.split(raw_voltage, np.where(raw_puv == max_temp)[0])[0]
    # rho_heat = np.split(rho, np.where(raw_puv == max_temp)[0])[0]
    # DTdt_heat = np.split(DTdt, np.where(raw_puv == max_temp)[0])[0]
    # DTdt_cool = np.split(DTdt, np.where(raw_puv == max_temp)[0])[1]

    # f_voltage_heat = int1d(temp_heat, voltage_heat, kind='quadratic')
    #                                       # fill_value='extrapolate')
    # f_rho_heat = int1d(temp_heat, rho_heat, kind='quadratic')
    #                                   # fill_value='extrapolate')
    f_DTdt_curr = int1d(temp_curr, DTdt_curr, kind='linear',
                        fill_value='extrapolate')
    f_DTdt_cool = int1d(temp_cool, DTdt_cool, kind='linear',
                        fill_value='extrapolate')

    # f_voltage_heat = int1d(temp_heat, voltage_heat, kind='linear',
    #                                       fill_value='extrapolate')
    # f_rho_heat = int1d(temp_heat, rho_heat, kind='linear',
    #                                   fill_value='extrapolate')
    degree = 21

    # f_temp_heat = int1d(time_heat, temp_heat, kind='linear',
    #                                    fill_value='extrapolate')
    # f_temp_cool = int1d(time_cool, temp_cool, kind='linear',
    #                                    fill_value='extrapolate')
    # p_temp_cool = np.poly1d(np.polyfit(time_cool, temp_cool, deg=degree))

    # f_DTdt_heat = int1d(temp_heat, DTdt_heat, kind='linear',
    #                                    fill_value='extrapolate')
    # p_DTdt_heat = np.poly1d(np.polyfit(temp_heat, DTdt_heat, deg=degree))

    # f_DTdt_heat_curr = int1d(temp_curr, DTdt_heat_curr,
    #                                         kind='linear',
    #                                         fill_value='extrapolate')
    # p_DTdt_heat_curr = np.poly1d(np.polyfit(temp_curr, DTdt_heat_curr, deg=degree))

    # f_DTdt_cool = int1d(temp_cool, DTdt_cool, kind='linear',
    #                     fill_value='extrapolate')
    p_DTdt_cool = np.poly1d(np.polyfit(temp_cool, DTdt_cool, deg=degree))

    # print(temp_heat)

    # f_voltage_heat = IUS(temp_heat, voltage_heat, ext=0)
    # f_voltage_heat = int1d(temp_heat, voltage_heat,
    #                                       kind='linear',
    #                                       fill_value='extrapolate')

    f_voltage_curr = int1d(temp_curr, voltage_curr, kind='linear',
                           fill_value='extrapolate')
    p_voltage_curr = np.poly1d(np.polyfit(temp_curr, voltage_curr, deg=degree))

    # f_rho_heat = IUS(temp_heat, rho_heat, ext=0)
    # f_rho_heat = int1d(temp_heat, rho_heat, kind='linear',
    #                                     fill_value='extrapolate')
    f_rho_curr = int1d(temp_curr, rho_curr, kind='linear',
                       fill_value='extrapolate')
    p_rho_curr = np.poly1d(np.polyfit(temp_curr, rho_curr, deg=degree))


    time_cool_new = np.linspace(min(time_cool), max(time_cool), 1000)
    time_curr_new = np.linspace(min(time_curr), max(time_curr), 1000)
    temp_cool_new = np.linspace(300, max(temp_cool), 1000)

    # print(time_curr_new)

    # plt.figure('heat')
    # # plt.plot(time_heat, temp_heat, 'r.')
    # plt.plot(temp_cool_new, f_temp_heat(temp_cool_new))

    # time_cool_new = np.linspace(min(time_cool), 20, 1000)
    temp_cool_new = np.linspace(300, max(temp_cool), 1000)
    temp_curr_new = np.linspace(300, max(temp_curr), 1000)

    # plt.plot(time_cool, temp_cool, 'r.')
    # plt.plot(time_cool_new, p_temp_cool(time_cool_new))

    # input()
    # exit()

    # def exponenial_func(x, a, b, c):
    #     return(a * np.exp(-b * x) + c)


    # popt, pcov = curve_fit(exponenial_func, time_cool, temp_cool, p0=(1, 1e-6, 1))
    # ep_temp_cool = np.poly1d(np.polyfit(time_cool, np.log(temp_cool), deg=3))  # , w=np.sqrt(temp_cool)))
    # ep_temp_curr = np.poly1d(np.polyfit(time_curr, np.log(temp_curr), deg=3))  # , w=np.sqrt(temp_cool)))

    # DTdt_cool_3 = np.gradient(ep_temp_cool(time_cool_new)) / \
                   # np.gradient(time_cool_new)
    # DTdt_curr_3 = np.gradient(ep_temp_curr(time_curr_new)) / \
    #                np.gradient(time_curr_new)

    # p_DTdt_cool_3 = np.poly1d(np.polyfit(temp_cool_new, DTdt_cool_3, deg=degree))
    # f_DTdt_cool_3 = int1d(temp_cool_new, DTdt_cool_3, kind='linear',
    #                       fill_value='extrapolate')
    # p_DTdt_curr_3 = np.poly1d(np.polyfit(temp_curr_new, DTdt_curr_3, deg=degree))
    # f_DTdt_curr_3 = int1d(temp_curr_new, DTdt_curr_3, kind='linear',
    #                       fill_value='extrapolate')

    # print(DTdt_curr)


    p_DTdt_cool_2 = np.poly1d(np.polyfit(temp_cool, DTdt_cool, deg=degree))
    f_DTdt_cool_2 = int1d(temp_cool, DTdt_cool, kind='linear',
                          fill_value='extrapolate')
    p_DTdt_curr_2 = np.poly1d(np.polyfit(temp_curr, DTdt_curr, deg=degree))
    f_DTdt_curr_2 = int1d(temp_curr, DTdt_curr, kind='linear',
                          fill_value='extrapolate')

    # print(min(temp_cool))
    # print(min(temp_curr))

    # tempcp = np.linspace(max_temp - 50, 1 * max_temp, 1000)
    # tempcp = np.linspace( 1 * max_temp2, u_temp_cool(time_cool[idx]), num=100)
    tempcp = np.linspace(max_temp * 1.0, min_temp, num=10000)

    # p_DTdt_heat_curr_2 = np.poly1d(np.polyfit(tempcp, f_DTdt_heat_curr(tempcp), deg=degree))
    # p_DTdt_cool_2 = np.poly1d(np.polyfit(tempcp, f_DTdt_cool(tempcp), deg=degree))

    # plt.figure('f_voltage_curr')
    # plt.plot(tempcp, f_voltage_curr(tempcp), label='f_voltage_curr')
    # plt.plot(tempcp, p_voltage_curr(tempcp), label='p_voltage_curr')
    # plt.plot(temp_curr, voltage_curr, 'r.', label='voltage_curr')
    # plt.plot(raw_time, raw_voltage, 'g.', label='raw_pyro & voltage')
    # plt.figure("DTdtcool")
    # plt.plot(temp_cool, DTdt_cool)
    # plt.figure("temp cool over time")
    # plt.plot(time_cool, temp_cool)

    # plt.legend()
    # input()
    # exit()

    # dcp2 = p_voltage_curr(tempcp)**2 / \
    #     (p_rho_curr(tempcp) * length**2) * \
    #     1 / (f_DTdt_curr_2(tempcp) - f_DTdt_cool_2(tempcp))
    # dcp3 = p_voltage_curr(tempcp)**2 / \
    #     (p_rho_curr(tempcp) * length**2) * \
    #     1 / (p_DTdt_curr_3(tempcp) - p_DTdt_cool_3(tempcp))
    # dcp1 = f_voltage_curr(tempcp)**2 / \
    #     (f_rho_curr(tempcp) * length**2) * \
    #     1 / (f_DTdt_curr(tempcp) - f_DTdt_cool(tempcp))
    dcp1 = 1 / (np.abs(f_DTdt_curr(tempcp))) * \
        ((f_voltage_curr(tempcp)**2 / (f_rho_curr(tempcp) * length**2)) - \
        ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s))

    cp1 = 1 / (np.abs(f_DTdt_curr(tempcp))) * \
        ((f_voltage_curr(tempcp)**2 / (f_rho_curr(tempcp) * length**2)) - \
        ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s)) / \
        f_den(tempcp)

    # print(f_DTdt_cool(tempcp))
    # print(f_DTdt_cool_2(tempcp))
    # print(p_DTdt_cool(tempcp))
    # print(p_DTdt_cool_2(tempcp))
    # exit()

    # dcp1 = 1 / (np.abs(f_DTdt_cool_2(tempcp))) * \
    #     ( ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s))

    # cp1 = 1 / (np.abs(f_DTdt_cool_2(tempcp))) * \
    #     ( ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s)) / \
    #     f_den(tempcp)

    # print(( - ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s)) / \
    #     f_den(tempcp))
    # exit()
    # dcp2 = 1 / (np.abs(f_DTdt_curr_2(tempcp))) * \
    #     ((p_voltage_curr(tempcp)**2 / (p_rho_curr(tempcp) * length**2)) - \
    #     ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s))

    # dcp3 = 1 / (np.abs(p_DTdt_curr_3(tempcp))) * \
    #     ((p_voltage_curr(tempcp)**2 / (p_rho_curr(tempcp) * length**2)) - \
    #     ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s))

    # plt.figure('check please:')
    # plt.plot(tempcp, f_voltage_curr(tempcp))
    # input()
    # exit()
    # # print(rho)
    # plt.figure('1 formula check')
    # plt.plot(tempcp, (f_voltage_curr(tempcp)**2 / (f_rho_curr(tempcp) * length**2)))
    # plt.figure('2 formula check')
    # plt.plot(tempcp, ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s))
    # plt.figure('gesamt formula check')
    # plt.plot(tempcp, (f_voltage_curr(tempcp)**2 / (f_rho_curr(tempcp) * length**2)) - \
    #          ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s))
    # plt.figure('f_epsht')
    # plt.plot(tempcp, f_epsht(tempcp))
    # print(sigma)
    # print(p)
    # print(1/s)
    # plt.plot(tempcp, ((f_epsht(tempcp))))
    # plt.plot(tempcp, f_voltage_curr(tempcp), label='f_voltage_curr')
    # plt.plot(tempcp, 1 / f_rho_curr(tempcp), label='f_rho_curr')
    # plt.plot(tempcp, 1 / f_DTdt_curr(tempcp), label='f_DTdt_curr')
    # plt.plot(tempcp, 1 / f_DTdt_cool(tempcp), label='f_DTdt_cool')
    # plt.plot(tempcp, f_DTdt_cool_2(tempcp), label='int1d_2')
    # plt.plot(tempcp, f_DTdt_cool_2(tempcp), label='int1d_3')
    # # plt.plot(tempcp, p_voltage_curr(tempcp), label='poly')
    # plt.legend()
    # plt.show()
    # input()
    # exit()
    # plt.plot(rho)
    # dcp1 = f_voltage_curr(tempcp)**2 / \
    #     (p_rho_curr(tempcp) * length**2) * \
    #     1 / (p_DTdt_heat_curr_2(tempcp) - p_DTdt_cool_3(tempcp))

    # ## recent working kind of
    # dcp1 = p_voltage_curr(tempcp)**2 / \
    #     (p_rho_curr(tempcp) * length**2) * \
    #     1 / (p_DTdt_curr_2(tempcp) - p_DTdt_cool_2(tempcp))
    # ## recent working kind of

    # dcp3 = f_voltage_heat(tempcp)**2 / \
    #     (f_rho_heat(tempcp) * length**2) * \
    #     1 / (u_DTdt_heat_2(tempcp) - u_DTdt_cool_2(tempcp))

    f_dcp1 = int1d(tempcp, dcp1, kind='linear',
                   fill_value='extrapolate')
    # f_dcp2 = interpolate.interp1d(tempcp, dcp2, kind='linear',
    #                               fill_value='extrapolate')

    # for i in range(len(tempcp)):
    #     print(f_DTdt_cool(tempcp[i]))
    #     dcp[i] = f_voltage_heat(tempcp[i])**2 / \
    #              (f_rho_heat(tempcp[i]) * length**2) * \
    #              1 / (f_DTdt_heat(tempcp[i]) - f_DTdt_cool(tempcp[i]))
        # print((f_rho_heat(tempcp[i]) * length**2))
        # print(f_DTdt_heat(tempcp[i]) - f_DTdt_cool(tempcp[i]))
    # exit()
    # dcp1_sgf = sgf(dcp1, 51, 5)

    # print(dcp1)
    # print(dcp2)
    # print("%f , %f" % (dcp1, dcp2))
    # print(dcp3)
    # plt.figure('dcp-check')
    # plt.plot(tempcp, dcp1, label='dcp1')
    # plt.plot(tempcp, dcp2, label='dcp2')
    # plt.plot(tempcp, dcp3, label='dcp3')
    # plt.legend()
    # plt.show()
    # input()
    # exit()
    # plt.show()
    # exit()
    # import matplotlib.ticker as mtick
    plt.figure('dcp')
    # plt.ticklabel_format(useOffset=False)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plt.title(r'volumetric heat capacity $\rho C_p$')
    plt.title(r'volumetric heat capacity $C_p$')
    plt.xlabel(r'temperature / $K$')
    # plt.ylabel(r'volumetric heat capacity / $J/K \cdot m 3$')
    plt.ylabel('volumetric heat capacity')
    # plt.plot(tempcp, dcp1, label=r'$\rho C_p$')
    plt.plot(tempcp, cp1, label=r'$C_p$')
    # plt.plot(tempcp, dcp2, label=r'$\delta C_p$ dcp2')
    # plt.plot(tempcp, dcp3, label=r'$\delta C_p$ d_3')
    # plt.plot(tempcp, f_dcp1(tempcp), label=r'func $\delta C_p$ (T)')
    # plt.plot(tempcp, dcp1_sgf, label=r'$\delta C_p - smooth$')
    # plt.plot(tempcp, dcp2)
    # plt.plot(tempcp, dcp3)
    lit_cp = 265        # specific heat capacity J / (kg * K)
    lit_delta = 8570    # density kg / m**3
    # print(np.float64(lit_cp * lit_delta))
    # plt.axhline(lit_cp * lit_delta, label='sim. param.', color='black')
    # plt.fill_between(tempcp, 0.95 * dcp1, 1.05 * dcp1, color='gray', alpha=0.5)
    plt.fill_between(tempcp, 0.95 * cp1, 1.05 * cp1, color='gray', alpha=0.5)
    # plt.fill_between(tempcp, 0.95 * dcp2, 1.05 * dcp2, color='gray', alpha=0.5)
    # plt.fill_between(tempcp, 0.95 * dcp3, 1.05 * dcp3, color='gray', alpha=0.5)
    # plt.plot(raw_puv, raw_puv + lit_cp * lit_delta, '.')
    plt.legend()
    plt.grid(True)

    # plt.show()
    # # plt.draw()
    # input()
    # exit()
    return(dcp1, f_dcp1)


def cal_cp_from_current4(rho, temp_den, den, length, epsht, temp_epsht,
                         raw_pyro=[], raw_time=[], raw_voltage=[]):
    import numpy as np
    from scipy import interpolate
    from scipy.interpolate import interp1d as int1d
    # from scipy.interpolate import UnivariateSpline as US
    # from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    import matplotlib.pyplot as plt
    # from scipy.signal import savgol_filter as sgf
    # from scipy.optimize import curve_fit

    import warnings
    warnings.simplefilter('ignore', np.RankWarning)
    # current =
    # DTdt1 = np.gradient(raw_pyro, edge_order=1) / \
    #     np.gradient(raw_time, edge_order=1)
    # T_a = 300  # K
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

    f_epsht = int1d(temp_epsht, epsht, kind='linear',
                    fill_value='extrapolate')
    f_den = int1d(temp_den, den, kind='linear',
                  fill_value='extrapolate')

    max_temp = np.max(raw_pyro)
    min_temp = np.min(raw_pyro)
    # max_temp_point = (np.argwhere(raw_pyro == np.max(raw_pyro)))[0]
    # print(max_temp)
    temp = raw_pyro
    time = raw_time
    voltage = raw_voltage

    # print(raw_voltage)
    # # exit()
    # voltage_off = (np.argwhere(np.abs(raw_voltage) > 0)[-1])
    # voltage_on = (np.argwhere(np.abs(raw_voltage) > 0)[1]) - 1

    # if max_temp_point > voltage_off:
    #     voltage_off = max_temp_point

    # print(voltage_on)
    # print(voltage_off)
    # print(raw_voltage)
    # print(raw_voltage[voltage_on])
    # print(raw_voltage[voltage_off])
    # print(np.abs(raw_voltage[voltage_off]))
    # exit()
    # voltage_off = np.where(np.roll(raw_voltage[:, 1], 1) != raw_voltage[:, 1])[0]

    DTdt = np.gradient(raw_pyro, edge_order=0) / \
        np.gradient(raw_time, edge_order=0)

    # plt.figure("temp")
    # plt.plot(DTdt,'.')
    # input()
    # exit()

    # temp_heat = np.split(raw_pyro, voltage_off)[0]
    # temp_curr = np.split(temp_heat, voltage_on)[1]
    # temp_cool = np.split(raw_pyro, voltage_off)[1]

    # min_temp = np.min(temp_cool)

    # time_heat = np.split(raw_time, voltage_off)[0]
    # time_curr = np.split(time_heat, voltage_on)[1]
    # time_cool = np.split(raw_time, voltage_off)[1]

    # voltage_heat = np.split(np.abs(raw_voltage), voltage_off)[0]
    # voltage_curr = np.split(voltage_heat, voltage_on)[1]

    # rho_heat = np.split(rho, voltage_off)[0]
    # rho_curr = np.split(rho_heat, voltage_on)[1]

    # DTdt_heat = np.split(DTdt, voltage_off)[0]
    # DTdt_curr = np.split(DTdt_heat, voltage_on)[1]
    # DTdt_cool = np.split(DTdt, voltage_off)[1]

    # print(temp_curr)
    # print(rho_curr)
    # print(np.split(raw_voltage, voltage_off)[0])
    # exit()

    # temp_heat = np.split(raw_pyro, np.where(raw_puv == max_temp)[0])[0]
    # time_heat = np.split(raw_time, np.where(raw_puv == max_temp)[0])[0]
    # temp_cool = np.split(raw_puv, np.where(raw_puv == max_temp)[0])[1]
    # time_cool = np.split(raw_time, np.where(raw_puv == max_temp)[0])[1]
    # voltage_heat = np.split(raw_voltage, np.where(raw_puv == max_temp)[0])[0]
    # rho_heat = np.split(rho, np.where(raw_puv == max_temp)[0])[0]
    # DTdt_heat = np.split(DTdt, np.where(raw_puv == max_temp)[0])[0]
    # DTdt_cool = np.split(DTdt, np.where(raw_puv == max_temp)[0])[1]

    # f_voltage_heat = int1d(temp_heat, voltage_heat, kind='quadratic')
    #                                       # fill_value='extrapolate')
    # f_rho_heat = int1d(temp_heat, rho_heat, kind='quadratic')
    #                                   # fill_value='extrapolate')
    f_DTdt_curr = int1d(temp_curr, DTdt_curr, kind='linear',
                        fill_value='extrapolate')
    f_DTdt_cool = int1d(temp_cool, DTdt_cool, kind='linear',
                        fill_value='extrapolate')
    # f_DTdt = int1d(temp, DTdt, kind='linear',
    #                fill_value='extrapolate')

    # f_voltage_heat = int1d(temp_heat, voltage_heat, kind='linear',
    #                                       fill_value='extrapolate')
    # f_rho_heat = int1d(temp_heat, rho_heat, kind='linear',
    #                                   fill_value='extrapolate')
    degree = 21

    f_temp_heat = int1d(time_heat, temp_heat, kind='linear',
                                       fill_value='extrapolate')
    f_temp_cool = int1d(time_cool, temp_cool, kind='linear',
                                       fill_value='extrapolate')
    # p_temp_cool = np.poly1d(np.polyfit(time_cool, temp_cool, deg=degree))
    # p_temp = np.poly1d(np.polyfit(time, temp, deg=degree))

    # f_DTdt_heat = int1d(temp_heat, DTdt_heat, kind='linear',
    #                                    fill_value='extrapolate')
    # p_DTdt_heat = np.poly1d(np.polyfit(temp_heat, DTdt_heat, deg=degree))

    # f_DTdt_heat_curr = int1d(temp_curr, DTdt_heat_curr,
    #                                         kind='linear',
    #                                         fill_value='extrapolate')
    # p_DTdt_heat_curr = np.poly1d(np.polyfit(temp_curr, DTdt_heat_curr, deg=degree))

    # f_DTdt_cool = int1d(temp_cool, DTdt_cool, kind='linear',
    #                     fill_value='extrapolate')
    # p_DTdt_cool = np.poly1d(np.polyfit(temp_cool, DTdt_cool, deg=degree))
    # p_DTdtl = np.poly1d(np.polyfit(temp, DTdt, deg=degree))

    # print(temp_heat)

    # f_voltage_heat = IUS(temp_heat, voltage_heat, ext=0)
    # f_voltage_heat = int1d(temp_heat, voltage_heat,
    #                                       kind='linear',
    #                                       fill_value='extrapolate')

    # f_voltage_curr = int1d(temp_curr, voltage_curr, kind='linear',
    #                        fill_value='extrapolate')
    # p_voltage_curr = np.poly1d(np.polyfit(temp_curr, voltage_curr, deg=degree))
    f_voltage_curr = int1d(temp, voltage, kind='linear',
                           fill_value='extrapolate')
    p_voltage_curr = np.poly1d(np.polyfit(temp, voltage, deg=degree))

    # f_rho_heat = IUS(temp_heat, rho_heat, ext=0)
    # f_rho_heat = int1d(temp_heat, rho_heat, kind='linear',
    #                                     fill_value='extrapolate')
    # f_rho_curr = int1d(temp_curr, rho_curr, kind='linear',
    #                    fill_value='extrapolate')
    # p_rho_curr = np.poly1d(np.polyfit(temp_curr, rho_curr, deg=degree))
    f_rho_curr = int1d(temp, rho, kind='linear',
                       fill_value='extrapolate')
    p_rho_curr = np.poly1d(np.polyfit(temp, rho, deg=degree))


    # time_cool_new = np.linspace(min(time_cool), max(time_cool), 1000)
    # time_curr_new = np.linspace(min(time_curr), max(time_curr), 1000)
    # temp_cool_new = np.linspace(300, max(temp_cool), 1000)

    # print(time_curr_new)

    # plt.figure('heat')
    # # plt.plot(time_heat, temp_heat, 'r.')
    # plt.plot(temp_cool_new, f_temp_heat(temp_cool_new))

    # time_cool_new = np.linspace(min(time_cool), 20, 1000)
    temp_cool_new = np.linspace(300, max(temp_cool), 1000)
    temp_curr_new = np.linspace(300, max(temp_curr), 1000)

    # plt.plot(time_cool, temp_cool, 'r.')
    # plt.plot(time_cool_new, p_temp_cool(time_cool_new))

    # input()
    # exit()

    # def exponenial_func(x, a, b, c):
    #     return(a * np.exp(-b * x) + c)


    # popt, pcov = curve_fit(exponenial_func, time_cool, temp_cool, p0=(1, 1e-6, 1))
    # ep_temp_cool = np.poly1d(np.polyfit(time_cool, np.log(temp_cool), deg=3))  # , w=np.sqrt(temp_cool)))
    # ep_temp_curr = np.poly1d(np.polyfit(time_curr, np.log(temp_curr), deg=3))  # , w=np.sqrt(temp_cool)))

    # DTdt_cool_3 = np.gradient(ep_temp_cool(time_cool_new)) / \
    #                np.gradient(time_cool_new)
    # DTdt_curr_3 = np.gradient(ep_temp_curr(time_curr_new)) / \
    #                np.gradient(time_curr_new)

    # p_DTdt_cool_3 = np.poly1d(np.polyfit(temp_cool_new, DTdt_cool_3, deg=degree))
    # f_DTdt_cool_3 = int1d(temp_cool_new, DTdt_cool_3, kind='linear',
    #                       fill_value='extrapolate')
    # p_DTdt_curr_3 = np.poly1d(np.polyfit(temp_curr_new, DTdt_curr_3, deg=degree))
    # f_DTdt_curr_3 = int1d(temp_curr_new, DTdt_curr_3, kind='linear',
    #                       fill_value='extrapolate')

    # print(DTdt_curr)


    # p_DTdt_cool_2 = np.poly1d(np.polyfit(temp_cool, DTdt_cool, deg=degree))
    # f_DTdt_cool_2 = int1d(temp_cool, DTdt_cool, kind='linear',
    #                       fill_value='extrapolate')
    # p_DTdt_curr_2 = np.poly1d(np.polyfit(temp_curr, DTdt_curr, deg=degree))
    # f_DTdt_curr_2 = int1d(temp_curr, DTdt_curr, kind='linear',
    #                       fill_value='extrapolate')

    # print(min(temp_cool))
    # print(min(temp_curr))

    # tempcp = np.linspace(max_temp - 50, 1 * max_temp, 1000)
    # tempcp = np.linspace( 1 * max_temp2, u_temp_cool(time_cool[idx]), num=100)
    tempcp = np.linspace(max_temp * 0.95, min_temp, num=10000)

    # p_DTdt_heat_curr_2 = np.poly1d(np.polyfit(tempcp, f_DTdt_heat_curr(tempcp), deg=degree))
    # p_DTdt_cool_2 = np.poly1d(np.polyfit(tempcp, f_DTdt_cool(tempcp), deg=degree))

    # plt.figure('f_voltage_curr')
    # plt.plot(tempcp, f_voltage_curr(tempcp), label='f_voltage_curr')
    # plt.plot(tempcp, p_voltage_curr(tempcp), label='p_voltage_curr')
    # plt.plot(temp_curr, voltage_curr, 'r.', label='voltage_curr')
    # plt.plot(raw_time, raw_voltage, 'g.', label='raw_pyro & voltage')
    # plt.legend()
    # input()
    # exit()

    # dcp2 = p_voltage_curr(tempcp)**2 / \
    #     (p_rho_curr(tempcp) * length**2) * \
    #     1 / (f_DTdt_curr_2(tempcp) - f_DTdt_cool_2(tempcp))
    # dcp3 = p_voltage_curr(tempcp)**2 / \
    #     (p_rho_curr(tempcp) * length**2) * \
    #     1 / (p_DTdt_curr_3(tempcp) - p_DTdt_cool_3(tempcp))
    # dcp1 = f_voltage_curr(tempcp)**2 / \
    #     (f_rho_curr(tempcp) * length**2) * \
    #     1 / (f_DTdt_curr(tempcp) - f_DTdt_cool(tempcp))
    dcp1_curr = 1 / (np.abs(f_DTdt(tempcp))) * \
        ((f_voltage_curr(tempcp)**2 / (f_rho_curr(tempcp) * length**2)) - \
        ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s))

    dcp1_cool = f_dtDT(tempcp) * (- (f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4) / s))

    cp1 = 1 / (np.abs(f_DTdt(tempcp))) * \
        ((f_voltage_curr(tempcp)**2 / (f_rho_curr(tempcp) * length**2)) - \
        ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s)) / \
        f_den(tempcp)

    # dcp2 = 1 / (np.abs(f_DTdt_curr_2(tempcp))) * \
    #     ((p_voltage_curr(tempcp)**2 / (p_rho_curr(tempcp) * length**2)) - \
    #     ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s))

    # dcp3 = 1 / (np.abs(p_DTdt_curr_3(tempcp))) * \
    #     ((p_voltage_curr(tempcp)**2 / (p_rho_curr(tempcp) * length**2)) - \
    #     ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s))

    # plt.figure('check please:')
    # plt.plot(tempcp, f_DTdt_curr_2(tempcp))
    # # print(rho)
    # plt.figure('1 formula check')
    # plt.plot(tempcp, (f_voltage_curr(tempcp)**2 / (f_rho_curr(tempcp) * length**2)))
    # plt.figure('2 formula check')
    # plt.plot(tempcp, ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s))
    # plt.figure('gesamt formula check')
    # plt.plot(tempcp, (f_voltage_curr(tempcp)**2 / (f_rho_curr(tempcp) * length**2)) - \
    #          ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s))
    # plt.figure('f_epsht')
    # plt.plot(tempcp, f_epsht(tempcp))
    # print(sigma)
    # print(p)
    # print(1/s)
    # plt.plot(tempcp, ((f_epsht(tempcp))))
    # plt.plot(tempcp, f_voltage_curr(tempcp), label='f_voltage_curr')
    # plt.plot(tempcp, 1 / f_rho_curr(tempcp), label='f_rho_curr')
    # plt.plot(tempcp, 1 / f_DTdt_curr(tempcp), label='f_DTdt_curr')
    # plt.plot(tempcp, 1 / f_DTdt_cool(tempcp), label='f_DTdt_cool')
    # plt.plot(tempcp, f_DTdt_cool_2(tempcp), label='int1d_2')
    # plt.plot(tempcp, f_DTdt_cool_2(tempcp), label='int1d_3')
    # # plt.plot(tempcp, p_voltage_curr(tempcp), label='poly')
    # plt.legend()
    # plt.show()
    # input()
    # exit()
    # plt.plot(rho)
    # dcp1 = f_voltage_curr(tempcp)**2 / \
    #     (p_rho_curr(tempcp) * length**2) * \
    #     1 / (p_DTdt_heat_curr_2(tempcp) - p_DTdt_cool_3(tempcp))

    # ## recent working kind of
    # dcp1 = p_voltage_curr(tempcp)**2 / \
    #     (p_rho_curr(tempcp) * length**2) * \
    #     1 / (p_DTdt_curr_2(tempcp) - p_DTdt_cool_2(tempcp))
    # ## recent working kind of

    # dcp3 = f_voltage_heat(tempcp)**2 / \
    #     (f_rho_heat(tempcp) * length**2) * \
    #     1 / (u_DTdt_heat_2(tempcp) - u_DTdt_cool_2(tempcp))

    f_dcp1 = int1d(tempcp, dcp1, kind='linear',
                   fill_value='extrapolate')
    # f_dcp2 = interpolate.interp1d(tempcp, dcp2, kind='linear',
    #                               fill_value='extrapolate')

    # for i in range(len(tempcp)):
    #     print(f_DTdt_cool(tempcp[i]))
    #     dcp[i] = f_voltage_heat(tempcp[i])**2 / \
    #              (f_rho_heat(tempcp[i]) * length**2) * \
    #              1 / (f_DTdt_heat(tempcp[i]) - f_DTdt_cool(tempcp[i]))
        # print((f_rho_heat(tempcp[i]) * length**2))
        # print(f_DTdt_heat(tempcp[i]) - f_DTdt_cool(tempcp[i]))
    # exit()
    # dcp1_sgf = sgf(dcp1, 51, 5)

    # print(dcp1)
    # print(dcp2)
    # print("%f , %f" % (dcp1, dcp2))
    # print(dcp3)
    # plt.figure('dcp-check')
    # plt.plot(tempcp, dcp1, label='dcp1')
    # plt.plot(tempcp, dcp2, label='dcp2')
    # plt.plot(tempcp, dcp3, label='dcp3')
    # plt.legend()
    # plt.show()
    # input()
    # exit()
    # plt.show()
    # exit()
    # import matplotlib.ticker as mtick
    plt.figure('dcp')
    # plt.ticklabel_format(useOffset=False)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plt.title(r'volumetric heat capacity $\rho C_p$')
    plt.title(r'volumetric heat capacity $C_p$')
    plt.xlabel(r'temperature / $K$')
    # plt.ylabel(r'volumetric heat capacity / $J/K \cdot m 3$')
    plt.ylabel('volumetric heat capacity')
    # plt.plot(tempcp, dcp1, label=r'$\rho C_p$')
    plt.plot(tempcp, cp1, label=r'$C_p$')
    # plt.plot(tempcp, dcp2, label=r'$\delta C_p$ dcp2')
    # plt.plot(tempcp, dcp3, label=r'$\delta C_p$ d_3')
    # plt.plot(tempcp, f_dcp1(tempcp), label=r'func $\delta C_p$ (T)')
    # plt.plot(tempcp, dcp1_sgf, label=r'$\delta C_p - smooth$')
    # plt.plot(tempcp, dcp2)
    # plt.plot(tempcp, dcp3)
    lit_cp = 265        # specific heat capacity J / (kg * K)
    lit_delta = 8570    # density kg / m**3
    # print(np.float64(lit_cp * lit_delta))
    # plt.axhline(lit_cp * lit_delta, label='sim. param.', color='black')
    # plt.fill_between(tempcp, 0.95 * dcp1, 1.05 * dcp1, color='gray', alpha=0.5)
    plt.fill_between(tempcp, 0.95 * cp1, 1.05 * cp1, color='gray', alpha=0.5)
    # plt.fill_between(tempcp, 0.95 * dcp2, 1.05 * dcp2, color='gray', alpha=0.5)
    # plt.fill_between(tempcp, 0.95 * dcp3, 1.05 * dcp3, color='gray', alpha=0.5)
    # plt.plot(raw_puv, raw_puv + lit_cp * lit_delta, '.')
    plt.legend()
    plt.grid(True)

    # plt.show()
    # # plt.draw()
    # input()
    # exit()
    return(dcp1, f_dcp1)


def cal_cp_from_current5(rho, temp_den, den, length, epsht, temp_epsht,
                         raw_pyro=[], raw_time=[], raw_voltage=[]):
    import numpy as np
    # from scipy import interpolate
    from scipy.interpolate import interp1d as int1d
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter as sgf
    from scipy.signal import wiener
    # from scipy.signal import medfilt as med

    import warnings
    warnings.simplefilter('ignore', np.RankWarning)

    # static values for calculation
    # to be set to variable values if they are provided by experiment
    # c_p = 0.265  # J/(g*K)
    sigma = 5.670367e-8  # W m-2 K-4
    a_2 = 7.567e-16  # 1.00747e-18  # N/(K**3 * m)
    c = 299792458  # m/s
    r_o = 7.6e-3
    r_i = 6e-3
    # p = 20e-3  # m
    p = r_o * 2 * np.pi
    # p = (r_o + r_i) * 2 * np.pi
    # delta = 8.57 * 10**6  # g/m**3
    # epsilon_ht = 0.02
    s = (r_o**2 - r_i**2) * np.pi  # m**2
    # s = (r_o**2) * np.pi  # m**2
    # s = 8.85 * 10**(-6) # m**2
    t_a = 300  # K
    # d = 100  # distance center of rotary mirror to sample (perpendicular)
    # l = 70e-3  # sample length

    f_epsht = int1d(temp_epsht, epsht, kind='linear',
                    fill_value='extrapolate')
    f_den = int1d(temp_den, den, kind='linear',
                  fill_value='extrapolate')

    # max_temp = np.max(raw_pyro)
    # min_temp = np.min(raw_pyro)
    max_temp_point = (np.argwhere(raw_pyro == np.max(raw_pyro)))[0] + 1

    # max_temp_point = (np.argwhere(raw_pyro == np.max(raw_pyro)))[0]
    # print(max_temp)

    temp = raw_pyro
    time = raw_time
    voltage = raw_voltage

    voltage_off = (np.argwhere(np.abs(voltage) > 0)[-1]) + 1
    # voltage_on = (np.argwhere(np.abs(raw_voltage) > 0)[1])

    if max_temp_point > voltage_off:
        voltage_off = max_temp_point

    # temp_heat = np.split(temp, voltage_off)[0]
    # temp_curr = np.split(temp_heat, voltage_on)[1]
    temp_cool = np.split(temp, voltage_off)[1]

    min_temp = np.min(temp_cool)
    max_temp = np.max(temp_cool)

    sgflength = np.trunc(np.shape(temp_cool)[0] / 100)
    if sgflength % 2 == 0:
        sgflength += 1

    # print(sgflength)

    # time_heat = np.split(time, voltage_off)[0]
    # time_curr = np.split(time_heat, voltage_on)[1]
    time_cool = np.split(time, voltage_off)[1]
    # temp_cool2 = sgf(temp_cool, sgflength, 1, mode='nearest')
    # temp_cool2 = wiener(temp_cool)
    temp_cool2 = temp_cool
    plt.figure("temp_cool2")
    plt.plot(time_cool, temp_cool2, 'b')
    # plt.plot(time_cool, temp_cool, 'r')
    # temp_cool4 = med(temp_cool)
    # p_temp_cool5 = np.poly1d(np.polyfit(time_cool, temp_cool, deg=15))
    # temp_cool5 = p_temp_cool5(time_cool)

    # DTdt = np.gradient(temp_cool, edge_order=0) / \
    #     np.gradient(time_cool, edge_order=0)

    # dtDT = np.gradient(time_cool, edge_order=0) / \
    #     np.gradient(temp_cool, edge_order=0)
    # dtDT2_a = np.gradient(time_cool, edge_order=0) / \
    #     np.gradient(temp_cool2, edge_order=0)

    # print(np.gradient(time_cool) / np.gradient(temp_cool2))
    # # print(np.gradient(temp_cool2))
    # print((temp_cool**4 - t_a**4) * sigma * p / s)
    # exit()
    # plt.figure('temptest')
    # plt.plot(time_cool, temp_cool, label='temp cool')
    # plt.plot(time_cool, temp_cool2, label='temp cool 2')
    # input()

    # dtDT2 = sgf(np.gradient(time_cool, edge_order=0) /
    #             np.gradient(temp_cool2, edge_order=0),
    #             sgflength, 0, mode='nearest')
    # dtDT2 = sgf(np.gradient(time_cool, temp_cool2, edge_order=0),
    #             sgflength, 0, mode='nearest')
    # dtDT2 = wiener(np.gradient(time_cool, temp_cool2, edge_order=0))
    # dtDT2 = 1 / wiener(np.gradient(temp_cool2, time_cool, edge_order=0))
    dtDT2 = 1 / (wiener(np.gradient(temp_cool2, edge_order=0) / np.gradient(time_cool, edge_order=0)))
    # dtDT2 = 1 / sgf(np.gradient(temp_cool2, edge_order=0) / np.gradient(time_cool, edge_order=0),
    #             sgflength, 0, mode='nearest')
    # dtDT2 = sgf(np.diff(time_cool) /
    #             np.diff(temp_cool2),
    #             sgflength, 0, mode='nearest')
    # dtDT2 = sgf(np.gradient(time_cool / temp_cool2, np.mean(np.diff(time_cool)), edge_order=0),
    #             sgflength, 0, mode='nearest')
    # dtDT3 = np.gradient(time_cool, edge_order=0) / \
    #     np.gradient(temp_cool3, edge_order=0)
    # dtDT4 = np.gradient(time_cool, edge_order=0) / \
    #     np.gradient(temp_cool4, edge_order=0)

    # f_DTdt = int1d(temp_cool, DTdt, kind='linear',
    #                fill_value='extrapolate')
    f_dtDT = int1d(temp_cool2, dtDT2, kind='linear', axis=0,
                   fill_value='extrapolate')
    # f_dtDT_a = int1d(temp_cool, dtDT2_a, kind='linear',
    #                fill_value='extrapolate')
    # f_dtDT2 = int1d(temp_cool, dtDT2, kind='linear',
    #                fill_value='extrapolate')
    # f_dtDT3 = int1d(temp_cool, dtDT3, kind='linear',
    #                fill_value='extrapolate')
    # f_dtDT4 = int1d(temp_cool, dtDT4, kind='linear',
    #                fill_value='extrapolate')
    # f_dtDT = int1d(temp_cool, dtDT2, kind='linear',
    #                fill_value='extrapolate')

    # p_dtDT = np.poly1d(np.polyfit(temp_cool, dtDT, deg=71))

    plt.figure("dtDT2")
    plt.plot(temp_cool2, 1 / dtDT2)

    # input()
    # exit()

    # plt.figure("time_cool")
    # plt.plot(time_cool, temp_cool, '.', label="data")
    # plt.plot(time_cool, temp_cool2, label="sgf")
    # # plt.plot(time_cool, temp_cool3, label="wiener")
    # # plt.plot(time_cool, temp_cool4, label="med")
    # # plt.plot(time_cool, temp_cool5, label="polyfit")
    # plt.legend()


    # plt.figure("temp hist")
    # plt.plot(time_cool, temp_cool, '.')
    # plt.plot(time_cool, sgf(temp_cool, 51, 3))

    plt.figure("f_dtDT")
    plt.plot(temp_cool2, dtDT2, 'r.')
    plt.plot(temp_cool2, f_dtDT(temp_cool2), label="data")
    # plt.plot(temp_cool, f_dtDT2(temp_cool), label="sgf")
    # # plt.plot(temp_cool, f_dtDT3(temp_cool), label="wiener")
    # # plt.plot(temp_cool, f_dtDT4(temp_cool), label="med")
    # plt.legend()
    # input()
    # exit()


    # input()
    # exit()
    # plt.figure('f_den')
    # plt.plot(temp_cool, f_den(temp_cool))

    # f_rho = int1d(temp, rho, kind='linear',
    #               fill_value='extrapolate')

    # plt.figure('f_rho')
    # plt.plot(temp_cool, f_rho(temp_cool))

    # f_voltage = int1d(temp, voltage, kind='linear',
    #                   fill_value='extrapolate')

    # plt.figure('f_voltage')
    # plt.plot(temp_cool, f_voltage(temp_cool))

    tempcp = np.linspace(max_temp * 1, min_temp, num=10000)

    # plt.figure('test')
    # plt.plot(tempcp, -1/ (f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4) / s))
    # input()
    # exit()

    # plt.figure("dtDT2")
    # # plt.plot(temp_cool2, dtDT2_a, '-', label="dtDT2_a")
    # plt.plot(tempcp, f_dtDT_a(tempcp), label="f_dtDT2_a")
    # # plt.plot(temp_cool2, dtDT2, '-', label="dtDT2")
    # plt.plot(tempcp, f_dtDT(tempcp), label="f_dtDT2")
    # plt.legend()

    # plt.figure('rest')
    # plt.plot(temp_cool2, (- f_epsht(temp_cool2) * sigma * p * (temp_cool2**4 - t_a**4) / s), '.')
    # plt.plot(tempcp, (- f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4) / s))
    # plt.legend()
    # input()
    # exit()


    # dcp1 = f_dtDT(tempcp) * (f_voltage(tempcp)**2 / ((f_rho(tempcp) * length**4) -
            # (f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4) / s)))
    dcp1 = f_dtDT(tempcp) * (- (f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4) / s))
    # dcp1 = f_dtDT(tempcp) * (- ( 1 * sigma * p * (tempcp**4 - t_a**4) / s))
    # dcp1 = f_dtDT(tempcp) * (- (f_epsht(tempcp) * c * a_2 * p * (tempcp**4 - t_a**4) /( 4 * s )))

    # cp1 = f_dtDT(tempcp) * \
    #     ((f_voltage(tempcp)**2 / (f_rho(tempcp) * length**2)) - \
    #     ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s)) / \
    #     f_den(tempcp)

    # cp1 = f_dtDT(tempcp) * \
    #     (-((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s)) / \
    #     f_den(tempcp)

    cp1 = dcp1 / f_den(tempcp)

    f_dcp1 = int1d(tempcp, dcp1, kind='linear',
                   fill_value='extrapolate')

    plt.figure('cp')
    # plt.ticklabel_format(useOffset=False)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plt.title(r'volumetric heat capacity $\rho C_p$')
    # plt.title(r'volumetric heat capacity $C_p$')
    plt.title(r'heat capacity $C_p$')
    plt.xlabel(r'temperature / $K$')
    # plt.ylabel(r'volumetric heat capacity / $J/K \cdot m 3$')
    # plt.ylabel('volumetric heat capacity')
    plt.ylabel('heat capacity')
    # plt.plot(tempcp, dcp1, label=r'$\rho C_p$')
    plt.plot(tempcp, cp1, label=r'$C_p$')
    # plt.plot(tempcp, dcp2, label=r'$\delta C_p$ dcp2')
    # plt.plot(tempcp, dcp3, label=r'$\delta C_p$ d_3')
    # plt.plot(tempcp, f_dcp1(tempcp), label=r'func $\delta C_p$ (T)')
    # plt.plot(tempcp, dcp1_sgf, label=r'$\delta C_p - smooth$')
    # plt.plot(tempcp, dcp2)
    # plt.plot(tempcp, dcp3)
    # lit_cp = 265        # specific heat capacity J / (kg * K)
    # lit_delta = 8570    # density kg / m**3
    # print(np.float64(lit_cp * lit_delta))
    # plt.axhline(lit_cp * lit_delta, label='sim. param.', color='black')
    # plt.fill_between(tempcp, 0.95 * dcp1, 1.05 * dcp1, color='gray', alpha=0.5)
    plt.fill_between(tempcp, 0.95 * cp1, 1.05 * cp1, color='gray', alpha=0.5)
    # plt.fill_between(tempcp, 0.95 * dcp2, 1.05 * dcp2, color='gray', alpha=0.5)
    # plt.fill_between(tempcp, 0.95 * dcp3, 1.05 * dcp3, color='gray', alpha=0.5)
    # plt.plot(raw_puv, raw_puv + lit_cp * lit_delta, '.')
    plt.legend()
    plt.grid(True)

    # plt.show()
    # # plt.draw()
    # input()
    # exit()
    return(dcp1, f_dcp1)


def cal_cp_from_current6(temp_rho, rho, temp_den, den, length, epsht, temp_epsht,
                         raw_pyro=[], raw_time=[], raw_voltage=[]):
    import numpy as np
    from scipy import interpolate
    from scipy.interpolate import interp1d as int1d
    from scipy.integrate import quad
    from scipy.signal import wiener
    from scipy import signal
    # from scipy.signal import butter
    # from scipy.signal import filtfilt
    # from scipy.interpolate import UnivariateSpline as US
    # from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter as sgf
    # from scipy.optimize import curve_fit
    from functions_py3 import mean_diff

    import warnings
    warnings.simplefilter('ignore', np.RankWarning)
    # current =
    # DTdt1 = np.gradient(raw_pyro, edge_order=1) / \
    #     np.gradient(raw_time, edge_order=1)
    # T_a = 300  # K
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

    f_epsht = int1d(temp_epsht, epsht, kind='linear',
                    fill_value='extrapolate')
    f_den = int1d(temp_den, den, kind='linear',
                  fill_value='extrapolate')
    f_rho = int1d(temp_rho, rho, kind='linear',
                  fill_value='extrapolate')

    max_temp = np.max(raw_pyro)
    min_temp = np.min(raw_pyro)
    # if min_temp < 400:
    #     min_temp = 400
    max_temp_point = (np.argwhere(raw_pyro == np.max(raw_pyro)))[0]
    # print(max_temp)
    temp = raw_pyro
    time = raw_time
    voltage = raw_voltage

    voltage_off = (np.argwhere(np.abs(voltage) > 0)[-1]) + 2
    voltage_off = max_temp_point

    # print(voltage_off)
    # print(voltage[voltage_off - 1])
    # print(voltage[voltage_off])
    # print(voltage[voltage_off + 1])
    # input()

    temp_curr = np.split(raw_pyro, voltage_off)[0]
    temp_cool = np.split(raw_pyro, voltage_off)[1]
    time_curr = np.split(raw_time, voltage_off)[0]
    time_cool = np.split(raw_time, voltage_off)[1]
    voltage_curr = np.split(voltage, voltage_off)[0]

    # print(np.shape(temp_curr))
    # print(np.mean(np.diff(time_curr)))
    # print(voltage)
    # input()

    # DTdt_curr = np.gradient(temp_curr, edge_order=0)[1:] / \
    #     np.diff(time_curr)

    # DTdt_cool = np.gradient(temp_cool, edge_order=0)[1:] / \
    #     np.diff(time_cool)

    # DTdt_curr = mean_diff(temp_curr) / \
    #     np.diff(time_curr)[:-1]

    # DTdt_cool = mean_diff(temp_cool) / \
    #     np.diff(time_cool)[:-1]

    # First, design the Buterworth filter
    # N  = 60    # Filter order
    # Wn = 0.3 # Cutoff frequency
    # B, A = signal.butter(N, Wn, output='ba')
    # # B, A = signal.ellip(4, 0.01, 120, 0.01, 'low')
    # filt_temp_curr = signal.filtfilt(B,A, temp_curr,method='gust')

    # DTdt_curr = mean_diff2(temp_curr, time_curr)
    # DTdt_curr = np.gradient(temp_curr, time_curr, edge_order=2)
    # DTdt_curr = mean_diff2(sgf(temp_curr, 3, 1), time_curr)
    DTdt_curr = mean_diff2(wiener(temp_curr), time_curr)

    # DTdt_cool = mean_diff2(temp_cool, time_cool)
    # DTdt_cool = np.gradient(temp_cool, time_cool, edge_order=2)
    DTdt_cool = mean_diff2(wiener(temp_cool), time_cool)

    # f_DTdt_curr = int1d(temp_curr, DTdt_curr, kind='linear',
    #                     fill_value='extrapolate')
    # f_DTdt_cool = int1d(temp_cool, DTdt_cool, kind='linear',
    #                     fill_value='extrapolate')

    f_DTdt_curr = int1d(temp_curr[1:-1], DTdt_curr, kind='linear',
                        fill_value='extrapolate')
    f_DTdt_cool = int1d(temp_cool[1:-1], DTdt_cool, kind='linear',
                        fill_value='extrapolate')

    degree = 21

    f_temp_curr = int1d(time_curr, temp_curr, kind='linear',
                        fill_value='extrapolate')
    f_temp_cool = int1d(time_cool, temp_cool, kind='linear',
                        fill_value='extrapolate')

    f_voltage_curr = int1d(temp_curr, voltage_curr, kind='linear',
                           fill_value='extrapolate')
    p_voltage_curr = np.poly1d(np.polyfit(temp_curr, voltage_curr, deg=degree))

    # f_rho_curr = int1d(temp_curr, rho, kind='linear',
    #                    fill_value='extrapolate')
    # p_rho_curr = np.poly1d(np.polyfit(temp_curr, rho, deg=degree))

    # time_cool_new = np.linspace(min(time_cool), 20, 1000)
    temp_cool_new = np.linspace(min(temp_cool), max(temp_cool), 1000)
    temp_curr_new = np.linspace(min(temp_curr), max(temp_curr), 1000)

    # tempcp = np.linspace(max_temp - 50, 1 * max_temp, 1000)
    # tempcp = np.linspace( 1 * max_temp2, u_temp_cool(time_cool[idx]), num=100)
    tempcp = np.linspace(max_temp * 0.95, min_temp, num=10000)

    # f_temp_cool = int1d(temp_cool, time_cool, kind='linear',
    #                     fill_value='extrapolate')

    # plt.figure('differentiate test')
    # plt.plot(temp_cool, f_DTdt_cool(temp_cool), 'b.', label='points')
    # plt.plot(tempcp, f_DTdt_cool(tempcp), label='int1d')
    # # plt.plot(temp_cool, , label='int1d')
    # plt.legend()
    # input()
    # exit()


    dcp1_curr = 1 / (np.abs(f_DTdt_curr(tempcp))) * \
        ((f_voltage_curr(tempcp)**2 / (f_rho(tempcp) * length**2)) - \
        ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s))

    dcp1_cool = (1 / f_DTdt_cool(tempcp)) * (- (f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4) / s))

    # cp1 = 1 / (np.abs(f_DTdt(tempcp))) * \
    #     ((f_voltage_curr(tempcp)**2 / (f_rho_curr(tempcp) * length**2)) - \
    #     ((f_epsht(tempcp) * sigma * p * (tempcp**4 - t_a**4)) / s)) / \
    #     f_den(tempcp)

    cp1_curr = dcp1_curr / f_den(tempcp)

    cp1_cool = dcp1_cool / f_den(tempcp)

    f_dcp1_curr = int1d(tempcp, dcp1_curr, kind='linear',
                        fill_value='extrapolate')
    f_dcp1_cool = int1d(tempcp, dcp1_cool, kind='linear',
                        fill_value='extrapolate')
    # f_dcp2 = interpolate.interp1d(tempcp, dcp2, kind='linear',
    #                               fill_value='extrapolate')

    # for i in range(len(tempcp)):
    #     print(f_DTdt_cool(tempcp[i]))
    #     dcp[i] = f_voltage_heat(tempcp[i])**2 / \
    #              (f_rho_heat(tempcp[i]) * length**2) * \
    #              1 / (f_DTdt_heat(tempcp[i]) - f_DTdt_cool(tempcp[i]))
        # print((f_rho_heat(tempcp[i]) * length**2))
        # print(f_DTdt_heat(tempcp[i]) - f_DTdt_cool(tempcp[i]))
    # exit()
    # dcp1_sgf = sgf(dcp1, 51, 5)

    # print(dcp1)
    # print(dcp2)
    # print("%f , %f" % (dcp1, dcp2))
    # print(dcp3)
    # plt.figure('dcp-check')
    # plt.plot(tempcp, dcp1, label='dcp1')
    # plt.plot(tempcp, dcp2, label='dcp2')
    # plt.plot(tempcp, dcp3, label='dcp3')
    # plt.legend()
    # plt.show()
    # input()
    # exit()
    # plt.show()
    # exit()
    # import matplotlib.ticker as mtick
    plt.figure('dcp')
    # plt.ticklabel_format(useOffset=False)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plt.title(r'volumetric heat capacity $\rho C_p$')
    plt.title(r'volumetric heat capacity $C_p$')
    plt.xlabel(r'temperature / $K$')
    # plt.ylabel(r'volumetric heat capacity / $J/K \cdot m 3$')
    plt.ylabel('volumetric heat capacity')
    # plt.plot(tempcp, f_dcp1_cool(tempcp), label=r'func $\delta C_p$ (T) cool')
    # plt.plot(tempcp, dcp1_cool, label=r'$\rho C_p$')
    # plt.plot(tempcp, cp1_cool, label=r'$C_p$_cool')
    # plt.plot(tempcp, cp1_curr, label=r'$C_p$_curr')
    # plt.plot(tempcp, dcp2, label=r'$\delta C_p$ dcp2')
    # plt.plot(tempcp, dcp3, label=r'$\delta C_p$ d_3')
    plt.plot(tempcp, f_dcp1_curr(tempcp), label=r'func $\delta C_p$ (T) curr')
    # plt.plot(tempcp, dcp1_sgf, label=r'$\delta C_p - smooth$')
    # plt.plot(tempcp, dcp2)
    # plt.plot(tempcp, dcp3)
    lit_cp = 265        # specific heat capacity J / (kg * K)
    lit_delta = 8570    # density kg / m**3
    # print(np.float64(lit_cp * lit_delta))
    # plt.axhline(lit_cp * lit_delta, label='sim. param.', color='black')
    # plt.fill_between(tempcp, 0.95 * dcp1_curr, 1.05 * dcp1_curr, color='gray', alpha=0.5)
    plt.fill_between(tempcp, 0.90 * dcp1_curr, 1.10 * dcp1_curr, color='gray', alpha=0.5)
    # plt.fill_between(tempcp, 0.95 * cp1_curr, 1.05 * cp1_curr, color='gray', alpha=0.5)
    # plt.fill_between(tempcp, 0.90 * cp1_curr, 1.10 * cp1_curr, color='gray', alpha=0.5)
    # plt.fill_between(tempcp, 0.95 * dcp1_cool, 1.05 * dcp1_cool, color='gray', alpha=0.5)
    # plt.fill_between(tempcp, 0.95 * dcp2, 1.05 * dcp2, color='gray', alpha=0.5)
    # plt.fill_between(tempcp, 0.95 * dcp3, 1.05 * dcp3, color='gray', alpha=0.5)
    # plt.plot(raw_puv, raw_puv + lit_cp * lit_delta, '.')
    plt.legend()
    plt.grid(True)

    # plt.show()
    # # plt.draw()
    # input()
    # exit()
    return(dcp1_cool, f_dcp1_cool)
    # return(dcp1_curr, f_dcp1_curr)


def sim_e_ht(raw_puv=[], raw_time=[], raw_pv=[],
                          raw_pv_time=[], litdata=[], plotresult=False):
    import numpy as np
    import matplotlib.pyplot as plt
    # import sys
    # import platform
    from scipy import interpolate
    # from functions import getcalibration
    # from functions import cal_pyrotemp

    # if raw_puv == []:
    #     raw_puv = np.load('raw_dat.npy')
    # if raw_time == []:
    #     raw_time = np.load('raw_time.npy')
    # if raw_pv == []:
    #     raw_pv = np.load('raw_pv.npy')
    # if raw_pv_time == []:
    #     raw_pv_time = np.load('raw_pv_time.npy')

    sigma = 5.670367e-8  # W m-2 K-4

    lit_delta = litdata[0]
    lit_cp = litdata[1]
    lit_epsht = litdata[2]
    # p = 20 * 10**(-3)  # m
    # s = p**2 * np.pi  # m**2

    puv_means = []
    cpt = []

    c_puv = []
    c_pv = []
    c_time = []
    c_pv_time = []

    c_puv = raw_puv[:, np.int(np.shape(raw_puv)[1] / 2)]
    c_pv = raw_pv[:, np.int(np.shape(raw_pv)[1] / 2)]
    c_time = raw_time[:, np.int(np.shape(raw_time)[1] / 2)]
    c_pv_time = raw_pv_time[:, np.int(np.shape(raw_pv_time)[1] / 2)]

    # if real values from pyrometers are achieved
    # they have to be converted to kelvin!!!
    # c_pv_90 = cal_pyrotemp(c_pv, 'pv')
    # c_puv_90 = cal_pyrotemp(c_puv, 'puv')
    # for demovalues no convertion is required
    c_puv_90 = c_puv
    c_pv_90 = c_pv


    I_pv = temp_to_intensity(c_pv_90, 900e-9, 1.0)
    I_puv = temp_to_intensity(c_puv_90, 900e-9, lit_epsht)

    epsht = I_puv / I_pv

    # plt.figure("test")
    # plt.plot(epsht)
    # input()
    # exit()

    # split = 1.0
    # pulse_stop = 0.4

    f_epsht = interpolate.interp1d(c_puv_90, epsht, kind='linear',
                                   fill_value='extrapolate')

    # print(c_puv)

    # c_puv = c_puv[raw_time[0] <= split]
    # c_puv = c_puv[raw_time[0] >= pulse_stop]
    # c_pv = c_pv[raw_time[0] <= split]
    # c_pv = c_pv[raw_time[0] >= pulse_stop]
    # c_time = c_time[raw_time[0] <= split]
    # c_time = c_time[raw_time[0] >= pulse_stop]
    # c_pv_time = c_pv_time[raw_time[0] <= split]
    # c_pv_time = c_pv_time[raw_time[0] >= pulse_stop]

    # print(np.shape(c_puv))

    # t_a = 300  # K

    # def sim_cal_delta_cp(puv_90, pv_90, time, t_a=300, p=7.6e-3 * 2 * np.pi):
    #     if t_a not in locals():
    #         t_a = 300

    #     if p not in locals():
    #         p = 7.6e-3 * 2 * np.pi  # m

    #     s = ((7.6e-3)**2 - (6e-3)**2) * np.pi  # m**2
    #     sigma = 5.670367e-8  # W m-2 K-4

    #     # dt_puv_90 = np.diff(puv_90) / np.diff(time)
    #     # print(np.shape(puv_90))
    #     dt_puv_90 = np.gradient(puv_90, edge_order=1) / np.gradient(time, edge_order=1)
    #     # print(time)
    #     # print(np.diff(puv_90))
    #     # puv_90_s = puv_90[:len(np.diff(puv_90))]
    #     puv_90_s = puv_90[:len(np.gradient(puv_90, edge_order=1))]
    #     # center_puv = np.int(len(puv_90_s) / 2)
    #     # puv_90_c = puv_90[center_puv]
    #     # print(puv_90_c)
    #     # print(np.shape(np.diff(pv_90)))
    #     # exit()
    #     # print(np.shape(np.diff(time)))
    #     # dt_pv_90 = np.diff(pv_90) / np.diff(time)
    #     # pv_90_s = pv_90[:len(np.diff(pv_90))]
    #     dt_pv_90 = np.gradient(pv_90, edge_order=1) / np.gradient(time, edge_order=1)
    #     pv_90_s = pv_90[:len(np.gradient(pv_90, edge_order=1))]
    #     # center_pv = np.int(len(pv_90_s) / 2)
    #     # pv_90_c = pv_90[center_pv]
    #     # print(np.diff(pv_90))
    #     # print(np.diff(time))
    #     # print(dt_pv_90**(-1))
    #     # print(np.diff(time) / np.diff(pv_90))
    #     # exit()
    #     # print(np.shape(pv_90_s))

    #     I_pv = temp_to_intensity(pv_90_s, 900e-9, 1.0)
    #     I_puv = temp_to_intensity(puv_90_s, 900e-9, lit_epsht)

    #     epsht = I_puv / I_pv
    #     # print(pv_90_s)
    #     # print(puv_90_s[1:5])
    #     # print(puv_90_s[1:5]**4)
    #     # exit()

    #     # #to generate plot over whole range set split in "tc_sim_p3_V2.py" to 10 or very high
    #     # plt.plot(time, pv_90_s, 'r', label=r'black body cavity $\varepsilon = 1$')
    #     # plt.plot(time, puv_90_s, 'b', label=r'back of specimen $\varepsilon = 0.3$')
    #     # # plt.plot(puv_90_s)
    #     # plt.grid(True)
    #     # plt.xlabel('time / s')
    #     # plt.ylabel('temperature / K')
    #     # plt.title('temperature history')
    #     # # plt.xlim([0, 0.6])
    #     # plt.legend()
    #     # # plt.grid(True)
    #     # plt.show()
    #     # exit()

    #     # import matplotlib.pyplot as plt

    #     # plt.plot(time, dt_pv_90**(-1))
    #     # plt.plot(time, 1 / dt_pv_90)
    #     # plt.show()
    #     # exit()

    #     # epsht = (pv_90_s**4 - t_a**4) / (puv_90_s**4 - t_a**4) * \
    #     #     dt_puv_90 / dt_pv_90

    #     # print(dt_puv_90 / dt_pv_90)
    #     # print(dt_pv_90**(-1))

    #     f_epsht = interpolate.interp1d(puv_90_s, epsht, kind='linear',
    #                                    fill_value='extrapolate')

    #     delta_cp = (-1) * (dt_pv_90)**(-1) * 1 * \
    #         sigma * p * (pv_90_s**4 - t_a**4) / s

    #     delta_cp[delta_cp < 0.] = 0.
    #     # print(pv_90_s)
    #     # print(delta_cp)
    #     # print(lit_delta * lit_cp)
    #     # exit()
    #     delta_cp_puv = (-1) * (dt_puv_90)**(-1) * f_epsht(puv_90_s) * \
    #         sigma * p * (puv_90_s**4 - t_a**4) / s

    #     delta_cp_puv[delta_cp_puv < 0.] = 0.

    #     # print(puv_90_s)
    #     # print(delta_cp_puv)
    #     # print((dt_puv_90)**(-1))
    #     # print(np.diff(time))
    #     # plt.plot(pv_90_s, pv_90_s**4)
    #     # plt.plot(time, -dt_pv_90**(-1) * pv_90_s**4 * sigma)

    #     # plt.axhline(y=lit_delta * lit_cp)
    #     # plt.show()
    #     # exit()

    #     # print(delta_cp)

    #     f_delta_cp = interpolate.interp1d(pv_90_s, delta_cp,
    #                                       kind='linear',
    #                                       fill_value='extrapolate'
    #                                       # fill_value=(min(delta_cp), max(delta_cp))
    #                                       )

    #     f_delta_cp_puv = interpolate.interp1d(puv_90_s, delta_cp,
    #                                           kind='linear',
    #                                           fill_value='extrapolate'
    #                                           # fill_value=(min(delta_cp_puv), max(delta_cp_puv))
    #                                           )

    #     # calculate epsht from delta_cp values!!!
        
    #     # plt.plot(pv_90_s, delta_cp)
    #     # plt.figure()
    #     # plt.plot(pv_90_s, f_delta_cp(pv_90_s))
    #     # plt.show()
    #     # exit()


    #     # delta_cp = (-1) * (dt_puv_90)**(-1) * f_epsht(puv_90_s) * sigma
    #     #             * p * (puv_90_s**4 - t_a**4) / s

    #     # f_delta_cp = interpolate.interp1d(puv_90_s, delta_cp, kind='linear',
    #     #                                   fill_value='extrapolate')

    #     return(epsht, delta_cp, f_epsht, f_delta_cp,
    #            delta_cp_puv, f_delta_cp_puv)

    # (epsht, delta_cp, f_epsht, f_delta_cp, delta_cp_puv, f_delta_cp_puv) = \
    #     sim_cal_delta_cp(c_puv_90, c_pv_90, c_time, t_a=300,
    #                      p=7.6e-3 * 2 * np.pi)

    # print(delta_cp)
    # print(dir(interpolate.interp1d))
    # print(f_epsht.item())
    # print(c_puv_90)
    # print(f_epsht(1800), f_epsht(1800.0001), f_epsht(1801))

    # plt.plot(c_puv_90, fun_epsht(c_puv_90), 'r.')
    # plt.show()
    # exit()

    # # print(min(c_puv_90))

    # temp_epsht = np.arange(np.ceil(min(c_puv_90))
    #                        np.floor(max(c_puv_90)), 0.1)
    # y_epsht = fun_epsht(temp_epsht)
    # print(fun_epsht(2000))

    # print(min(c_puv))
    # exit()
    # # print(type(epsht_all))
    # plt.figure('epsht_all')
    # print(min(c_pv_90))
    # print(max(c_pv_90))
    # print(plotresult)
    # print(c_pv_90)

    if plotresult is True:
        # newrange = np.linspace(min(c_puv_90), max(c_pv_90))
        plt.figure('normal emissivity')
        plt.title(r'normal emissivity $\varepsilon_{n}$')
        plt.plot(c_puv_90, epsht, 'r.', label='data')
        plt.fill_between(c_puv_90[:],
                         epsht * 0.95,
                         epsht * 1.05,
                         facecolor='black', alpha=0.5)
        # plt.plot(newrange, f_epsht(newrange), label='lit.')
        plt.axhline(y=lit_epsht, label='sim. param.', color='black')
        plt.ylim([lit_epsht - 0.1, lit_epsht + 0.1])
        # plt.axhline(y=1)
        plt.xlabel('temperature / K')
        plt.ylabel(r'normal emissivity $\varepsilon_{n}$ / a.u.')
        plt.legend()
        plt.grid(True)

        # # newrange1 = np.linspace(min(c_puv_90), max(c_pv_90))
        # plt.figure('volumetric heat capacity')
        # # plt.title(r'volumetric heat capacity $C_p$')
        # plt.plot(c_pv_90, delta_cp, 'g.', label='data')
        # # plt.plot(c_puv_90[:], delta_cp_puv, 'r.', label='data_puv')
        # # plt.plot(newrange, f_delta_cp(newrange), label='sim. param.')
        # plt.fill_between(c_pv_90,
        #                  delta_cp * 0.95,
        #                  delta_cp * 1.05,
        #                  facecolor='black', alpha=0.5)
        # # plt.fill_between(c_puv_90[:],
        # #                  delta_cp_puv * 0.95,
        # #                  delta_cp_puv * 1.05,
        # #                  facecolor='black', alpha=0.5)
        # plt.axhline(y=lit_delta * lit_cp, label='sim. param.')
        # # print(lit_delta * lit_cp)
        # # plt.ylim([0, lit_delta * lit_cp * 1.05])
        # plt.legend()
        # # plt.ylim(ymin=0)
        # plt.grid(True)
        # plt.xlabel('temperature / K')
        # plt.ylabel(r'volumetric heat capacity $C_p$ / $J \cdot m^{-3} \cdot K^{-1}$')
        # plt.show(0)
        # plt.show(block=False)

        # plt.show()
        # exit()
    return(epsht, f_epsht, c_time)


def sim_e_ht2(c_pv=[], c_puv=[], c_voltage=[], litdata=[], plotresult=False):
    import numpy as np
    import matplotlib.pyplot as plt
    # import sys
    # import platform
    from scipy import interpolate
    # from functions import getcalibration
    # from functions import cal_pyrotemp
    from functions_py3 import intensity_to_temp
    
    wl = 900  # nm

    sigma = 5.670367e-8  # W m-2 K-4

    lit_delta = litdata[0]
    lit_cp = litdata[1]
    lit_epsht = litdata[2]
    # p = 20 * 10**(-3)  # m
    # s = p**2 * np.pi  # m**2

    # plt.plot(c_voltage[:, 2])
    # input()

    v_off = np.int((np.argwhere(np.abs(c_voltage[:, 2]) != 0))[1]) + 1

    # print(np.shape(c_voltage[:, 2]))

    # print(v_off)

    # print(c_puv[:,2])
    # print(c_pv[:,2])
    # exit()

    c_puv = c_puv[v_off:, :]
    c_pv = c_pv[v_off:, :]

    epsht = c_puv[:, 2] / c_pv[:, 2]

    # plt.figure("test")
    # plt.plot(epsht)
    # input()
    # exit()

    c_pv_90 = intensity_to_temp(c_pv[:, 2], wl, epsht, n=1)
    c_puv_90 = intensity_to_temp(c_puv[:, 2], wl, epsht, n=1)
    # print(c_pv_90)
    # plt.plot(c_pv_90)
    # input()
    # exit()
    # split = 1.0
    # pulse_stop = 0.4

    f_epsht = interpolate.interp1d(c_puv_90, epsht, kind='linear',
                                   fill_value='extrapolate')

    if plotresult is True:
        # newrange = np.linspace(min(c_puv_90), max(c_pv_90))
        plt.figure('normal emissivity')
        plt.title(r'normal emissivity $\varepsilon_{n}$')
        plt.plot(c_puv_90, epsht, 'r.', label='data')
        plt.fill_between(c_puv_90[:],
                         epsht * 0.95,
                         epsht * 1.05,
                         facecolor='black', alpha=0.5)
        # plt.plot(newrange, f_epsht(newrange), label='lit.')
        # plt.axhline(y=lit_epsht, label='sim. param.', color='black')
        plt.ylim([lit_epsht - 0.1, lit_epsht + 0.1])
        # plt.axhline(y=1)
        plt.xlabel('temperature / K')
        plt.ylabel(r'normal emissivity $\varepsilon_{n}$ / a.u.')
        plt.legend()
        plt.grid(True)

        # # newrange1 = np.linspace(min(c_puv_90), max(c_pv_90))
        # plt.figure('volumetric heat capacity')
        # # plt.title(r'volumetric heat capacity $C_p$')
        # plt.plot(c_pv_90, delta_cp, 'g.', label='data')
        # # plt.plot(c_puv_90[:], delta_cp_puv, 'r.', label='data_puv')
        # # plt.plot(newrange, f_delta_cp(newrange), label='sim. param.')
        # plt.fill_between(c_pv_90,
        #                  delta_cp * 0.95,
        #                  delta_cp * 1.05,
        #                  facecolor='black', alpha=0.5)
        # # plt.fill_between(c_puv_90[:],
        # #                  delta_cp_puv * 0.95,
        # #                  delta_cp_puv * 1.05,
        # #                  facecolor='black', alpha=0.5)
        # plt.axhline(y=lit_delta * lit_cp, label='sim. param.')
        # # print(lit_delta * lit_cp)
        # # plt.ylim([0, lit_delta * lit_cp * 1.05])
        # plt.legend()
        # # plt.ylim(ymin=0)
        # plt.grid(True)
        # plt.xlabel('temperature / K')
        # plt.ylabel(r'volumetric heat capacity $C_p$ / $J \cdot m^{-3} \cdot K^{-1}$')
        # plt.show(0)
        # plt.show(block=False)

        # plt.show()
        # exit()
    return(epsht, f_epsht, c_pv_90)


def sim_e_ht3(c_pv=[], c_puv=[], c_voltage=[], litdata=[], plotresult=False):
    import numpy as np
    import matplotlib.pyplot as plt
    # import sys
    # import platform
    from scipy import interpolate
    # from functions import getcalibration
    # from functions import cal_pyrotemp
    from functions_py3 import intensity_to_temp
    from scipy.signal import savgol_filter as sgf
    from scipy.signal import wiener
    from scipy.signal import medfilt2d as med2d
    from scipy.signal import medfilt as med
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)

    wl = 900  # nm

    sigma = 5.670367e-8  # W m-2 K-4

    lit_delta = litdata[0]
    lit_cp = litdata[1]
    lit_epsht = litdata[2]
    # p = 20 * 10**(-3)  # m
    # s = p**2 * np.pi  # m**2

    # plt.plot(c_voltage[:, 2])
    # input()
    # plt.figure("test")
    # plt.plot(c_voltage)
    # print(c_voltage)
    # # exit()
    # print(np.shape(c_voltage))
    # input()
    max_temp_point = np.int((np.argwhere(c_puv == np.max(c_puv)))[0] + 1)

    v_off = np.int((np.argwhere(np.abs(c_voltage) != 0))[1]) + 1

    if max_temp_point > v_off:
        v_off = max_temp_point

    # print(v_off)

    # print(np.shape(c_voltage[:, 2]))

    # print(v_off)

    # print(c_puv[:,2])
    # print(c_pv[:,2])
    # exit()

    c_puv = c_puv[v_off:]
    c_pv = c_pv[v_off:]

    epsht = c_puv / c_pv

    # plt.figure("test")
    # plt.plot(epsht)
    # input()
    # exit()
    # plt.figure("test1")
    # plt.plot(c_puv, '-')
    # plt.figure("test2")
    # plt.plot(c_pv, '-')

    c_pv_90 = intensity_to_temp(c_pv, wl, 1, n=1)
    c_puv_90 = intensity_to_temp(c_puv, wl, epsht, n=1)

    c_puv_int = np.linspace(np.min(c_puv), np.max(c_puv), num=1000)
    c_puv_90_temp = np.linspace(np.min(c_puv_90), np.max(c_puv_90), num=1000)
    # print(c_pv_90)
    # plt.plot(c_pv_90)
    # input()
    # exit()
    # split = 1.0
    # pulse_stop = 0.4

    # f_epsht = interpolate.interp1d(c_puv_90, wiener(epsht,np.int(len(epsht)/25)), kind='linear',
                                   # fill_value='extrapolate')
    f_epsht = interpolate.interp1d(c_puv_90, epsht, kind='linear',
                                   fill_value='extrapolate')

    # f_epsht_int = interpolate.interp1d(np.insert(c_puv, 0, 0), wiener(np.insert(epsht, 0, epsht[0]),np.int((len(epsht)+1)/25)), kind='linear',
    #                                    fill_value='extrapolate')
    f_epsht_int = interpolate.interp1d(np.insert(c_puv, 0, 0),np.insert(epsht, 0, epsht[0]), kind='linear',
                                       fill_value='extrapolate')

    # c_puv_90_int = temp_to_intensity2(c_puv_90, wl, f_epsht_int, n=1)
    degree = 20
    p_epsht = np.poly1d(np.polyfit(c_puv_90, epsht, deg=degree))
    p_epsht_int = np.poly1d(np.polyfit(np.insert(c_puv, 0, 0), np.insert(epsht, 0, epsht[0]), deg=degree))

    # c_puv_int = np.linspace(0, np.max(c_puv), num=10000)
    # plt.figure("epsht temp")
    # plt.plot(c_puv_90, epsht, '.')
    # plt.plot(c_puv_90_temp, f_epsht(c_puv_90_temp))
    # plt.plot(c_puv_90_temp, p_epsht(c_puv_90_temp))
    # plt.figure("epsht intensity")
    # plt.plot(c_puv, epsht, '.')
    # plt.plot(c_puv_int, f_epsht_int(c_puv_int))
    # plt.plot(c_puv_int, p_epsht_int(c_puv_int))
    # print(f_epsht_int(c_puv_int))
    # input()
    # exit()
    # print(c_puv)
    # print(c_puv_90_int)
    # exit()

    if plotresult is True:
        # newrange = np.linspace(min(c_puv_90), max(c_pv_90))
        plt.figure('normal emissivity')
        plt.title(r'normal emissivity $\varepsilon_{n}$')
        plt.plot(c_puv_90, epsht, 'r.', label='data')
        plt.fill_between(c_puv_90[:],
                         epsht * 0.95,
                         epsht * 1.05,
                         facecolor='black', alpha=0.5)
        plt.plot(c_puv_90, p_epsht(c_puv_90), 'g', label='poly fit')
        # plt.plot(newrange, f_epsht(newrange), label='lit.')
        # plt.axhline(y=lit_epsht, label='sim. param.', color='black')
        plt.ylim([lit_epsht - 0.1, lit_epsht + 0.1])
        # plt.axhline(y=1)
        plt.xlabel('temperature / K')
        plt.ylabel(r'normal emissivity $\varepsilon_{n}$ / a.u.')
        plt.legend()
        plt.grid(True)

        # # newrange1 = np.linspace(min(c_puv_90), max(c_pv_90))
        # plt.figure('volumetric heat capacity')
        # # plt.title(r'volumetric heat capacity $C_p$')
        # plt.plot(c_pv_90, delta_cp, 'g.', label='data')
        # # plt.plot(c_puv_90[:], delta_cp_puv, 'r.', label='data_puv')
        # # plt.plot(newrange, f_delta_cp(newrange), label='sim. param.')
        # plt.fill_between(c_pv_90,
        #                  delta_cp * 0.95,
        #                  delta_cp * 1.05,
        #                  facecolor='black', alpha=0.5)
        # # plt.fill_between(c_puv_90[:],
        # #                  delta_cp_puv * 0.95,
        # #                  delta_cp_puv * 1.05,
        # #                  facecolor='black', alpha=0.5)
        # plt.axhline(y=lit_delta * lit_cp, label='sim. param.')
        # # print(lit_delta * lit_cp)
        # # plt.ylim([0, lit_delta * lit_cp * 1.05])
        # plt.legend()
        # # plt.ylim(ymin=0)
        # plt.grid(True)
        # plt.xlabel('temperature / K')
        # plt.ylabel(r'volumetric heat capacity $C_p$ / $J \cdot m^{-3} \cdot K^{-1}$')
        # plt.show(0)
        # plt.show(block=False)

        # plt.show()
        # exit()
    return(epsht, f_epsht, c_pv_90, f_epsht_int, c_puv_int)
    # return(epsht, p_epsht, c_pv_90, p_epsht_int, c_puv_int)


def rectangle_selector():
    # from __future__ import print_function
    """
    Do a mouseclick somewhere, move the mouse to some destination, release
    the button.  This class gives click- and release-events and also draws
    a line or a box from the click-point to the actual mouseposition
    (within the same axes) until the button is released.  Within the
    method 'self.ignore()' it is checked whether the button from eventpress
    and eventrelease are the same.

    """
    from matplotlib.widgets import RectangleSelector
    import numpy as np
    import matplotlib.pyplot as plt

    rect = np.ndarray(())
    x1 = .0
    y1 = .0
    x2 = .0
    y2 = .0
    # toggle_selector.rect = ()


    def line_select_callback(eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        line_select_callback.x1 = x1
        line_select_callback.y1 = y1
        line_select_callback.x2 = x2
        line_select_callback.y2 = y2
        # print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        # print(" The button you used were: %s %s" % (eclick.button, erelease.button))
        # print(x1, y1)
        # print(x2, y2)
        return(x1, y1, x2, y2, eclick.button, erelease.button)

    # def line_select():


    # print(x1, y1)
    # print(x2, y2)


    def toggle_selector(event):
        print(' Key pressed.')
        # print(event.key)
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)
        if event.key is 'enter' and toggle_selector.RS.active:
            bla = (line_select_callback.x1,
                             line_select_callback.y1,
                             line_select_callback.x2,
                             line_select_callback.y2, )
            # toggle_selector.rect = (line_select_callback.x1,
            #                  line_select_callback.y1,
            #                  line_select_callback.x2,
            #                  line_select_callback.y2, )
            toggle_selector.rect = bla
            print("test")
            print(bla)


    fig, current_ax = plt.subplots()                 # make a new plotting range
    N = 100000                                       # If N is large one can see
    x = np.linspace(0.0, 10.0, N)                    # improvement by use blitting!

    plt.plot(x, +np.sin(.2*np.pi*x), lw=3.5, c='b', alpha=.7)  # plot something
    plt.plot(x, +np.cos(.2*np.pi*x), lw=3.5, c='r', alpha=.5)
    plt.plot(x, -np.sin(.2*np.pi*x), lw=3.5, c='g', alpha=.3)

    print("\n      click  -->  release")

    # drawtype is 'box' or 'line' or 'none'
    toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()
    # x1, y1, x2, y2, bla, blub = line_select_callback(eclick, erelease)
    print(bla)
    # print(x2, y2)

    # print(np.shape(rect))


def span_selector():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
    ax1.set(facecolor='#FFFFCC')

    x = np.arange(0.0, 5.0, 0.01)
    y = np.sin(2*np.pi*x) + 0.5*np.random.randn(len(x))

    ax1.plot(x, y, '-')
    ax1.set_ylim(-2, 2)
    ax1.set_title('Press left mouse button and drag to test')

    ax2.set(facecolor='#FFFFCC')
    line2, = ax2.plot(x, y, '-')


    def onselect(xmin, xmax):
        indmin, indmax = np.searchsorted(x, (xmin, xmax))
        indmax = min(len(x) - 1, indmax)

        thisx = x[indmin:indmax]
        thisy = y[indmin:indmax]
        line2.set_data(thisx, thisy)
        ax2.set_xlim(thisx[0], thisx[-1])
        ax2.set_ylim(thisy.min(), thisy.max())
        fig.canvas.draw()

    # set useblit True on gtkagg for enhanced performance
    span = SpanSelector(ax1, onselect, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red'))


    plt.show()


def span_selector2(ax1, ax2):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
    ax1.set(facecolor='#FFFFCC')

    x = np.arange(0.0, 5.0, 0.01)
    y = np.sin(2*np.pi*x) + 0.5*np.random.randn(len(x))

    ax1.plot(x, y, '-')
    ax1.set_ylim(-2, 2)
    ax1.set_title('Press left mouse button and drag to test')

    ax2.set(facecolor='#FFFFCC')
    line2, = ax2.plot(x, y, '-')


    def onselect(xmin, xmax):
        indmin, indmax = np.searchsorted(x, (xmin, xmax))
        indmax = min(len(x) - 1, indmax)

        thisx = x[indmin:indmax]
        thisy = y[indmin:indmax]
        line2.set_data(thisx, thisy)
        ax2.set_xlim(thisx[0], thisx[-1])
        ax2.set_ylim(thisy.min(), thisy.max())
        fig.canvas.draw()

    # set useblit True on gtkagg for enhanced performance
    span = SpanSelector(ax1, onselect, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red'))


    plt.show()


def load_reference(refname):
    import numpy as np

    data = np.load(refname)

    return(data['x'], data['y'])


def mean_diff(x):
    import numpy as np

    # x_diff = np.diff(x)
    # x_diff_1 = np.roll(x_diff, -1)

    z = ((np.roll(x, -1) - x) + (x - np.roll(x, 1))) / 2

    z = np.delete(z, 0)
    z = np.delete(z, -1)
    # z = np.insert(z, 0, 0)

    # print(x)
    # print(np.roll(x, 1))
    # print(np.roll(x, -1))
    # print(x_diff_1)
    # print(np.shape(x))

    # z = (x_diff_1 + x_diff)
    # z = np.ediff1d(x, to_begin=x[0], to_end=x[-1])
    # z = np.ediff1d(x)
    # print(np.shape(z))

    return(z)


def mean_diff2(y, x, x_axis=None, y_axis=None):
    import numpy as np

    # x_diff = np.diff(x)
    # x_diff_1 = np.roll(x_diff, -1)

    if x_axis == 0:
        z_x = (((np.roll(x, -1, axis=0) - x) + (x - np.roll(x, 1, axis=0)))) / 2
    elif x_axis == 1:
        z_x = (((np.roll(x, -1, axis=1) - x) + (x - np.roll(x, 1, axis=1)))) / 2
    else:
        z_x = (((np.roll(x, -1) - x) + (x - np.roll(x, 1)))) / 2

    if y_axis == 0:
        z_y = (((np.roll(y, -1, axis=0) - y) + (y - np.roll(y, 1, axis=0)))) / 2
    elif y_axis == 1:
        z_y = (((np.roll(y, -1, axis=1) - y) + (y - np.roll(y, 1, axis=1)))) / 2
    else:
        z_y = (((np.roll(y, -1) - y) + (y - np.roll(y, 1)))) / 2

    z = z_y / z_x

    z = np.delete(z, 0, axis=y_axis)
    z = np.delete(z, -1, axis=y_axis)

    # print(z_y)

    return(z)


def mean_diff3(y, x, x_axis=None, y_axis=None):
    import numpy as np

    # x_diff = np.diff(x)
    # x_diff_1 = np.roll(x_diff, -1)

    if x_axis == 0:
        z_x = (((((np.roll(x, -2, axis=0) - x) + (x - np.roll(x, 0, axis=0)))) / 2) +
               (((np.roll(x, 0, axis=0) - x) + (x - np.roll(x, 2, axis=0)))) / 2) / 2
    elif x_axis == 1:
        z_x = (((((np.roll(x, -2, axis=1) - x) + (x - np.roll(x, 0, axis=1)))) / 2) + 
               (((np.roll(x, 0, axis=1) - x) + (x - np.roll(x, 2, axis=1)))) / 2) / 2
    else:
        z_x = (((((np.roll(x, -2) - x) + (x - np.roll(x, 0)))) / 2) + 
               (((np.roll(x, 0) - x) + (x - np.roll(x, 2)))) / 2) / 2

    if y_axis == 0:
        z_y = (((((np.roll(y, -2, axis=0) - y) + (y - np.roll(y, 0, axis=0)))) / 2) + 
               (((np.roll(y, 0, axis=0) - y) + (y - np.roll(y, 2, axis=0)))) / 2) / 2
    elif y_axis == 1:
        z_y = (((((np.roll(y, -2, axis=1) - y) + (y - np.roll(y, 0, axis=1)))) / 2) +
               (((np.roll(y, 0, axis=1) - y) + (y - np.roll(y, 2, axis=1)))) / 2) / 2
    else:
        z_y = (((((np.roll(y, -2) - y) + (y - np.roll(y, 0)))) / 2) +
               (((np.roll(y, 0) - y) + (y - np.roll(y, 2)))) / 2) / 2

    z = z_y / z_x

    z = np.delete(z, 0, axis=y_axis)
    # z = np.delete(z, 0, axis=y_axis)
    z = np.delete(z, -1, axis=y_axis)
    # z = np.delete(z, -2, axis=y_axis)

    return(z)


def mean_diff_all(y, x, x_axis=None, y_axis=None, num=1):
    import numpy as np


    if num % 2 == 0:
        num -= 1
    # print(num)

    hn = np.int(num/2)
    if hn == 0:
        hn = 1
    # print(hn)
    # if x_axis == 0:
    #     z_x = np.zeros_like(x[:-1, :])
    # elif x_axis == 1:
    #     z_x = np.zeros_like(x[:, :-1])
    # else:
    #     z_x = np.zeros_like(x[:-1, :])

    # if y_axis == 0:
    #     z_y = np.zeros_like(y[:-1, :])
    # elif y_axis == 1:
    #     z_y = np.zeros_like(y[:, :-1])
    # else:
    #     z_y = np.zeros_like(y[:-1, :])


    # for i in np.arange(-hn, hn + 1, 1):
    #     print(i)
    #     if x_axis == 0:
    #         z_x = z_x + (np.roll(np.diff(x, axis=0), num, axis=0))
    #     elif x_axis == 1:
    #         z_x = z_x + (np.roll(np.diff(x, axis=1), num, axis=1))
    #     else:
    #         z_x = z_x + (np.roll(np.diff(x, axis=0), num))
    #     print(z_x)

    #     if y_axis == 0:
    #         z_y = z_y + (np.roll(np.diff(y, axis=0), num, axis=0))
    #     if y_axis == 1:
    #         z_y = z_y + (np.roll(np.diff(y, axis=1), num, axis=1))
    #     else:
    #         z_y = z_y + (np.roll(np.diff(y, axis=0), num))

    # print(np.roll(x, 0, axis=1))

    z_x = np.zeros_like(x[:, :])

    z_y = np.zeros_like(y[:, :])

    for i in np.arange(-hn, hn + 1, 1):
        # print(i)
        if x_axis == 0:
            z_x = z_x + (np.roll(x, hn - 1, axis=0) - (np.roll(x, hn + 1, axis=0)))# + \
                         # ((np.roll(x, hn, axis=0) - np.roll(x, hn + 2, axis=0))) / 2)
        elif x_axis == 1:
            z_x = z_x + (np.roll(x, hn - 1, axis=1) - (np.roll(x, hn + 1, axis=1)))# + \
                         # ((np.roll(x, hn, axis=1) - np.roll(x, hn + 2, axis=1))) / 2)
        else:
            z_x = z_x + (np.roll(x, hn - 1) - (np.roll(x, hn + 1)))# + \
                         # ((np.roll(x, hn) - np.roll(x, hn + 2))) / 2)
        # print(z_x)

        if y_axis == 0:
            z_y = z_y + (np.roll(y, hn - 1, axis=0) - (np.roll(y, hn + 1, axis=0)))# + \
                         # ((np.roll(y, hn, axis=0) - np.roll(y, hn + 2, axis=0))) / 2)
        if y_axis == 1:
            z_y = z_y + (np.roll(y, hn - 1, axis=1) - (np.roll(y, hn + 1, axis=1)))# + \
                         # ((np.roll(y, hn, axis=1) - np.roll(y, hn + 2, axis=1))) / 2)
        else:
            z_y = z_y + (np.roll(y, hn - 1) - (np.roll(y, hn + 1)))# + \
                         # ((np.roll(y, hn) - np.roll(y, hn + 2))) / 2)
    # if x_axis == 0:
    #     z_x = (np.roll(np.diff(x), -1, axis=0) + np.roll(np.diff(x), 1, axis=0)) / 2


    # if x_axis == 0:
    #     z_x = (np.roll(np.diff(y), -1, axis=0) + np.roll(np.diff(y), 1, axis=0)) / 2
    z = (z_y / (num - 1)) / (z_x / (num - 1))

    z = np.delete(z, hn, axis=y_axis)
    z = np.delete(z, -hn, axis=y_axis)
    # print(z_y / num)

    return(z)


# (a, b, c, d) = delta_cp_and_
# print(a,b,c(1400),c(2300),d)
# passt


# blub = data_to_function(plotit=False)

# print(blub(20))

# blub(0)(20)

# Testground for new functions

# import numpy as np
# bla = np.arange(0, 235 * 18).reshape((235, 18)).T
# # print(np.shape(bla))
# for c in range(np.shape(bla)[1]):
#     bla[:, c] = bla[:, c] + c * 102
# # print(np.shape(bla))
# print(bla)
# print('bla ende')
# # bla = np.arange(1,1000)
# # print(bla)
# blub = dt_to_dx(bla, 60, 150, True)

# import matplotlib.pyplot as plt
# plt.plot(blub[0], bla[0])
# plt.show()
# print(blub)
