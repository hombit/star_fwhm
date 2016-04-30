#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import os
import re
from sys import argv
from astropy.io import fits
from scipy import ndimage
from scipy.optimize import curve_fit
from time import sleep
from datetime import datetime, timedelta
from argparse import ArgumentParser


perimeter_width = 0.05 # part of image
sigma0_arcsec = 1
fwhm2sigma =  np.sqrt( 8 * np.log(2) )

#inner_radius = 10 
#outer_radius = 30
#preview_max_size = 50

data_t = []
fwhm_data_x = []
fwhm_data_y = []


def file_number(filename):
    return int(re.search(r'(\d+).\w+$', filename).groups()[0])

def mtime(filename):
    return os.path.getmtime( filename )

def ls_dir(directory, min_mtime=0):
    ls = [ os.path.join(directory, x) for x in os.listdir(directory) ]
    focus_files = list( filter(
        lambda x:
            mtime(x) > min_mtime and 'focus' in x,
        ls
    ) )
    focus_files.sort( key = mtime )
    return focus_files


def gauss(x, a, sigma, x0):
    return a * np.exp( - (x-x0)**2 / (2 * sigma**2) )


class FocusFiles():
    def __init__(self, directory, recent_only=False):
        self.data = []
        self.lastmtime = 0
        self.directory = directory
        self.recent_only = recent_only
    def renew(self):
        self.data = ls_dir(self.directory, min_mtime=self.lastmtime)
        if len(self.data) != 0:
            self.lastmtime = mtime( self.data[-1] )
            if self.recent_only:
                self.data = (self.data[-1],)
        return self.data
    

def data_gen(directory, scale, recent_only):
    sigma0 = sigma0_arcsec / scale
    focus_files = FocusFiles(directory, recent_only=recent_only)
    while True:
        focus_files.renew()
        while len(focus_files.data) == 0:
            focus_files.renew()
            sleep(0.1)
            yield
        for filename in focus_files.data:
            yield fits_handler( filename, sigma0 )


def fits_handler(filename, sigma0):
    with fits.open(filename) as f:
        for part in f:
            data = part.data
            if isinstance(data, np.ndarray):
                break
        else:
            raise RuntimeError('Cannot find image data in fitfile: {}'.format(filename))
        xsize = data.shape[0]
        ysize = data.shape[1]
        perimeter_xwidth = int(perimeter_width * xsize)
        perimeter_ywidth = int(perimeter_width * ysize)
        background = 0.25 * (data[0:perimeter_xwidth].mean() + data[-perimeter_xwidth:].mean() + data[perimeter_xwidth:-perimeter_xwidth][0:perimeter_ywidth].mean() +  data[perimeter_xwidth:-perimeter_xwidth][-perimeter_ywidth:].mean())
        data = data - background
        x_center, y_center = ( int(np.round(x)) for x in ndimage.center_of_mass(data) )
        if x_center >= xsize or x_center < 0:
            x_center = xsize/2
        if y_center >= ysize or y_center < 0:
            y_center = ysize/2
        try:
            param_x, _ = curve_fit(
                gauss,
                np.arange(xsize),
                data[:,y_center],
                p0 = ( data[x_center, y_center], sigma0, x_center )
            )
        except RuntimeError:
            param_x = (0, 0, 0)
        try:
            param_y, _ = curve_fit(
                gauss,
                np.arange(ysize),
                data[x_center,:],
                p0 = ( data[x_center, y_center], sigma0, y_center )
            )
        except RuntimeError:
            param_y = (0, 0, 0)
        sigma_x = min( xsize/3, abs(param_x[1]) )
        sigma_y = min( ysize/3, abs(param_y[1]) )
        fwhm_x = fwhm2sigma * sigma_x
        fwhm_y = fwhm2sigma * sigma_y

        #inner_square = data[x_center-inner_radius:x_center+inner_radius+1, y_center-inner_radius:y_center+inner_radius+1].sum()
        #outer_square = data[x_center-outer_radius:x_center+outer_radius+1, y_center-outer_radius:y_center+outer_radius+1].sum()
        #qual = inner_square/(outer_square-inner_square)

        reduce_ratio = 1
#        if max(xsize,ysize) > preview_max_size:
#            reduce_ratio = np.ceil(max(xsize, ysize) / preview_max_size)
        X, Y = np.meshgrid( np.arange(np.ceil(ysize/reduce_ratio)), np.arange(np.ceil(xsize/reduce_ratio)) )

        data[x_center,:] = 0
        data[:,y_center] = 0

    return mtime(filename), fwhm_x, fwhm_y, X, Y, data[::reduce_ratio,::reduce_ratio]



def run(d, scale, ax1, ax2, line1, line2, cont):
    if d == None:
        return [line1, line2]
    t, fwhm_x, fwhm_y, X, Y, data = d
    data_t.append(t)
    tmin, tmax = ax1.get_xlim()
    fwhm_data_x.append(fwhm_x * scale)
    fwhm_data_y.append(fwhm_y * scale)
    data_dt = []
    number_of_visible = 0
    for datum_t in data_t:
        data_dt.append( datum_t - t )
        if data_dt[-1] >= tmin:
            number_of_visible += 1
    ax1.set_ylim(
        min( min(fwhm_data_x[-number_of_visible:]), min(fwhm_data_y[-number_of_visible:]) ) * 0.9,
        max( max(fwhm_data_x[-number_of_visible:]), max(fwhm_data_y[-number_of_visible:]) ) * 1.2
    )
    line1.set_data(data_dt, fwhm_data_x)
    line2.set_data(data_dt, fwhm_data_y)
    ax2.cla()
    line2 = ax2.contourf( X, Y, data, 10 )
    return [line1, line2]



##########



def main():
    parser = ArgumentParser(description='''
        Draw star size and contour map from directory with FITS files.
        Files have to match expression 'focus*NUMBER.EXT', where NUMBER is integer and EXT is something like 'fits' or 'FIT'
    ''')
    parser.add_argument( 
       '-d', '--dir',
        dest='dir', action='store',
        default='.', 
        help='directory with FITS files (default is current directory)'
    )
    parser.add_argument(
        '-s', '--scale',
        dest='scale', action='store',
        default=1,
        help="scale of the image, arcsec  per pixel (default is 1''/pixel)"
    )
    parser.add_argument(
        '-r', '--recent-only',
        dest='recent_only', action='store_true',
        help="draw only most recent files, it could be useful when fits appears rapidly or directory is full of files that you wouldn't like to draw (default is false)"
    )
    args = parser.parse_args()
    directory = os.path.expanduser(args.dir)
    scale = float(args.scale)
    recent_only = args.recent_only

    fig, (ax1, ax2) = plt.subplots(1,2)
    line1, = ax1.plot( [],[], 'x', label='x FWHM' )
    line2, = ax1.plot( [],[], '*', label='y FWHM' )
    cont, = ax2.plot( [],[] )
    ax1.set_xlim(-120, 5)
    ax1.set_ylim(0.3, 5)
    ax1.grid()
    ax1.set_xlabel('Seconds ago')
    ax1.set_ylabel('FWHM, arcsec')
    ax1.legend( loc=2, borderaxespad=0. )
#    fig.autofmt_xdate()
    
    ani = anim.FuncAnimation(
        fig,
        run,
        frames=data_gen(directory, scale, recent_only),
        fargs=(scale, ax1, ax2, line1, line2, cont),
        repeat=False
    )
    plt.show()


if __name__ == "__main__":
    main()
