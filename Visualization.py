from typing import Callable

import cartopy.crs as ccrs  # cartopy coordinate reference system used for maps
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.function_base import linspace
import scipy.fftpack
from numpy.core.fromnumeric import take
from scipy.fftpack.pseudo_diffs import shift

''' :( 2D plots '''
def plot_sin() -> None:
  plt.style.use('seaborn-dark-palette')
  # Basic sinc plot
  w = 2 * np.pi
  x_t = np.linspace(0, 1, 100)
  y_t = np.sin(w * x_t)
  plt.plot( x_t, y_t, '*', label = 'sin wave')
  plt.xlabel('Radians (w/wo)')
  plt.ylabel('Amplitude')
  plt.show()

def plot_cos() -> None:
  # Basic sinc plot
  plt.style.use('seaborn-dark-palette')
  w = 2* np.pi
  x_t = np.linspace(0, 1, 100)
  y_t = np.cos(w * x_t)
  plt.plot( x_t, y_t, '*', label = 'sin wave')
  plt.xlabel('Radians (w/wo)')
  plt.ylabel('Amplitude')
  plt.show()

def plot_subplots() -> None :
  # Basic sinc subplots
  plt.style.use('seaborn-dark-palette')
  x_t = np.linspace(0, 1, 100)
  y_t = None
  plt.figure(figsize=(12,10))
  plt.style.use('seaborn-dark-palette')

  for freq in list(range(1, 5)):
    plt.subplot(2, 3, freq)
    y_t = np.sin(freq * 2* np.pi * x_t)
    plt.title('Increase of w by {0}'.format(freq))
    plt.plot(x_t, y_t)
    plt.xlabel('w/wo')
    plt.ylabel('Amplitude')
    plt.grid()
  plt.show()

def mesh_grid() -> None :
  # Calculate combinations of vector x and y mapped to vecotr's x_mesh and y_mesh

  # Parameters:
  #   None

  # Returns:
  #   None

  x = [1, 2, 3, 4]
  y = [12, 22]
  x_mesh, y_mesh = np.meshgrid(x,y) # x_mesh y_mesh are combinations of the two array(sets) (Xi, Yi)....combinations mapped to array x_mesh and y_mesh
  print(str(x_mesh) + ' \n' + str(y_mesh) )

def map_projection(mode : int = 0, ) -> None :
  # Plot map projections (mark map location of Stanford)

  # Parameters:
  #   None

  # Returns:
  #   None

  fig = plt.figure(figsize=(12, 6 )) # 12 inches width 4 inches height
  ax = None  # axes object

  if mode == 0:
    ax = plt.axes(projection=ccrs.PlateCarree())  # equirectangular projection

  if mode == 1 or mode == 3:
    ax = plt.axes(projection=ccrs.Mollweide())

  if mode == 2:
    zones = range(1, 61)
    for current_zone in zones:
      ax = fig.add_subplot( 1 , len(zones), current_zone, projection=ccrs.UTM(zone = current_zone, southern_hemisphere = True))
      ax.set_title(current_zone)
      ax.coastlines(resolution = '110m')

  if mode == 3:
    limits = [
      -135, # longitude min
      -60,  # longitude max
       10, # latitude min
       60   # longitude max
      ]
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(limits)  # zoom in on earth

  ax.gridlines(draw_labels=True)
  ax.add_feature(cfeature.LAND)
  ax.add_feature(cfeature.OCEAN)
  ax.add_feature(cfeature.BORDERS)
  ax.add_feature(cfeature.LAKES, alpha=0.7)
  ax.add_feature(cfeature.RIVERS)
  ax.add_feature(cfeature.STATES, linestyle='-')
  stanford_long, stanford_lat = -122.1661, 37.4241
  ax.set_extent([-122.8, -122, 37.3, 38.3]) # zoom map
  ax.plot(stanford_long, stanford_lat, color='red', linewidth=3, marker='o') # Standford  Marker
  ax.text(stanford_long+ 0.002, stanford_lat - 0.012, ' STANFORD') #  add text right (0.2) down(0.2)
  plt.show()

'''  :( ) 3D plots '''
def plot_spiral() -> None :
  # plot spiral to  3D plane

  # Parameters:
  #   None

  # Returns:
  #   None

  pi = np.pi
  wo = 2*pi
  ax = plt.axes(projection='3d')
  t = np.linspace(0, wo*pi, 100)
  x = np.sin (t)
  y = np.cos(t)
  ax.plot(x, y, t)
  ax.set_xlabel('X', labelpad=20)
  ax.set_ylabel('Y', labelpad=20)
  ax.set_zlabel('T', labelpad=20)

  plt.show()

def plot_random_dots() -> None :
   # plot random dots to 3D plane

  # Parameters:
  #   None

  # Returns:
  #   None

  x = np.random.random(20)
  y = np.random.random(20)
  z = np.random.random(20)
  ax = plt.axes(projection = '3d')
  ax.grid()
  ax.scatter(x, y, z, c = 'y', s= 500)
  ax.set_title('3D Scatter Plot')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.show()


''' DSP '''
def easy_dft_example() -> None:
  # Example of fast fourier transform

  # Parameters:
  # None

  # Returns:
  # None

  parameters = {}
  parameters['N'] = 500      # samples
  parameters['fs'] = 1000.0    # 1 kHz sampling
  parameters['Ts'] = 1.0 / parameters.get('fs') # 1ms sample time
  parameters['x_t'] = np.linspace(0, parameters.get('N') * parameters.get('Ts'), parameters.get('N') )
  parameters['freq_component'] = 2e2 # 200 Hz
  parameters['y_t'] = np.sin(  parameters['freq_component'] * 2 * np.pi * parameters.get('x_t') )
  parameters['y_f'] = scipy.fftpack.fft(parameters['y_t'])
  parameters['x_f'] = np.linspace(0.0001, parameters['fs'] / 2 , parameters['N'] // 2)

  plt.figure(figsize=(12,10))
  plt.subplot(1,2, 1)
  plt.plot(parameters['x_f'], np.abs( parameters['y_f'][ : int (parameters['N'] // 2)  ]) )
  plt.ylabel('strength')
  plt.xlabel('w')
  plt.subplot(1, 2, 2)
  plt.plot(parameters['x_f'], np.tan ( parameters['y_f'][ : parameters['N'] // 2 ] / parameters['x_f'] ))
  plt.ylabel('strength')
  plt.xlabel('w')
  plt.ylim(-0.5, 0.5)
  plt.show()

def ftt_center_origin(x: np.ndarray, y: np.ndarray):
  # Reflect discrete spectrum about origin. The output's sequence size is doubled.

  # Parameter
  #   x (1D ndarray):  frequency sample points. The sequence is half of the full width of fs/2
  #   y_f (1D ndarray):   frequency sampled spectrum. The seqence is half of the full width of fs/2

  # Returns:
  #   None

  x_out = None
  y_out = None
  mid = int( x.size/ 2)
  odd_offset = mid % 2

  x_out = np.zeros(x.size * 2)
  y_out = np.zeros(x.size * 2)

  for i in range (int(x.size / 2) ):
    x_out[i] = -i
    x_out[ i + mid +  odd_offset ] = i
    y_out[i] =  y[i]
    y_out[mid + odd_offset + i ] = y[i]

  return [x_out, y_out]

def autocorrelation_example() -> None:
  # Compute autocorrelation and plot the result

  # Parameters:
  #   None

  # Returns:
  #   None

  # square wave
  N = 1024*2
  T = 2
  ts = T / N
  x_t = np.arange(0, N, 1) # sequence of samplex indices
  x_ts = x_t * ts # sequence of sample times
  y_t = np.concatenate((np.zeros( int(N / 4) ), np.ones( int(N / 4) ) ,  np.zeros( int(N/2) )))
  arr_shiftable_y = np.zeros(y_t.size)
  y_time_corr = np.zeros(y_t.size )
  y_corr = np.zeros(y_t.size )

# add noise
  np.random.seed(19680801)
  noise = np.random.randn((x_t.size)) ** 2

  # convolution
  for i in range( y_t.size  ) :

    # Time Convolution #
    # right shift array by size
    for k in range(arr_shiftable_y.size - 1, 0, -1):
      arr_shiftable_y[k] = arr_shiftable_y[k - 1]
    # insert new value to front of array
    arr_shiftable_y[0] = y_t[i]
    # 1024 parralel multiply # correlation sum
    correl = sum (np.multiply(y_t, arr_shiftable_y))
    y_time_corr[i] = correl

    # Correlation #
    zeros_N = np.zeros(N)
    y_shifted_y_t =  np.concatenate( (zeros_N[0 :  i] , y_t[ 0: N - i  ]) )
    y_corr[i] = np.sum (  np.multiply(y_shifted_y_t, y_t) )

  # add noise
  y_t_noise = y_t * noise

  #plot
  fig, ax = plt.subplots(constrained_layout=True)

  plotRow = 4
  plotCols = 2

  plt.subplot(plotRow, plotCols,1)
  plt.ylabel('y_t')
  plt.xlabel('ts')
  plt.title('Src')
  plt.plot( x_t , y_t, '-o', marker='o', linestyle='--')

  plt.subplot(plotRow,plotCols,2)
  plt.ylabel('y_t')
  plt.xlabel('shift')
  plt.title('Spectrum')
  plt.xlim(0,int(N / 2))
  plt.semilogy(x_t , abs(scipy.fftpack.fft(y_t)), )

  plt.subplot(plotRow,plotCols,3)
  plt.ylabel('y_t')
  plt.xlabel('shift')
  plt.title('Src')
  plt.plot(x_t , y_t_noise, '-o', marker='o', linestyle='--')

  plt.subplot(plotRow,plotCols,4)
  plt.ylabel('y_t')
  plt.xlabel('shift')
  plt.title('Src + Noise Spectrum')
  plt.xlim(0,int(N / 2))
  plt.semilogy(x_t , abs(scipy.fftpack.fft(y_t_noise)), )
  plt.gca().twinx() # secondary y axis

  plt.subplot(plotRow,plotCols,5)
  plt.ylabel('y_t')
  plt.xlabel('shift')
  plt.title('Time Autocorrelation')
  plt.plot(x_t , y_time_corr, '-o', marker='o', linestyle='--')

  plt.subplot(plotRow,plotCols,6)
  plt.ylabel('y_t')
  plt.xlabel('shift')
  plt.title('Spectrum')
  plt.xlim(0,int(N / 2))
  plt.semilogy(x_t , abs(scipy.fftpack.fft(y_time_corr)), )

  plt.subplot(plotRow,plotCols,7)
  plt.ylabel('y_t')
  plt.xlabel('shift')
  plt.title(' Autocorrelation')
  plt.plot(x_t , y_corr, '-o', marker='o', linestyle='--')

  plt.subplot(plotRow,plotCols,8)
  plt.ylabel('y_t')
  plt.xlabel('shift')
  plt.title('Spectrum')
  plt.xlim(0,int(N / 2))
  plt.semilogy(x_t , abs(scipy.fftpack.fft(y_corr)), )

  fig.tight_layout(pad=1.0)
  plt.show()

def freq_shift_example() -> None:
  fig = plt.figure(figsize=(18, 8 )) # 12 inches width 4 inches height
  # Example of frequency shift
  N = 1024
  T = 2
  ts = T / N  # sampling interval
  fs = 1 / ts # sampling frequency
  fo = 1 # 1Hz component
  x_t = np.arange(0, N, 1) # sequence of samplex indices
  x_ts = x_t * ts # sequence of sample times
  y_t = np.cos(fo * 2 * np.pi * x_ts)
  y_f = scipy.fftpack.fft(y_t)
  [x_f_centered, y_f_centered ]= ftt_center_origin(x_t , y_f )

  fs_div_4 = fs / 4 # 1024/4 = 256
  carrier = np.cos(fs_div_4 * 2 * np.pi * x_ts)  # sampled carrier signal
  modulate_y_t = np.multiply(carrier, y_t) # carrier * signal
  modulate_y_f = scipy.fftpack.fft(modulate_y_t)
  modulate_x_f_centered, modulate_y_f_centered = ftt_center_origin(x_t , modulate_y_f)

  plt.grid()
  plt.subplot(3,2,1)
  plt.xlabel('')
  plt.ylabel('')
  plt.title(' 1 Symbol Per Second (N = 1024) ', fontsize=8)
  plt.plot(x_t, y_t, marker= 'x')

  plt.subplot(3,2,2)
  plt.ylabel('strength')

  plt.subplot(3,2,3)
  plt.ylabel('strength')
  plt.title('Frequency Magnitude 1 Hertz Signal', fontsize=8)
  plt.plot(x_t, np.abs(y_f) )

  plt.subplot(3,2,4)
  plt.ylabel('strength')
  plt.title('Frequency Magnitude Shifted 1 Hertz Signal' , fontsize=8)
  plt.plot(x_f_centered, np.abs(y_f_centered) )

  plt.subplot(3,2,5)
  plt.xlabel('f')
  plt.ylabel('strength')
  plt.title('Frequency Magnitude Shifted + Modulate', fontsize=8)
  plt.plot(x_t, np.abs(modulate_y_f) )

  plt.subplot(3, 2, 6)
  plt.xlabel('f')
  plt.ylabel('strength')
  plt.title('Frequency Magnitude Shifted + Modulate', fontsize=8)
  plt.plot(modulate_x_f_centered, np.abs(modulate_y_f_centered) )

  plt.show()

  return

if __name__ == '__main__':
  autocorrelation_example()

