from typing import Callable

import cartopy.crs as ccrs  # cartopy coordinate reference system used for maps
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
from numpy.core.fromnumeric import take

plt.style.use('seaborn-dark-palette')

''' # 2D plots '''
def plot_sin() -> None:
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
  w = 2* np.pi
  x_t = np.linspace(0, 1, 100)
  y_t = np.cos(w * x_t)
  plt.plot( x_t, y_t, '*', label = 'sin wave')
  plt.xlabel('Radians (w/wo)')
  plt.ylabel('Amplitude')
  plt.show()

def plot_subplots() -> None :
  # Basic sinc subplots
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
  parameters['y_t'] = np.sin(  parameters['freq_component'] * 2 * np.pi * parameters.get('x_t') )  # f * 2 * pi
  parameters['y_f'] = scipy.fftpack.fft(parameters['y_t'])
  parameters['x_f'] = np.linspace(0.0001, parameters['fs'] / 2 , parameters['N'] // 2)   # dc - fs / 2

  plt.figure(figsize=(12,10))
  plt.subplot(1,2, 1)
  plt.plot(parameters['x_f'], np.abs( parameters['y_f'][ : int (parameters['N'] // 2)  ]) )
  plt.ylabel('strength')
  plt.xlabel('w')
  plt.subplot(1, 2, 2)
  plt.plot(parameters['x_f'], np.tan ( parameters['y_f'][ : parameters['N'] // 2 ] / parameters['x_f'] ))  # integer divide //
  plt.ylabel('strength')
  plt.xlabel('w')
  plt.ylim(-0.5, 0.5)
  plt.show()

def gen_fft(n: int, ts: float, y_t: np.ndarray, x_t: np.ndarray) -> None:
  # generate fft using argument parameters

  # Parameters:
  #   n(int): number of samples
  #   ts(float): sample period
  #   y_t:  y-axis
  #   x_t : x-axis

  x_f = np.linspace(0.0001,  n * ts,  n )   #(int)(n / 2))
  y_f = scipy.fftpack.fft( y_t )

  x_f_centered,y_f_centered = ftt_center_origin(x_f , y_f )

  plt.subplot(2,2,1)
  plt.ylabel('y_t')
  plt.xlabel('ts')
  plt.title('Autocorrelation Data Sequence')
  plt.plot(x_t, y_t, '-o', marker='o', linestyle='--')

  plt.subplot(2,2,2)
  plt.ylabel('y_t')
  plt.xlabel('fs')
  plt.title('Frequency Response')
  plt.plot(x_f_centered, y_f_centered, '-o', marker='o', linestyle='--')

  plt.subplot(2,2,3)
  plt.ylabel('Phase')
  plt.xlabel('fs')
  plt.title('Phase Response')
  plt.ylim(-20,20)
  plt.plot( x_f_centered, np.tan(y_f_centered/ x_f_centered ) , '-o', marker='o', linestyle='--')

  plt.show()

def ftt_center_origin(x: np.ndarray, y: np.ndarray, inplace: bool = False):
  # partitions the x and y samples in half (folding frequency) and transforms plot with the origin in the center of the x-axis

  # Parameter
  #   x_f (1D ndarray):    Frequency domain samples
  #   y_f (1D ndarray):    Amplitude or component strength

  # Returns:
  #   None
  #     y = [ABCDDCBA]  -> 8 symbols
  #     y[7]  ->  y[3]
  #     y[6]  ->  y[2]
  #     y[5]  ->  y[1]
  #     y[4]  ->  y[0]
  #     y = [DCBAABCD]
  x_out = None
  y_out = None
  mid = int( x.size / 2)
  odd_offset = mid % 2

  if inplace:
    for i in range(int(x.size / 2) ):

      # cross frequency axis to send the frequency
      x[i] =  i - mid  # left
      x[mid + odd_offset  + i] =  i # right

      # cross component 'strength' as well
      tmp_y = y[mid + odd_offset + i ]
      y[mid + odd_offset + i ] = y[i]
      y[i] = tmp_y
      return x,y
  else:
    x_out = np.zeros(x.size)
    y_out = np.zeros(x.size)
    for i in range(int(x.size / 2)):
      x_out[i] = i - mid
      x_out[mid + odd_offset  + i] = i

      y_out[mid + odd_offset + i ] = y[i]
      y_out[i] = y[i - mid]

    return x_out, y_out

def wave_gen (mode: int = 0, N : int = 1024) -> dict:
  # Generate sample waveform: swtiched by mode argument

  # Parameters:s
  #   mode(int): selectable waveform

  # Returns:
  #   {y: ndarray-type, x: ndarray-type} (dict) : 1024 sample waveform

  param = {}
  param['N'] =  1024

  if mode == 0:
    # square wave
    param['x_out'] = np.linspace(0.001, param.get('N'), param.get('N'))
    param['y_out'] = np.concatenate((np.zeros( 256), np.ones( 256) ,  np.zeros( 512 )))

  if mode == 1:
    # sine wave
    param['fo'] = 8e2   # 800_Hz
    param['fs'] = 2e3   # 2_kHz
    param['Ts'] = 1.0 / param.get('fs') # 2_ms sample time
    param['y_t'] = np.sin(  param['fo'] * 2 * np.pi * param.get('x_t') )  # f * 2 * pi
    param['x_t'] = np.linspace(0, param.get('N') * param.get('Ts'), param.get('N') )

    param['x_out'] = np.linspace(0.0001, param['fs'] / 2 , param['N'] // 2)   # dc to fs / 2
    param['y_out'] = scipy.fftpack.fft(param['y_t'])


  return {
    'x': param.get('x_out'),
    'y': param.get('y_out')
  }

def easy_auto() -> None:
  # Compute autocorrelation and plot the result

  # Parameters:
  #   None

  # Returns:
  #   None

  y_corr = None
  data = wave_gen() # returns { x: x-axis samples, y: y-axis samples}
  y_corr = np.zeros(data.get('x').size )

  # convolution
  for i in range( data.get('x').size  ) :
    # shift vector
    logical_right_shift_by_i = np.concatenate(  (np.zeros(i),data.get('y')[0 : data.get('y').size - i]))
    # 1024 parralel multiply
    # correlation sum
    correl = sum (np.multiply(data.get('y'), logical_right_shift_by_i))
    y_corr[i] = correl

  # plt.subplot(2,1, 1)
  # plt.ylabel('Amplitude')
  # plt.xlabel('Ts')
  # plt.title('Unit')
  # plt.grid()

  # plt.plot( data.get('x'),  data.get('y') , '-x')
  # plt.subplot(2,1, 2)
  # plt.ylabel('Amplitude')
  # plt.xlabel('Ts')
  # plt.title('correlation: auto')
  # plt.plot( data.get('x'), (y_corr -y_corr.min()) / (y_corr.max() - y_corr.min())  , '-x', marker='o')
  # plt.grid()
  # plt.show()

  gen_fft(n = y_corr.size, ts= (data.get('x')[1] - data.get('x')[0]) , y_t = y_corr, x_t = data.get('x'))

'''  # 3D plots '''
def plot_spiral() -> None :
  # plot spiral to  3D plane

  # Parameters:
  #   None

  # Returns:
  #   None
  pi = np.pi
  wo = 2*pi
  fig = plt.figure(figsize = (10,10))
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

def mesh_grid() -> None :
  # Calculate combinations of vector x and y mapped to vecotr's x_mesh and y_mesh

  # Parameters:
  #   None

  # Returns:
  #   None

  x = [1, 2, 3, 4]
  y = [12, 22]
  x_mesh, y_mesh = np.meshgrid(x,y)

  # x_mesh y_mesh are combinations of the two array(sets) (Xi, Yi)....combinations mapped to array x_mesh and y_mesh
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
      # ax = fig.add_subplot( 1 , len(zones), current_zone, projection=ccrs.UTM(zone = current_zone, southern_hemisphere = True)
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

  # plot figure(s)

  # ax.stock_img() # debug only
  ax.gridlines(draw_labels=True)
  ax.add_feature(cfeature.LAND)
  ax.add_feature(cfeature.OCEAN)
  ax.add_feature(cfeature.BORDERS)
  ax.add_feature(cfeature.LAKES, alpha=0.7)
  ax.add_feature(cfeature.RIVERS)
  ax.add_feature(cfeature.STATES, linestyle='-')
  stanford_long, stanford_lat = -122.1661, 37.4241
  ax.set_extent([-122.8, -122, 37.3, 38.3]) # zoom map
  ax.plot(stanford_long, stanford_lat, color='red', linewidth=3, marker='o') # Standford University Marker
  ax.text(stanford_long+ 0.002, stanford_lat - 0.012, ' STANFORD') #  add text right (0.2) down(0.2)
  plt.show()


if __name__ == '__main__':
  easy_auto()

