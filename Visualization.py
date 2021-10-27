from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs  as ccrs # cartopy coordinate reference system used for maps
import cartopy.feature as cfeature
plt.style.use('seaborn-dark-palette')

def plot_sin() -> None:
  # Basic sinc plot
  w = 2* np.pi
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
  w = 2* np.pi
  x_t = np.linspace(0, 1, 100)
  y_t = None
  plt.figure(figsize=(12,10))
  plt.style.use('seaborn-dark-palette')

  for i in list(range(1, 5)):

    plt.subplot(2, 3, i)
    y_t = np.sin(i * w * x_t)
    plt.title('Increase of w by {0}'.format(i))
    plt.plot(x_t, y_t)
    plt.xlabel('w/wo')
    plt.ylabel('Amplitude')
    plt.grid()
  plt.show()

# 3D Plot
def plot_spiral() -> None :
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
  x = np.random.random(20)
  y = np.random.random(20)
  z = np.random.random(20)

  fig  = plt.figure(figsize = (10, 10))
  ax = plt.axes(projection = '3d')
  ax.grid()

  ax.scatter(x, y, z, c = 'y', s= 500)
  ax.set_title('3D Scatter Plot')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  plt.show()

def mesh_grid() -> None :
  x = [1, 2, 3, 4]
  y = [12, 22]
  x_mesh, y_mesh = np.meshgrid(x,y)

  # x_mesh y_mesh are combinations of the two array(sets) (Xi, Yi)....combinations mapped to array x_mesh and y_mesh
  print(str(x_mesh) + ' \n' + str(y_mesh) )

# Maps
def map_projection(mode : int = 0, ) -> None :
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
  map_projection(3)

