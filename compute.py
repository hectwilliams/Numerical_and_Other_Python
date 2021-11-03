

'''
Parallel Basics

* physical core is a thread
* 'logical' core is threads created by abtraction of higher number of threads within same package

  Multi-Processor (i.g.  coprocessor)
    [processor 1 - single core ]
    [processor 2 - single core ]

  Multi-core (i.g. dual core )
    [
      [core]
      [core]
    ]

  Multi-core x2 (hardware abstraction)
    [
      [core] = [
        [sub_core]
        [sub_core]
      ]
      [core] = [
        [sub_core]
        [sub_core]
      ]
    ]

  Controlled by scheduler (kernel mode)
    pre-preemptive - no wait states , strict time window per instruction or command
    non-preemptive - wait states

'''

'''
Process Thread

  Scheduler
    [program instance on memory block ] -> process A
    [program instance on memory block ] -> process B
    [program innstane  on memory block ] -> process C

  Memory block (RAM blocks)
    [ .data | .bss | .heap --> | <-- .stack ]

  Mutilple cores(i.e. threads) can read/write from memory memory blocks (shared access)

  Process (thread shared access during the lifetime of this process):
  [
    thread 1
    thread 2
    ...
    ..
    .
  ]

  Processes (independent program)
    Program instance A
    Program instance B

    Both are instances pulled from static memory or FLASH

    Either change in A or B does not influence another program instance like program instance C
'''

'''
  Python Global Interpreter Lock (GIL)
    Prevents multiple threads from running simultaneously
      Note: OS wants to multi-thread. Python was designed way in the past during the first single core processors
'''

'''
  Disadvantages
    spawn new processes takes time
    maintenance of spawned processes
'''

import multiprocessing as mp
import os
import time
from typing import Iterable, List
import numpy as np
from matplotlib import pyplot
import random

def identity (value):
  # returns the input (i.e. passthrough)

  # Parameters:
  #   value (any)

  # Returns
  #   value (any)

  return value

def simple_example_map() -> None:
  #  Initializes workers pool and executes N(i.e. NUM_OF_SAMPLES) number of tasks

  # Parameters:
  #   None

  # Returns
  #   None

  NUM_OF_SAMPLES = 100
  pool = mp.Pool (processes = mp.cpu_count())
  pool.map(identity, list(range( int(NUM_OF_SAMPLES) )) ) # workers threads operate on function(i.e. identity)

def plot_simple_example_map_perf(N):
  #  Run preceding function N number of times, record the time to complete, and plot the time results (n vs time)

  # Parameters:
  #   N (int): Iteration count

  # Returns
  #   None

  pyplot.figure(figsize=(15,10))
  x = []
  y = []

  for i in range(N):
    t_start = time.time()
    simple_example_map()
    y.append(time.time() - t_start)
    x.append(i)

  # cast list into np type array
  y = np.array(y)
  x = np.array(x)

  # normal plot
  pyplot.subplot(2, 2, 1)
  pyplot.xlabel('iteration')
  pyplot.ylabel('sec')
  pyplot.title('Execution Time')
  pyplot.grid()
  pyplot.plot( np.array(x), np.array(y), '-o', label = 'execution time')
  pyplot.plot( np.array(x), np.average(y) * (np.divide(y,y)) , '-o', label = 'average') # label for key
  pyplot.legend()

  # normalize
  ymin = np.min(y)
  ymax = np.max(y)
  y_normalized =  (y - ymin) / (ymax - ymin)
  pyplot.subplot(2, 2, 2)
  pyplot.xlabel('iteration')
  pyplot.ylabel('sec')
  pyplot.title('Execution Time Normalized')
  pyplot.grid()
  pyplot.plot( np.array(x), y_normalized ,'-o', label = 'execution time')
  pyplot.legend()

  # decibel
  pyplot.subplot(2, 2, 3)
  pyplot.xlabel('iteration')
  pyplot.ylabel('dB')
  pyplot.title('Execution: Log Plot')
  pyplot.grid()
  pyplot.plot( np.array(x), 10 * np.log10(y_normalized) , '-o' , label = 'execution time' )
  pyplot.legend()

  pyplot.show()


'''Processes can communicate data between one another
    Queue class

  The following example uses multiprocessing Queue class to share data between processes
'''

def get_on_bus(q: mp.Queue, ids: List) -> None :
  # Load bus with people(ids) waiting

  # Parameters:
  #   q (Queue): container
  #   ids : list of people symbolized by ID number

  # Returns
  #   None

  id = None

  while not q.full():
    id = ids.pop()
    q.put(id)
    print('{0} entered bus\t waitline {1}'.format(id, str(ids) ) )

def get_off_bus(q: mp.Queue) -> None :
  # Empty bus if at least one person is on the bus

  # Parameters:
  #   q (Queue): container shared amongst processes

  # Returns
  #   None

  while not q.empty():
    print( '{0} got off bus'.format(q.get()) )

def ride_bus_queue_example():
  # processes share queue adding riders to bus

  # Parameters:
  # None

  # Returns:
  # None

  seat_count = 10
  q = mp.Queue(seat_count) # data

  p1 = mp.Process(target = get_on_bus, args=(q,[i for i in range(seat_count)],)) # comma at the end
  p1.start()

  p2 = mp.Process(target = get_off_bus, args=(q,))
  p2.start()

ride_bus_queue_example()