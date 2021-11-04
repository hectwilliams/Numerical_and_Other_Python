

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
import time
import os
import random
import array

from typing import Iterable, List
import numpy as np
from matplotlib import pyplot

import lazy_logger


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

'''
  Processes can communicate data between one another
    Queue class

  The following example uses multiprocessing Queue class to share data between processes (seperate programs sharing data)

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

  print('\t child Process: {0}'.format(mp.current_process()))

  while not q.empty():
    print( '{0} got off bus'.format(q.get()) )

def ride_bus_queue_example() -> None :
  # Parent and child process share queue: adding riders to bus(parent) and removing (child)
  # The following example spawns one child at a time

  # Parameters:
  # None

  # Returns:
  # None

  seat_count = 10
  overload_factor = 100
  q = mp.JoinableQueue(seat_count)  # Queue
  passengers =  [i for i in range(overload_factor * 10)]
  child_process = None

  # main process adds riders
  while passengers.__len__() > 0:
    get_on_bus( q, passengers )

    # subprocess offloads riders
    child_process = mp.Process(target = get_off_bus, args=(q,))
    child_process.start()
    child_process.join() # worker empties queue

''' '''

def a_drive(q: mp.Queue) -> None:
  obj = q.get()
  obj['music'] = 'Country'
  obj['passengers'] = 2
  q.put(obj)

def b_drive(q: mp.Queue) -> None:
  obj = q.get()
  obj['music'] = 'Alternative'
  obj['passengers'] = 1
  q.put(obj)

def c_drive(q: mp.Queue) -> None:
  obj = q.get()
  obj['music'] = 'Rock'
  obj['passengers'] = 3
  q.put(obj)

def car_example() -> None :
  # Children share access to Parent's car

  # Parameters:
  # None

  # Returns:
  # None

  car = {
    'make': 'A3',
    'model': 'Audi',
    'year' : '2020',
    'music': '',
    'number_of_passengers': '',
  }

  q = mp.Queue(1)  # capacity = 1
  q.put(car)  # add car to queue

  print(car)

  process_a = mp.Process(target = a_drive, args=(q,))
  process_b = mp.Process(target = b_drive, args=(q,))
  process_c = mp.Process(target = c_drive, args=(q,))

  process_a.start()
  process_b.start()
  process_c.start()

  # wait for children to complete
  process_a.join()
  process_b.join()
  process_c.join()

  # state of car
  print(q.get())

''' '''

'''
  Processes can communicate data between one another
    Pipe class

  The following example uses multiprocessing Queue class to share data between processes (seperate programs sharing data)

'''
logger = lazy_logger.lazy_logger()

def pipe_example () -> None :
  msg = 'BobwThe Builder'
  ret = ''
  arr_buffer = np.array([0] * 10, dtype=bytes)
  producer, consumer = mp.Pipe()

  producer.send(msg)
  ret = consumer.recv()
  print('rcvd  :\t{0}'.format(ret))

  producer.send_bytes(msg.encode(), 1, 2) # encode to binary . offset, number of bytes read
  ret = consumer.recv_bytes(20) # rcv maximum length of 20 bytes
  print('rcvd truncated :\t {0}'.format(ret))
  print(ret) # b'ob The Builder

  producer.send_bytes(msg.encode(), 0, 4) # encode to binary . offset, number of bytes read
  if consumer.poll(): # poll connection for data
    ret = consumer.recv_bytes_into(arr_buffer)
    print('rcvd size:\t {0}'.format(ret))
    print(arr_buffer.view() )   # view
    logger.debug("Houston, we have a %s", "interesting problem")

pipe_example()
