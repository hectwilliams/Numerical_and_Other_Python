

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
import threading as thrd
import multiprocessing as mp
import time
import os
import random
import array

from typing import Iterable, Iterator, List
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

def thread_hello_0(*args: tuple, **kwargs: object):
  iterations = 4
  for i in range(iterations):
    time.sleep(2.4)
    print("\thi my name is magneto", args , kwargs)

def thread_hello_1(*args: tuple, **kwargs: object):
  iterations = 4
  for i in range(iterations):
    time.sleep(1.0)
    print("\thi my name is wolverine", args , kwargs)

'''
  Threads live within processes. Mutiple threads can run per process sharing resources to execute tasks
'''
def thread_example():
  # basic example of threads: two functions competing for resources in order print hello and their name

  # Parameters:
  # None

  # Returns:
  # None

  # thread instance
  thr_1 = thrd.Thread(target=thread_hello_0, name='thread_0', args=(1929,11), kwargs={'nickname': 'Cyclops'})
  thr_2 = thrd.Thread(target=thread_hello_1, name='thread_1', args=(1600,33), kwargs={'nickname': 'Raiden'})

  # run thread (Automatically invokes run() method)
  thr_1.start()
  thr_2.start()

  print('List of threads:')
  for i in thrd.enumerate():
    print(str(i))
  print('Child threads competing for resources (preemptive)')

  # block until both threads have completing executing statement
  thr_1.join()
  thr_2.join()

def morning_routine_bro(*args: tuple, **kwargs: object)  -> None:
  #  morning routine using lock object to update shared object

  # Parameters:
  # args (tuple) - 2 element array storing lock object and shared data object
  # kwargs (dict) - Key/Value data

  # Returns:
  # None

  lck = args[0]
  data = args[1]
  counter = 0

  lck.acquire()
  data['shower_in_use'] = True
  data['shower_user'] = 'Bro'
  print( "Bro: using bathroom...locking the bathroom door".format(kwargs.get('who_am_i')) )
  time.sleep(2)
  print( "Bro: going in the shower...unlocking the bathroom door".format(kwargs.get('who_am_i')) )

  lck.release()

  time.sleep(2)

  # wait (eventually some will enter bathroom)
  while not lck.locked() and counter != 1000:
    time.sleep(1)
    print('Bro: yay, bathroom is free for me')

  # sis is using sink
  while lck.locked():
    time.sleep(1)
    print('Bro: I\'m using the shower, get out')

  # bro is leaving shower ...update shared object
  lck.acquire()
  data['shower_in_use'] = False
  data['shower_user'] = ''
  data['recent_shower_user'] = 'Bro'
  print('Bro: leaving bathroom')
  print(data)
  lck.release()

def morning_routine_sis(*args: tuple, **kwargs: object) -> None:
  #  morning routine using lock object to updated shared object

  # Parameters:
  # args (tuple) - 2 element array storing lock object and shared data object
  # kwargs (dict) - Key/Value data
  # Returns:
  # None

  # lock instance
  lck = args[0]
  data = args[1]

  while lck.locked():
    time.sleep(1)
    print("Sis: I need to brush my teeth, unlock the door")

  # wait until bro enters shower and unlocks
  while not data['shower_in_use'] or lck.locked():
    pass

  # unlocked
  lck.acquire()
  print('Sis: Brushing teeth now')
  data['sink_in_use'] = True
  data['sink_user'] = 'Sis'
  data['recent_sink_user'] = ''
  print(data)
  time.sleep(2)
  data['sink_in_use'] = False
  data['sink_user'] = ''
  data['recent_sink_user'] = 'Sis'
  print('Sis: leaving bathroom')
  print(data)
  lck.release()


def thread_example_morning_routine():
  # this example illustrates locking objects using two siblings access to bathroom

  # Parameters:
  # None

  # Returns:
  # None

  # lock instance
  lck = thrd.Lock()

  shared_obj= {
    'name': 'bathroom',
    'shower_in_use': 0,
    'shower_user': '',
    'recent_shower_user': '',
    'sink_in_use': 0,
    'sink_user': '',
    'recent_sink_user': ''
  }

  # thread instance(s)
  bro = thrd.Thread(target=morning_routine_bro,args=(lck,shared_obj), kwargs={'who_am_i': 'bro'})
  sis = thrd.Thread(target=morning_routine_sis, args=(lck,shared_obj), kwargs={'who_am_i': 'sis'})

  bro.start()
  sis.start()

def police_call(*args, **kwargs):
  # police call thread which updates object whether criminals have been found

  # Parameters:
  # args (tuple (cond_lock, house_object))

  # Returns:
  # None

  cond_lck = args[0]
  house = args[1]

  with cond_lck:
    cond_lck.notify()
    # randomly flag whether intruders were caught
    house['was_arrested']  = True if(random.random() > 0.5) else False

def house(*args, **kwargs):
  # house thread uses conditional lock to process homeowner experience after home is vandalized

  # Parameters:
  # args (tuple (cond_lock, house_object))
  # kwargs (dictinary) empty

  # Returns:
  # None

  cond_lck = args[0]
  house = args[1]
  police = None

  with cond_lck:
    print('HOUSE-OWNER: Going for a long walk')
    cond_lck.wait()
    if house.get('removed_cnt') > 0:
      print('HOUSE-OWNER: someone robbed my home!')
      if house['saw_something'] :
        print('HOUSE-OWNER: I think I see someone is still there !')
        cond_lck.wait() # thief 2 thread should call notify after wait, else the system will pause!!
        print('HOUSE-OWNER: Calling 911')
        police = thrd.Thread(target=police_call, args=args, kwargs=kwargs)  # yay rlock !
        police.start()
        cond_lck.wait()
        police.join() # must wait for thread to finish

        if house['was_arrested']:
          print('HOUSE-OWNER: They have captured the thieves ')
        else:
          print('HOUSE-OWNER: They got away ')


def thief_01(*args, **kwargs):
  # thief thread uses conditional lock loot/damage house and notify owner

  # Parameters
  # None

  # Returns
  # None

  cond_lck = args[0]
  house = args[1]

# conditional thread lock instance
  with cond_lck:
    # Require conditional lock.
    # Note: contextual 'with' block calls require upon entering block. Conditional lock is not freed until thread calls notify()
    cond_lck.notify() # release conditional lock from wait (i.e. wake up sleeping thread)
    for i in range(2):
      house['removed_cnt'] = house.get('removed_cnt') + 1
      print('THIEF_01: stole items from house')
    print('THIEF_01: escaped to car')
    # randomly flag if other criminal can be visible
    house['saw_something']  = True if(random.random() > 0.5) else False
  # release conditional lock

def thief_02(*args, **kwargs):
  # thief thread uses conditional lock loot/damage house and notify owner

  # Parameters:
  # args (tuple (cond_lock, house_object))
  # kwargs (dictinary) empty

  # Returns:
  # None

  cond_lck = args[0]
  house = args[1]
  counter = 0

  with cond_lck:
    cond_lck.notify() # release conditional lock from wait (i.e. wake up sleeping thread)
    for i in range(3):
      house['removed_cnt'] = house.get('removed_cnt') + 1
      print('THIEF_02: stole items from house')
    print('THIEF_02: escaped to car')

def cond_example() -> None:
  # example using conditional lock

  # Parameters
  # None

  # Returns
  # None

  house_obj = {
    'removed_cnt': 0,   # items stolen
    'saw_something': False  # house owner see someone
  }
  lck = thrd.Condition()
  house_ = thrd.Thread(target=house, args=(lck, house_obj), kwargs={})
  person_01 = thrd.Thread(target=thief_01, args=(lck, house_obj), kwargs={})
  person_02 = thrd.Thread(target=thief_02, args=(lck, house_obj), kwargs={})

  house_.start()
  person_01.start()
  person_02.start()
  person_02.join()

def sem_producer(*args):
  mutex = args[0]
  data = args[1]

  print()
  while 1:
    mutex.acquire()
    data['num_of_elements'] = data.get('num_of_elements') + random.randint(1,10)
    mutex.release()
    print('\t producer adds element:  {0} '.format( data.get('num_of_elements') ))
    time.sleep(1)

def sem_consumer_01(*args):
  mutex = args[0]
  data = args[1]

  while 1:
    mutex.acquire()
    if data.get('num_of_elements') != 0:
      data['num_of_elements'] = data.get('num_of_elements') - 1
    mutex.release()
    print('\t 1 consumes element: {0} '.format( data.get('num_of_elements') ))
    time.sleep(1)

def sem_consumer_02(*args):
  mutex = args[0]
  data = args[1]

  while 1:
    mutex.acquire()
    if data.get('num_of_elements') != 0:
      data['num_of_elements'] = data.get('num_of_elements') - 1
    mutex.release()
    print('\t 2 consumes element:  {0} '.format( data.get('num_of_elements') ))
    time.sleep(1)

def semaphore_example() -> None:
  # semaphore example

  # Parameters
  # None

  # Returns
  # None

  sem = thrd.Semaphore()
  obj = {'num_of_elements': 0}
  thrd.Thread(target=sem_producer, args=(sem, obj)).start()
  thrd.Thread(target=sem_consumer_01, args=(sem, obj)).start()
  thrd.Thread(target=sem_consumer_02, args=(sem, obj)).start()


semaphore_example()

