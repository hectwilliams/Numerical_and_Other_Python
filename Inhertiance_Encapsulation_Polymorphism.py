import random
import numpy as np
from typing import List
from math import ceil

import documentation_ as doc

'''

 !! Never forget
 args(*) tuple inputs
 args(**) key/value inputs


  @ Object has support for an operation, but not yet implemented -> NotImplementedError
    def identity(data: object) -> Any:

      raise NotImplementedError

  @ Passing arguments of incorrect type -> TypeError
    Operation on an object is not supported

  @ Passing arguments with incorrect values -> ValueError
    test(string, string, int)
    expect(string, string,string)

'''

MESSAGE_LIST = [
  "Salutation",
  "Hello World",
  "Today is a good day"
]

__all__ = [
  "SuperClass",
  "SubClass"
]

class SuperClass():

  def __init__(self: object) -> None:
    self.class_name = "SuperClass Here"
    self.msg = 'I am {0} '.format(self.class_name)
    self.__testmessage = 'hello me' # private ... (python does not have private truly private variables)
    self.print(self.msg)

  def print(self: object, string: str) -> None:
    print("super-class message: {0}".format(string) )

  def mult (self, a, b) -> None:
    print(a*b)

  def sort_number_list(self, collection: List[int], option: int = 0) -> List[int]:
    raise NotImplementedError

class SubClass(SuperClass):

  def __init__(self: object) -> None:
    self.class_name = None
    self._location = 'Sunnyvale'
    # super().__init__() ### uncommenting inherits parent attributes: corresponding attributes are immutable

    if self.class_name == None:
      self.class_name = "SubClass Here"
      self.msg = 'I am {0} '.format(self.class_name)
      # self.print(self.msg)

  @doc.override
  def print(self: object, string : str) -> None:
    print("sub-class message: {0}".format(string))

  def accumulate(self, *args) -> int:
    sum_result = 0
    for number in args:
      sum_result = number + sum_result
    return sum_result

  def sort_number_list(self, collection: List[int], option: int, obj: object) -> List[int]:
    # selection sort method (liken to CASE STATEMENT )
    sort_func_selector = [
      bubble_sort,
      insertion_sort,
      selection_sort,
      quick_sort,
      merge_sort,
      max_heap,
      min_heap,
    ][option] # immediate execution

    if type(obj) is dict:
      obj['name'] = sort_func_selector.__name__

    sort_func_selector(collection)

    return collection

if __name__ == '__main__':
  obj_super_class = SuperClass()
  print(obj_super_class._SuperClass__testmessage)   ### attributes are not hidden (just mangled or hashed!)

  obj_sub_class = SubClass()
  obj_sub_class.mult(1,2) # inherited method
  print('print num of (1,2,3,4) : {0} '.format(obj_sub_class.accumulate(1,2,3,4)))
  obj_sub_class.accumulate(1,2,3,4)
  print(obj_sub_class._location)
else:
  # DO NOT EXECUTE OPERATIONS IF 'THIS' MODULE IS CALLED AFTER IMPORTING A PARENT MODULE
  pass

### Helper Functions

def bubble_sort (collection: List[int]) -> None:
  is_sorted = None
  swap_data = None
  for iteration in collection:
    is_sorted = True
    for j in range(collection.__len__() - 1):
      if (collection[j] > collection[j + 1] ):
        swap_data = collection[j+1]
        collection[j+1] = collection[j]
        collection[j] = swap_data
        is_sorted = False
    if is_sorted:
      break

def selection_sort (collection: List[int]) -> None:
  # ''every iteration search for minimum'
  min_index = 0
  tmp_data = None
  next_avail_min_slot = 0 # position in array where most recent mininum value is stored

  for next_avail_min_slot in range(collection.__len__()):
    min_index = next_avail_min_slot
    #search for mininimum
    for i in range(next_avail_min_slot, collection.__len__()):
      if collection[i] < collection[min_index]:
        min_index = i
    # new minimum found , update available min slot and shift array
    if min_index != next_avail_min_slot:
      #store latest minimum at min_index
      tmp_data = collection[min_index]
      # right shift (+1) array blocks in position 0 to min_index-1 (i.e. circular shift)
      for j in range(min_index, next_avail_min_slot, -1):
        collection[j] = collection[j - 1]
      # overwrite array position next_avail_min_slot with tmp_data
      collection[next_avail_min_slot] = tmp_data

def insertion_sort (collection: List[int]) -> None:
  # iterate and search sub-array for any values > than tail of sub-array
  tmp_data = None
  for end_pos in range(collection.__len__()):
    for i in range(0, end_pos):
      # value at ith position is less than end_pos value
      if collection[i] > collection[end_pos]:
        tmp_data = collection[end_pos]
        #circular right shift of subarray ith to endpos
        for j in range(end_pos, i, -1):
          collection[j] = collection[j - 1]
        collection[i] =  tmp_data
        break

def quick_sort (collection: List[int]) -> None:
  '''
    Recursion is replaced with table storing sub-arrray head and tail positions of nonpartitioned array slots
  '''
  table = []
  pivot = partition_pos = None
  sub_array_bound = None  # stores min max index of array e.g. [0 , 6]
  swap_pos = 0 # position of leftmost available array slot where values <= pivot can be swapped(i.e. moved to left most side of partition). At the end of array iteration, swap_pos -1 holds the parition position where the array is split into two sub_arrays
  tmp_data = None  # for swappable logic

  table.append([0, collection.__len__() - 1])   #init table
  while table.__len__() !=0:
    sub_array_bound = table.pop(0) # remove from head of array
    pivot = collection[sub_array_bound[1]] # pivot located at end of sub_array tail
    swap_pos = sub_array_bound [0] # lowest position is actually lowest sub-array bound
    for i in range(sub_array_bound[0], sub_array_bound[1] + 1):
      if collection[i] <= pivot:
        # swap
        tmp_data = collection[swap_pos]
        collection[swap_pos] = collection[i]
        collection[i] = tmp_data
        # swap position not updated on tail index
        if i != sub_array_bound[1]:
          swap_pos = swap_pos + 1  # move swap position marker forward
    # if left side exists, add min and max indices to table (refer to sub_array_bound )
    if swap_pos - 1 > sub_array_bound[0]:
      table.append([sub_array_bound[0], swap_pos - 1])
    # if right side exists, add to table
    if swap_pos + 1 < sub_array_bound[1]:
      table.append([swap_pos + 1 , sub_array_bound[1]])

def merge_sort (collection: List[int]) -> None:
  array_a = array_b = None
  mid = int(collection.__len__() / 2)
  pos_a = pos_b = None
  if mid == 0:
    return
  else:
    array_a = collection[0 : mid]
    array_b = collection[mid : :]
    merge_sort(array_a)
    merge_sort(array_b)
    pos_a = pos_b = 0
    for i in range(collection.__len__()):
      # overwrite input array with array b and a elements (added in order from least to greatest)
      if pos_b == array_b.__len__() :
        collection[i] = array_a[pos_a]
        pos_a += 1
      elif pos_a == array_a.__len__() :
        collection[i] = array_b[pos_b]
        pos_b += 1
      elif array_a[pos_a] <= array_b[pos_b]   :
        collection[i] = array_a[pos_a]
        pos_a += 1
      elif array_b[pos_b] <= array_a[pos_a] :
        collection[i] = array_b[pos_b]
        pos_b += 1

def swap(collection: List, n: int, m: int ):
  tmp = collection[n]
  collection[n] = collection[m]
  collection[m] = tmp

def heapfiy(collection: List[int] , block_count = 0, is_max_heapify: bool = True) -> None:
  current_node_pos = min_node_pos = ceil( collection.__len__() / 2 )
  child_node_left_pos = child_node_right_pos = None

  while (current_node_pos >=0) :
    # set min node to current node
    min_node_pos = current_node_pos
    #check left node for min
    child_node_left_pos  = 2 * current_node_pos + 1
    if child_node_left_pos < collection.__len__() - block_count:
      if collection[child_node_left_pos] < collection[min_node_pos] if is_max_heapify else  collection[child_node_left_pos] > collection[min_node_pos] :
        min_node_pos = child_node_left_pos
    #check right node for min
    child_node_right_pos  = 2 * current_node_pos + 2
    if child_node_right_pos < collection.__len__() - block_count:
      if collection[child_node_right_pos] < collection[min_node_pos] if is_max_heapify else collection[child_node_right_pos] > collection[min_node_pos] :
        min_node_pos = child_node_right_pos
    #swap if current_node(node position upon enternig loop)  does not match min_node
    if min_node_pos != current_node_pos:
      swap(collection, min_node_pos, current_node_pos)
    current_node_pos -= 1


def max_heap(collection: List):

  for i in range(collection.__len__() - 1):
    #heap
    heapfiy(collection, i)
    #swap zeroth element with next available array slot in tail (i.e. array_length - 1 - i)
    swap(collection, 0, collection.__len__() - 1 - i)

def min_heap(collection: List):

  for i in range(collection.__len__() - 1):
    #heap
    heapfiy(collection, i, False)
    #swap zeroth element with next available array slot in tail (i.e. array_length - 1 - i)
    swap(collection, 0, collection.__len__() - 1 - i)
