import random
import numpy as np
from typing import List

import documentation_ as doc

# !! Never forget
# args(*) tuple inputs
# args(**) key/value inputs

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
      selection_sort
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
  # partition logic

  #recursion

  pass


