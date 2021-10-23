import random
import datetime
from typing import Any

from Inhertiance_Encapsulation_Polymorphism import SubClass

__all__ = [
  "Profiler"
]

class Profiler():

  def __init__(self: object) -> None:
    pass

  def test_gen(self: object, callback : Any, *args: tuple) -> None:
    try:
      print(str(callback) + '\t' + 'args = {0}'.format(args) + '\t' + str(callback(*args)), end='\n\n')
    except(TypeError):
      pass

  def test_gen(self, n , functor, *args):
    result = None
    time_array = []
    time_array.append( datetime.datetime.now().microsecond )
    input_args_str = str(args)
    obj = {} # for 'other' data from tests within test chain
    hide_data = False
    for i in range(n):
      time_array.append( datetime.datetime.now().microsecond )
      result = str(functor(*args, obj)) # call
      time_array[-1] = datetime.datetime.now().microsecond - time_array[-1]
      print('{4}\nFunctionName = {0} \nTimeElapsed = {3} ms\nInput(i.e. args) = {1}\n Result = {2} \n{4}'.format( str(obj.get('name')) , 'hidden' if hide_data else  input_args_str   ,'hidden' if hide_data else str(result), time_array[-1] / 10e3 , ("--"*100) + '\n' ))
    time_array[0] = datetime.datetime.now().microsecond - time_array[0]

  def test(self, N = 50):
    test_data = random.sample(range(0,10000), 10000)
    # self.test_gen(N, sum, [10, 1, 11])
    # self.test_gen(1,SubClass().sort_number_list, random.sample(range(0,1000), 1000), 3)
    # self.test_gen(1,SubClass().sort_number_list, test_data[: :] , 1)
    # self.test_gen(1,SubClass().sort_number_list, test_data[: :] , 2)
    # self.test_gen(1,SubClass().sort_number_list, test_data[: :], 3)
    # self.test_gen(1,SubClass().sort_number_list, [test_data[: :]], 4)
    # self.test_gen(1,SubClass().sort_number_list, [test_data[: :]], 5)
    self.test_gen(1,SubClass().sort_number_list, [7, 3, 2, 16, 24, 4, 11, 9], 5)
    self.test_gen(1,SubClass().sort_number_list, [7, 3, 2, 16, 24, 4, 11, 9], 6)



Profiler().test()