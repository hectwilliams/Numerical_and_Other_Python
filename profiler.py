import random
import datetime
from typing import Any

from Inhertiance_Encapsulation_Polymorphism import SubClass

'''

  @ Object has support for an operation, but not yet implemented -> NotImplementedError
    def identity(data: object) -> Any:

      raise NotImplementedError

  @ Passing arguments of incorrect type -> TypeError
    Operation on an object is not supported

  @ Passing arguments with incorrect values -> ValueError
    test(string, string, int)
    expect(string, string,string)
'''

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

    for i in range(n):
      time_array.append( datetime.datetime.now().microsecond )
      result = str(functor(*args, obj)) # call
      time_array[-1] = datetime.datetime.now().microsecond - time_array[-1]
      print('\nFunctionName = {0}\nInput(i.e. args) = {1}\nResult = {2}\nTimeElapsed = {3} ms \n'.format( str(obj.get('name')) , input_args_str , str(result), time_array[-1] / 10e3 ))
    time_array[0] = datetime.datetime.now().microsecond - time_array[0]

  def test(self, N = 50):
    # self.test_gen(N, sum, [10, 1, 11])
    self.test_gen(1,SubClass().sort_number_list, random.sample(range(0,1000), 1000), 2)
    self.test_gen(1,SubClass().sort_number_list, random.sample(range(0,1000), 1000), 1)
    self.test_gen(1,SubClass().sort_number_list, [10,80,30,90,40,50,70], 3)

Profiler().test()