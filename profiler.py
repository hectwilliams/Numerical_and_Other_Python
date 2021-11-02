import random
import datetime
import time
from typing import Any

from Inhertiance_Encapsulation_Polymorphism import SubClass, determinant

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
    hide_data = True
    start_time = end_time = 0
    for i in range(n):
      # time_array.append( datetime.datetime.now().microsecond )
      start_time =  time.time_ns()
      result = str(functor(*args, obj)) # call
      end_time  =  time.time_ns()
      print('{2}\nFunctionName = {0} \nTimeElapsed = {1} ms\n{2}'.format( str(obj.get('name')) , ( end_time - start_time) / 10e6 , ("--"*100) + '\n' ))

      # print('{4}\nFunctionName = {0} \nTimeElapsed = {3} ms\nInput(i.e. args) = {1}\n Result = {2} \n{4}'.format( str(obj.get('name')) , 'hidden' if hide_data else  input_args_str   ,'hidden' if hide_data else str(result), time_array[-1] / 10e3 , ("--"*100) + '\n' ))
    # time_array[0] = datetime.datetime.now().microsecond - time_array[0]

  def test_sorts(self, N = 50):
    test_data = random.sample(range(0,N*2), N*2)
    # self.test_gen(N, sum, [10, 1, 11])
    # self.test_gen(1,SubClass().op_list, random.sample(range(0,1000), 1000), 3)
    # self.test_gen(1,SubClass().op_list, test_data[: :] , 1)
    # self.test_gen(1,SubClass().op_list, test_data[: :] , 2)
    # self.test_gen(1,SubClass().op_list, test_data[: :], 3)
    # self.test_gen(1,SubClass().op_list, [test_data[: :]], 4)
    # self.test_gen(1,SubClass().op_list, [test_data[: :]], 5)
    # self.test_gen(1,SubClass().op_list, [7, 3, 2, 16, 24, 4, 11, 9], 5)
    # self.test_gen(1,SubClass().op_list, test_data[: :], 2)
    # self.test_gen(1,SubClass().op_list, test_data[: :], 3)
    self.test_gen(1,SubClass().op_list, test_data[: :], 4)
    # self.test_gen(1,SubClass().op_list, test_data[: :], 5)
    self.test_gen(1,SubClass().op_list, test_data[: :], 6)

  def test_matrix(self):

    test_list = [
      {
         'func_test_data': [
            [1,   3,       2],
            [-3, -1,      -3],
            [2,   3,       1]
          ],
        'func_test_expect': -15
      },

      {
         'func_test_data': [
            [6, -2],
            [3,  3]
          ],
        'func_test_expect': 24
      },

      {
         'func_test_data': [
            [-7, -10 , 4],
            [3, -9, 2],
            [7, 1, 2]

          ],
        'func_test_expect': 324
      },

           {
         'func_test_data': [
            [2, -1, 3, 0],
            [-3, 1, 0, 4],
            [-2, 1, 4, 1],
            [-1, 3, 0, - 2]
          ],
        'func_test_expect': -102
      },

       {
         'func_test_data': [
            [0, 1 , 2, 0],
            [1, 0, 3, 2],
            [2, -2, -5, -1],
            [3, -1, 1, 3]
          ],
        'func_test_expect': 9
      },

      {
         'func_test_data': [
            [ 5, -7, 2, 2],
            [ 0, 3, 0, -4],
            [-5, -8, 0, 3],
            [0, 5, 0, -6 ]
          ],
        'func_test_expect': 20
      },
     {
         'func_test_data': [
            [ 4 , 0, -7, 3, -5],
            [ 0, 0, 2, 0, 0],
            [7 , 3, -6, 4, -8 ],
            [ 5, 0, 5, 2, -3],
            [ 0, 0, 9, -1, 2 ]
          ],
        'func_test_expect': 6
      }

    ]

    for i,test_obj in enumerate(test_list):
      sol = ( determinant( test_obj.get('func_test_data')))
      expect = (test_obj.get('func_test_expect'))
      print (' {0}:\tcalculated {1}\t expect {2}\t Test passed: {3} \n'.format(i, sol,expect ,sol  ==  expect  )  )

Profiler().test_sorts()
