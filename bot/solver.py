import random


class MontecarloSolver(object):
	def __init__(self, *args, **kwargs):
		random.seed(kwargs.get('seed', None))
		self.ranges = list(args)

	def _variate(self, params):
		for i in range(0, len(params)):
			params[i] = random.uniform(self.ranges[i][0],self.ranges[i][1])

	def solve(self, problem, max_error=0.01):
		i = 0
		error = float('inf')  # positive infinity
		params = [0.0] * len(self.ranges)

		while error >= max_error:
			i = i + 1
			self._variate(params)
			error = abs(problem(params))
			yield i, error, params


# class BinarySpacePartitionSolver(object):
# 	def __init__(self, *args):
# 		self.ranges = list(args)

# 	def _variate(self, params):
# 		pass

# 	def solve(self, problem, max_error=0.01):
# 		i = 0
# 		error = float('inf')
# 		params = [0.0] * len(self.ranges)

# 		while error >= max_error:
# 			i = i + 1
# 			self._variate(params)




if __name__ == '__main__':
	import math

	problem = lambda x: x[0]**2.0 - 4.0*x[0]*math.sin(x[0])*x[1] # Equation: x^2 - 4x*sin(x)*y = 0
	solver = MontecarloSolver([1.0, 100.0], [1.0, 100.0])
	solutions = solver.solve(problem, max_error=0.05)
	
	solution = None
	try:
		while True:
			solution = solutions.next()
			# print solution

	except StopIteration:
		print "Solution is %s on %s-th iteration with error %s" % (solution[2], solution[0], solution[1])
		pass
