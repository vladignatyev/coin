#!coding: utf-8


##	
#	Abstract classes
##

class Node(object):
	def __init__(self):
		self.edges = []


class Edge(object):
	def __init__(self, input_node, output_node):
		self.input_node = input_node
		self.output_node = output_node


class Network(object):
	def __init__(self):
		pass  # for future


class Evaluator(object):
	def __init__(self, trained_network):
		self.network = trained_network

	def evaluate(self, *args):
		pass


##	
#	Implementations
##

class FuncNode(Node):
	def __init__(self):
		super(FuncNode, self).__init__()
		self.func = lambda edges_values: sum([v for v in edges_values])  # simple summator


class ValueFuncNodeEvaluator(Evaluator):
	def __init__(self, trained_network):
		super(ValueFuncNodeEvaluator, self).__init__(trained_network)

	def evaluate(self, *args):
		input_values = {}
		nodes_inputs = {}
		outputs_values = {}

		# Match input values by nodes
		for i in range(0, len(self.network.inputs)):
			input_node = self.network.inputs[i]
			value = args[i]
			input_values[input_node] = value

		# Initialize array for values obtained from inputs to the current output
		for output_node in self.network.outputs:
			nodes_inputs[output_node] = []

		# fill up array of inputs for each output node
		for edge in self.network.edges:
			nodes_inputs[edge.output_node].append(input_values[edge.input_node])

		# evaluate func in output node with provided inputs
		for output_node in nodes_inputs:
			outputs_values[output_node] = output_node.func(nodes_inputs[output_node])

		# reduce results to match given order
		results = []
		for output in self.network.outputs:
			results.append(outputs_values[output])

		return results


class FullyConnectedLayer(Network):  # Layer
	def __init__(self, inputs_count, outputs_count, input_node_cls=Node, node_cls=FuncNode, edge_cls=Edge):
		super(FullyConnectedLayer, self).__init__()

		self._edges = []
		self._inputs = []
		self._outputs = []

		for i in range(0, outputs_count):
			self._outputs.append(node_cls())
		
		for i in range(0, inputs_count):
			input_node = input_node_cls()  # create
			self._inputs.append(input_node)

			for output_node in self._outputs:  # connect
				self._edges.append(edge_cls(input_node, output_node))

	@property
	def edges(self):
		return self._edges

	@property
	def inputs(self):
	  return self._inputs
	
	@property
	def outputs(self):
	  return self._outputs


if __name__ == '__main__':
	trained_network = FullyConnectedLayer(2, 6)
	layer = ValueFuncNodeEvaluator(trained_network)

	outputs = layer.evaluate(0.2, 0.3)
	print outputs
	# assert output1 == output2 == 0.0
