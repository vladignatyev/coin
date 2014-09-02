import csv

from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.datasets import SupervisedDataSet

from bot.training import days


def load_all_ticks():
	ticks = []
	data = days('btceUSD.days.csv')

	try:
		while True:
			price, volume, date = data.next()
			ticks.append((price, volume))
	except StopIteration:
		return ticks


def sliding_window(size, data_source):
	i = 0
	for i in range(0, len(data_source) - size):
		window = []
		for j in range(0, size):
			window.append(data_source[i+j])

		yield window

def normalize(v, _max, _min):
    return 2.0 / (_max - _min) * (v - _min) - 1.0


def denormalize(v, _max, _min):
    return (v + 1.0) / 2.0 * (_max - _min) + _min

def normalizer(window, normalize_func):
	first_sample = window[0]
	dimension = len(first_sample)
	result = [[0.0]*dimension] * len(window)

	mins = [float('inf')] * dimension
	maxs = [-float('inf')] * dimension

	for sample in window:
		for i in range(0, dimension):
			mins[i] = min(mins[i], sample[i])
			maxs[i] = max(maxs[i], sample[i])


	for i in range(0, len(window)):
		for j in range(0, dimension):
			result[i][j] = normalize_func(window[i][j], maxs[j], mins[j])

	return result, mins, maxs


def normalizeWindow(window):
	return normalizer(window, normalize)

def denormalizeVector(vector, _max, _min):
	result = [0.0] * len(vector)
	for i in range(0, len(vector)):
		result[i] = denormalize(vector[i], _max, _min)
	return result

def create_network(data_dimension=2, history_interval=5, prediction_interval=5, hiddenLayersCount=3):
	n = FeedForwardNetwork()

	inLayer = SigmoidLayer(data_dimension * history_interval) # prices + volumes
	hiddenLayers = []

	for i in range(0, hiddenLayersCount):
	    hiddenLayers.append(SigmoidLayer(data_dimension * history_interval))

	outLayer = SigmoidLayer(prediction_interval)

	n.addInputModule(inLayer)
	for layer in hiddenLayers:
	    n.addModule(layer)
	n.addOutputModule(outLayer)

	n.addConnection(FullConnection(inLayer, hiddenLayers[0]))

	for i in range(1, len(hiddenLayers)):
	    n.addConnection(FullConnection(hiddenLayers[i - 1], hiddenLayers[i]))

	n.addConnection(FullConnection(hiddenLayers[len(hiddenLayers) - 1], outLayer))
	n.sortModules()
	return n

def train_price_volume_sliding_window(network, training_set, data_dimension=2, price_index=0, history_interval=5, prediction_interval=5):
	trainer = RPropMinusTrainer(network, verbose=True, batchlearning=True, learningrate=0.01, lrdecay=0.0, momentum=0.0, weightdecay=0.0)
	dataset = SupervisedDataSet(history_interval * data_dimension, prediction_interval)
	source_window = sliding_window(history_interval, training_set)
	forecast_window = sliding_window(prediction_interval, training_set[history_interval-1:])

	try:
		while True:
			ticks = source_window.next()
			forecast_ticks = forecast_window.next()

			samples = normalizeWindow(ticks)[0]
			forecasts_ticks = normalizeWindow(forecast_ticks)[0]

			forecasts = []

			for forecast in forecasts_ticks:
				forecasts.append(forecast[price_index])


			
			inputs = []


			flattenInput = [item for sublist in samples for item in sublist]
			flattenOutput = forecasts

			print 'Flatten input %s' % flattenInput
			print 'Flatten ou %s' % flattenOutput

			dataset.appendLinked(flattenInput, flattenOutput)


	except StopIteration:
		trainer.setData(dataset)
		trainer.trainUntilConvergence(validationProportion=0.50, maxEpochs=1000, verbose=False)





if __name__ == '__main__':
	price_index = 0
	history_interval = 5

	ticks = load_all_ticks()

	network = create_network(history_interval=history_interval)

	train_end = validate_start = 300

	train_price_volume_sliding_window(network, ticks[0:train_end], history_interval=history_interval, price_index=0)

	writer = csv.writer(open('nexttest.predictions.csv', 'w'), delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)

	
	days = validate_start

	for i in range(0, validate_start): # 5 - prediction interval
		writer.writerow([i+1, 0.0])

	while days < len(ticks)-history_interval:
		sample = ticks[days:days+history_interval]
		normalized, mins, maxs = normalizeWindow(sample)
		print "Normalized"
		print normalized
		flattened = [item for sublist in normalized for item in sublist]
		response = network.activate(flattened)

		forecast_prices = denormalize(response, mins[0], maxs[0])

		prices_output = []
		for v in forecast_prices:
			prices_output.append(float(v))

		for i in range(0,len(prices_output)):
			writer.writerow([days+i+1, prices_output[i]])

		days += history_interval


