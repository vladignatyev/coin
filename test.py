from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.datasets import SupervisedDataSet
from bot.training import days


n = FeedForwardNetwork()

inLayer = LinearLayer(10)
hiddenLayers = []

for i in range(0, 3):
    hiddenLayers.append(LinearLayer(10))

outLayer = LinearLayer(5)

n.addInputModule(inLayer)
for layer in hiddenLayers:
    n.addModule(layer)
n.addOutputModule(outLayer)

n.addConnection(FullConnection(inLayer, hiddenLayers[0]))

for i in range(1, len(hiddenLayers)):
    n.addConnection(FullConnection(hiddenLayers[i - 1], hiddenLayers[i]))

n.addConnection(FullConnection(hiddenLayers[len(hiddenLayers) - 1], outLayer))
n.sortModules()



# training set
DS = SupervisedDataSet(10, 5)

data = days('btceUSD.days.csv')


def normalize(v, _max, _min):
    return 2.0 / (_max - _min) * (v - _min) - 1.0


def denormalize(v, _max, _min):
    return (v + 1.0) / 2.0 * (_max - _min) + _min


day = 0
while day < 100:
    window_prices = []
    window_volumes = []
    for i in range(0, 5):
        tick = data.next()
        price, volume = tick[0], tick[1]
        window_prices.append(price)
        window_volumes.append(volume)

    price_min = min(*window_prices)
    price_max = max(*window_prices)

    volume_min = min(*window_volumes)
    volume_max = max(*window_volumes)

    forecast = []
    for i in range(0, 5):
        tick = data.next()
        price = tick[0]
        forecast.append(price)

    price_max = max(price_max, *forecast)
    price_min = min(price_min, *forecast)

    for i in range(0, 5):
        window_prices[i] = normalize(window_prices[i], price_max, price_min)
        window_volumes[i] = normalize(window_volumes[i], volume_max, volume_min)

    for i in range(0, 5):
        forecast[i] = normalize(forecast[i], price_min, price_max)

    inputs = [0.0] * 10
    inputs[0] = window_prices[0]
    inputs[2] = window_prices[1]
    inputs[4] = window_prices[2]
    inputs[6] = window_prices[3]
    inputs[8] = window_prices[4]

    inputs[1] = window_volumes[0]
    inputs[3] = window_volumes[1]
    inputs[5] = window_volumes[2]
    inputs[7] = window_volumes[3]
    inputs[9] = window_volumes[4]

    DS.appendLinked(inputs, forecast)

    day += 1


# training
trainer = RPropMinusTrainer(n, verbose=True, batchlearning=True, learningrate=0.01, lrdecay=0.0, momentum=0.0,
                            weightdecay=0.0)
trainer.setData(DS)
trainer.trainUntilConvergence(validationProportion=0.25, maxEpochs=100)

# validating
valid_data = days('btceUSD.days.csv')
for i in range(0, 105):
    valid_data.next()

window = []

max_price = 0.0
min_price = float('inf')

max_volume = 0.0
min_volume = float('inf')

print "Test window: "
for i in range(0, 5):
    tick = data.next()
    price, volume, date = tick[0], tick[1], tick[2]
    window.append(price)
    window.append(volume)
    max_price = max(max_price, price)
    min_price = min(min_price, price)

    max_volume = max(max_volume, volume)
    min_volume = min(min_volume, volume)

    print "%s | %s %s" % (date, price, volume)

window[0] = normalize(window[0], max_price, min_price)
window[2] = normalize(window[2], max_price, min_price)
window[4] = normalize(window[4], max_price, min_price)
window[6] = normalize(window[6], max_price, min_price)
window[8] = normalize(window[8], max_price, min_price)

window[1] = normalize(window[1], max_volume, min_volume)
window[3] = normalize(window[3], max_volume, min_volume)
window[5] = normalize(window[5], max_volume, min_volume)
window[7] = normalize(window[7], max_volume, min_volume)
window[9] = normalize(window[9], max_volume, min_volume)

output = n.activate(window)

print denormalize(output[0], max_price, min_price)
print denormalize(output[1], max_price, min_price)
print denormalize(output[2], max_price, min_price)
print denormalize(output[3], max_price, min_price)
print denormalize(output[4], max_price, min_price)
