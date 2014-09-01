from datetime import date
from bot.training import historicalData

prices = historicalData('./btceUSD.csv')


import csv

f = open('btceUSD.days.csv', 'w')
writer = csv.writer(f)

PRICE = 0
VOLUME = 1

try:
	today = []
	today_timestamp = None
	while True:
		tick = prices.next()
		timestamp, price, volume = int(tick[0]) * 1.0, float(tick[1]), float(tick[2])

		t = date.fromtimestamp(timestamp)
		if not today_timestamp:
			today_timestamp = t

		if t != today_timestamp:
			open_price = today[0][PRICE]
			close_price = today[len(today)-1][PRICE]
			
			median_price = 0.0

			total_volume = 0.0
			for item in today:
				total_volume += item[VOLUME]
				median_price = median_price + item[PRICE] * item[VOLUME]

			median_price = median_price / total_volume

			data_item = (str(today_timestamp), open_price, close_price, median_price, total_volume)
			writer.writerow(data_item)

			today_timestamp = t
			today = []
		
		today.append((price, volume))


except StopIteration:

	pass