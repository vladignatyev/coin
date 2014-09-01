#!coding: utf-8
import csv


def historicalData(csv_filename):
	f = open(csv_filename, 'r')
	reader = csv.reader(f, delimiter=',')
	for row in reader:
		timestamp = row[0]
		price = row[1]
		volume = row[2]
		yield (timestamp, price, volume)



def days(csv_filename):
	f = open(csv_filename, 'r')
	reader = csv.reader(f, delimiter=',')
	for row in reader:
		date = row[0]
		open_price = row[1]
		close_price = row[2]
		median_price = row[3]
		volume = row[4]
		yield (float(median_price), float(volume), date)
