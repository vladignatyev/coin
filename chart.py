import csv


def predicted(csv_filename):
    f = open(csv_filename, 'r')
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        yield {
            'date': row[0],
            'predicted': float(row[1]),
            'real': float(row[2])
        }


if __name__ == '__main__':
    source = predicted('./predictions.csv')

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from itertools import *
    from datetime import datetime

    dates = [y for y in imap(lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), predicted('./predictions.csv'))]
    prices_predicted = [y for y in imap(lambda x: x['predicted'], predicted('./predictions.csv'))]
    prices_real = [y for y in imap(lambda x: x['real'], predicted('./predictions.csv'))]

    fig, ax = plt.subplots()
    ax.set_title('USD/BTC price prediction')

    line1, = ax.plot(dates, prices_predicted, label='Predictied')
    line2, = ax.plot(dates, prices_real, label='Real')
    ax.legend(handles=[line1, line2], loc=4)

    mondays = mdates.WeekdayLocator(mdates.MONDAY)
    monthsFmt = mdates.DateFormatter("%d %b '%y")
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.autoscale_view()

    ax.grid(True)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()

    fig.savefig('fig.png')