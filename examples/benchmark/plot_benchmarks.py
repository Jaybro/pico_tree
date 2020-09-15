#!/usr/bin/env python

import argparse
from argparse import ArgumentParser
import json
import re
import matplotlib.pyplot as plt


def create_arguments():
    parser = ArgumentParser(description='benchmark visualization tool')
    parser.add_argument('-json_file', '-j', type=argparse.FileType('r'),
                        required=True,
                        help='JSON file with benchmark data')

    return parser


def get_benchmark_subset(benchmarks, pattern):
    return [x for x in benchmarks if pattern.match(x['name'])]


def get_benchmarks(json):
    benchmarks = [x for x in json['benchmarks'] if x['name'].endswith('_mean')]

    if not benchmarks:
        benchmarks = json['benchmarks']

    return [
        get_benchmark_subset(benchmarks, re.compile(r'.+(Nano|Pico)Build.+')),
        get_benchmark_subset(benchmarks, re.compile(r'.+(Nano|Pico)Knn.+')),
        get_benchmark_subset(benchmarks, re.compile(r'.+(Nano|Pico)Radius.+'))
    ]


def get_plots(benchmarks_subset, pattern):
    plots = dict()

    for x in benchmarks_subset:
        m = pattern.match(x['name'])
        k = m.group('lib') + '_' + m.group('label')
        plots[k] = {'x': [], 'y': []}

    for x in benchmarks_subset:
        m = pattern.match(x['name'])
        k = m.group('lib') + '_' + m.group('label')
        plots[k]['x'].append(float(m.group('x')))
        plots[k]['y'].append(float(x['cpu_time']))

    return plots


def get_figure(plots, title):
    fig, ax = plt.subplots()

    for label in plots:
        x = plots[label]['x']
        y = plots[label]['y']
        ax.plot(x, y, label=label, marker='x')

    ax.set_xlabel('max leaf size')
    ax.set_ylabel('cpu time ms')
    ax.set_title(title)
    ax.legend()

    return fig, ax


def main():
    parser = create_arguments()
    args = parser.parse_args()

    benchmarks = get_benchmarks(json.load(args.json_file))

    re_str_build = r'^.+/(?P<label>Ct|Rt)(?P<lib>Nano|Pico).+/(?P<x>\d+).*$'
    re_str_other = r'^.+/(Ct|Rt)(?P<lib>Nano|Pico).+/(?P<x>\d+)/(?P<label>\d+).*$'

    plots_build = get_plots(benchmarks[0], re.compile(re_str_build))
    plots_knn = get_plots(benchmarks[1], re.compile(re_str_other))
    plots_radius = get_plots(benchmarks[2], re.compile(re_str_other))

    get_figure(plots_build, 'build time')
    get_figure(plots_knn, 'knn search time')
    get_figure(plots_radius, 'radius search time')

    plt.show()


if __name__ == '__main__':
    main()
