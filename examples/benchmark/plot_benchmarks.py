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
        get_benchmark_subset(benchmarks, re.compile(r'.+/Build.+')),
        get_benchmark_subset(benchmarks, re.compile(r'.+/(Knn|Nn).+')),
        get_benchmark_subset(benchmarks, re.compile(r'.+/Radius.+'))
    ]


def get_plots(benchmarks_subset, pattern):
    plots = dict()

    for x in benchmarks_subset:
        m = pattern.match(x['name'])
        k = m.group('tree') + '_' + \
            m.group('type') + (('_' + m.group('arg'))
                               if m.group('arg') else '')
        plots[k] = {'x': [], 'y': []}

    for x in benchmarks_subset:
        m = pattern.match(x['name'])
        k = m.group('tree') + '_' + \
            m.group('type') + (('_' + m.group('arg'))
                               if m.group('arg') else '')
        plots[k]['x'].append(float(m.group('x')))
        plots[k]['y'].append(float(x['cpu_time']))

    return plots


def get_figure(plots, title):
    fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)

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
    re_info = r'^Bm(?P<tree>.+)/(Build|Knn|Nn|Radius)(?P<type>(Ct|Rt)[^/]*)/(?P<x>\d+)(/(?P<arg>\d+))?$'
    plots = [get_plots(b, re.compile(re_info)) for b in benchmarks]

    # Format is determined by filename extension
    extension = '.png'
    get_figure(plots[0], 'build time')[0].savefig(f'./build_time{extension}')
    get_figure(plots[1], 'knn search time')[
        0].savefig(f'./knn_search_time{extension}')
    get_figure(plots[2], 'radius search time')[
        0].savefig(f'./radius_search_time{extension}')
    plt.show()


if __name__ == '__main__':
    main()
