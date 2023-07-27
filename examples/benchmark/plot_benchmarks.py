#!/usr/bin/env python

from argparse import ArgumentParser, FileType
import json
import re
import matplotlib.pyplot as plt


def create_arguments():
    parser = ArgumentParser(description='benchmark visualization tool')
    parser.add_argument('-json_file', '-j', type=FileType('r'),
                        required=True,
                        help='JSON file with benchmark data')

    return parser


def filter_benchmarks(benchmarks, filter_pattern):
    return [x for x in benchmarks if filter_pattern.match(x['name'])]


def filter_benchmark_categories(json):
    benchmarks = [x for x in json['benchmarks'] if x['name'].endswith('_mean')]

    if not benchmarks:
        benchmarks = json['benchmarks']

    return [
        filter_benchmarks(benchmarks, re.compile(r'.+/Build.+')),
        filter_benchmarks(benchmarks, re.compile(r'.+/(Knn|Nn).+')),
        filter_benchmarks(benchmarks, re.compile(r'.+/Radius.+'))
    ]


def create_plots(benchmarks, pattern):
    plots = dict()

    for x in benchmarks:
        m = pattern.match(x['name'])
        k = m.group('tree') + '_' + \
            m.group('type') + (('_' + m.group('arg'))
                               if m.group('arg') else '')
        plots[k] = {'x': [], 'y': []}

    for x in benchmarks:
        m = pattern.match(x['name'])
        k = m.group('tree') + '_' + \
            m.group('type') + (('_' + m.group('arg'))
                               if m.group('arg') else '')
        plots[k]['x'].append(float(m.group('x')))
        plots[k]['y'].append(float(x['cpu_time']))

    return plots


def create_figure(plots, title):
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

    benchmarks = filter_benchmark_categories(json.load(args.json_file))
    re_info = r'^Bm(?P<tree>.+)/(Build|Knn|Nn|Radius)(?P<type>(Ct|Rt)[^/]*)/(?P<x>\d+)(/(?P<arg>\d+))?(_mean)?$'
    plots = [create_plots(b, re.compile(re_info)) for b in benchmarks]
    titles = ['build time', 'knn search time', 'radius search time']
    # Format is determined by filename extension
    extension = '.png'
    file_names = [
        f'./build_time{extension}',
        f'./knn_search_time{extension}',
        f'./radius_search_time{extension}']

    for i in range(len(plots)):
        create_figure(plots[i], titles[i])[0].savefig(file_names[i])
    plt.show()


if __name__ == '__main__':
    main()
