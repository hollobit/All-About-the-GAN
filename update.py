# -*- coding: utf-8 -*-
""" Update Readme.md and cumulative_gans.jpg """
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime

def load_data():
    """ Load GANs data from the AllGAN.csv file """
    import csv
    import codecs

    with codecs.open('AllGAN-r2.tsv',"rbU", "utf-8") as fid:
        reader = csv.DictReader(fid, delimiter='\t')
        gans = [row for row in reader]
    return gans


def update_readme(gans):
    """ Update the Readme.md text file from a Jinja2 template """
    import jinja2 as j2

    gans.sort(key=lambda v: v['Title'].upper())
    j2_env = j2.Environment(loader=j2.FileSystemLoader('.'),
                            trim_blocks=True, lstrip_blocks=True)

    j2_env.globals['nowts'] = datetime.datetime.now()

    with open('README-one.md', 'w') as fid:
        print(j2_env.get_template('README.j2.md').render(gans=gans), file=fid)

def update_index(gans):
    """ Update the index.html text file from a Jinja2 template """
    import jinja2 as j2

    try:
        gans.sort(key=lambda v: ((int(v['Year']) if v['Year'].isdigit() else v['Year'])
        , (int(v['Month']) if v['Month'].isdigit() else v['Month'])), reverse=True)
    except:
        pass
    j2_env = j2.Environment(loader=j2.FileSystemLoader('.'),
                            trim_blocks=True, lstrip_blocks=True)

    j2_env.globals['nowts'] = datetime.datetime.now()

    with open('docs/index.html', 'w') as fid:
        print(j2_env.get_template('INDEX.j2.md').render(gans=gans), file=fid)


def update_figure(gans):
    """ Update the figure cumulative_gans.jpg """
    data = np.array([int(gan['Year']) + int(gan['Month']) / 12
                     for gan in gans])
    x_range = int(np.ceil(np.max(data) - np.min(data)) * 12) + 1
    y_range = int(np.ceil(data.size / 10)) * 10 + 1

    with plt.style.context("seaborn"):
        plt.hist(data, x_range, cumulative="True")
        plt.xticks(range(2014, 2018))
        plt.yticks(np.arange(0, y_range, 15))
        plt.title("Cumulative number of named GAN papers by month")
        plt.xlabel("Year")
        plt.ylabel("Total number of papers")
        plt.savefig('cumulative_gans.jpg')


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf-8')

    GANS = load_data()
    update_readme(GANS)
    update_index(GANS)
#    update_figure(GANS)
