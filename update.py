# -*- coding: utf-8 -*-
""" Update Readme.md and cumulative_gans.jpg """
from __future__ import print_function
from __future__ import division
from wordcloud import WordCloud
from wordcloud import STOPWORDS

import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import pandas as pd

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

def update_wordcloud_title():
    """ Update the figure wordcloud_title.jpg """

    data = pd.read_csv('AllGAN-r2.tsv',delimiter='\t', encoding='utf-8')

#    tmp_data = data['Title'].split(" ") for x in data

#    count_list = list([list(x) for x in data['Title'].value_counts().reset_index().values])

#    wordcloud = WordCloud(stopwords=STOPWORDS,relative_scaling = 0.2,
#                        max_words=2000, background_color='white').generate_from_frequencies(tmp_data)
    stopwords = set(STOPWORDS)
    #ganstop = ['Generative','Adversarial', 'Networks', 'Network', 'GAN', 'GANs', 'using', 'Learning', 'Training', 'Generation',
    #        'Neural', 'Net', 'Model', 'Nets', 'Deep', 'Based', 'Via', 'Conditional', 'Models', 'Examples']
    #stopwords.add(ganstop)

    stopwords.add('Generative')
    stopwords.add('Adversarial')
    stopwords.add('Networks')
    stopwords.add('Network')
    stopwords.add('GAN')
    stopwords.add('GANs')
    stopwords.add('using')
    stopwords.add('Learning')
    stopwords.add('Training')
    stopwords.add('Generation')
    stopwords.add('Neural')
    stopwords.add('Net')
    stopwords.add('Model')
    stopwords.add('Nets')
    stopwords.add('Deep')
    stopwords.add('Based')
    stopwords.add('Via')
    stopwords.add('Conditional')
    stopwords.add('Models')
    stopwords.add('Examples')

    wordcloud = WordCloud(stopwords=stopwords,relative_scaling = 0.2, random_state=3,
                    max_words=2000, background_color='white').generate(' '.join(data['Title']))

    plt.figure(figsize=(12,12))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    #plt.show()
    #plt.savefig('wordcloud_title.png')
    wordcloud.to_file('wordcloud_title.png')
    wordcloud.to_file('docs/png/wordcloud_title.png')

def update_wordcloud_category():
    """ Update the figure wordcloud_category.jpg """

    data = pd.read_csv('AllGAN-r2.tsv',delimiter='\t', encoding='utf-8')

    wordcloud = WordCloud(stopwords=STOPWORDS,relative_scaling = 0.2, random_state=3,
                max_words=2000, background_color='white').generate(' '.join(data['Category']))

    plt.figure(figsize=(12,12))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    #plt.show()
    #plt.savefig('wordcloud_title.png')
    wordcloud.to_file('wordcloud_category.png')
    wordcloud.to_file('docs/png/wordcloud_category.png')

if __name__ == '__main__':
    try:
        reload(sys)  # Python 2
        sys.setdefaultencoding('utf-8')
    except NameError:
        pass         # Python 3

    GANS = load_data()
    update_wordcloud_title()
    update_wordcloud_category()
    update_readme(GANS)
    update_index(GANS)
#    update_figure(GANS)
