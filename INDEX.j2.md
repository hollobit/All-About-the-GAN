<!DOCTYPE html>
<html lang="en">
<head>
  <!-- Global site tag (gtag.js) - Google Analytics -->
  <script async src="https://www.googletagmanager.com/gtag/js?id=UA-120748798-1"></script>
  <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());

    gtag('config', 'UA-120748798-1');
    </script>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:200,400,700" rel="stylesheet">
    <link rel="stylesheet" type="text/css" href="wp.min.css">
    <link rel="stylesheet"
          href="default.min.css">
    <script src="highlight.min.js"></script>
    <script>hljs.initHighlightingOnLoad();</script>
    <title>All about the GANs(Generative Adversarial Networks) - Summarized lists for GAN</title>
</head>
<body>
<!-- Github banner -->
<a href="https://github.com/hollobit/All-About-the-GAN"><img style="position: fixed; top: 0; right: 0; border: 0;" src="png/forkme.png" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_darkblue_121621.png"></a>

<!-- Menu -->
<input type="checkbox" id="menu"><label for="menu" id="open">☰</label>
<aside>
    <div class="logo">All about the GANs</div>
    <nav>
        <div>
            <a href="#whats-gans">what is GANs?</a>
            <a href="#whats-this">What is this list</a>
            <input type="checkbox" id="s1-all" checked>
            <label for="s1-all">All GANs</label>
            <div class="s1">
                <a href="#doc-whatsnew">What's new</a>
                <a href="#doc-y2018">2018</a>
                <a href="#doc-y2017">2017</a>
                <a href="#doc-y2016">2016</a>
                <a href="#doc-y2015">2015</a>
                <a href="#doc-y2014">2014</a>
                <a href="#doc-medical">Medical</a>
            </div>
        </div>
    </nav>
</aside>

<!-- Main content -->
<main>
    <!-- What is this? -->
    <h1 id="whats-gans">What is GANs?</h1>
    <p>GANs(Generative Adversarial Networks) are the models that used in unsupervised machine learning, implemented by a system of two neural networks competing against each other in a zero-sum game framework. It was introduced by <a href="https://scholar.google.ca/citations?user=iYN86KEAAAAJ&hl=en">Ian Goodfellow</a> et al. in 2014.</p>

    <P><img src="https://deeplearning4j.org/img/gan_schema.png" ALT="concept diagram of GAN model. Credit: O’Reilly" width="100%"> (Credit: O’Reilly)</p>

    <code>
    "The most important one, in my opinion, is adversarial training (also called GAN for Generative Adversarial Networks). This is an idea that was originally proposed by Ian Goodfellow when he was a student with Yoshua Bengio at the University of Montreal (he since moved to Google Brain and recently to OpenAI).
    This, and the variations that are now being proposed is the most interesting idea in the last 10 years in ML, in my opinion."  
    (<a href-"https://www.quora.com/What-are-some-recent-and-potentially-upcoming-breakthroughs-in-deep-learning">Facebook’s AI research director Yann LeCun</a>)
    </code>

    <hr>



    <!-- How can I use it? -->
    <h1 id="whats-this">What is this list?</h1>
    <p>The purpose of this repository is providing the curated list of the state-of-the-art works on the field of Generative Adversarial Networks since their introduction in 2014.</p>

    <p><img src="png/wordcloud_title.png" width="100%" ALT="Title Word Cloud of GAN papers"><br>
    (Word cloud of Title)</p>

    <p><img src="png/wordcloud_category.png" width="100%" ALT="Category Word Cloud of GAN papers"><br>(Word cloud of Category)</p>

    <p>This list provides a curated list that merged information from various GAN lists and repositories as below:</p>
    <h3>Reference repositories</h3>
      <ul>
      <li><a href="https://github.com/hindupuravinash/the-gan-zoo">[GAN zoo]</a> - A list of all named GANs! by hindupuravinash</li>
      <li>Delving deep into Generative Adversarial Networks (GANs) <a href="https://github.com/GKalliatakis/Delving-deep-into-GANs">[Delving]</a> by GKalliatakis</li>
      <li>Awesome GAN for Medical Imaging  <a href="https://github.com/xinario/awesome-gan-for-medical-imaging/">[Medical]</a> by xinario</li>
      <li><a href="https://github.com/zhangqianhui/AdversarialNetsPapers/">[Adversarial Nets Papers]</a> The classic about Generative Adversarial Networks</li>
      <li><a href="https://github.com/nightrome/really-awesome-gan">[Really Awesome GAN]</a> by nightrome</li>
      <li><a href="https://github.com/shawnyuen/GANsPaperCollection">[GANs Paper Collection]</a> by shawnyuen</li>
      <li><a href="https://github.com/nashory/gans-awesome-applications">[GAN awesome applications]</a> by nashory</li>
      <li><a href="https://github.com/dongb5/GAN-Timeline">[GAN timeline]</a> by dongb5</li>
      <li><a href="https://github.com/khanrc/tf.gans-comparison">[GAN comparison without cherry-picking]</a> by khanrc</li>
      <li>Collection of generative models in <a href="https://github.com/eriklindernoren/Keras-GAN">[Keras]</a>, <a href="https://github.com/znxlwm/pytorch-generative-model-collections">[Pytorch version]</a>, <a href="https://github.com/hwalsuklee/tensorflow-generative-model-collections">[Tensorflow version]</a>, <a href="https://github.com/pfnet-research/chainer-gan-lib">[Chainer version]</a></li>
      <li><a href="https://github.com/tensorlayer/tensorlayer">[Tensor layer]</a></li>
      <li><a href="https://github.com/ppwwyyxx/tensorpack">[Tensor pack]</a></li>
      </ul>

      <p></p>
      <p>You can also check out the same data in a tabular format with functionality to filter by year or do a quick search by title <a href="https://github.com/hollobit/All-About-the-GAN/blob/master/AllGAN-r2.tsv">here</a>.</p>
      <p>Contributions for <a href="https://github.com/hollobit/All-About-the-GAN/">this repository</a> are always welcome!!</p>
      <p>Please contact me at <a href="mailto:hollobit@etri.re.kr">hollobit@etri.re.kr</a> or send a pull request. You can have to add links through pull requests or create an issue which something I missed or need to start a discussion.</p>
      <hr>
    <!--  Documentation -->
    <h1 id="s1-all">All GANs</h1>


    <!-- Menu -->

    {% set ncount = {'value': 1} %}
    <h2 id="doc-whatsnew">What's new</h2>
    <dl>
    {% for gan in gans if 'New' in gan['Category'] %}
    <dt><b>{{ gan['Title'] }}</b>  (No: {{ gan['Mnum']}})</dt>
    <dd><a href="http://www.google.com/search?q={{ gan['Title']|urlencode() }})">[Search]</a>  <a href="http://scholar.google.com/scholar?q={{ gan['Title']|urlencode() }})">[Scholar]</a>  
     {%- if gan['pdf'] != '-' and gan['pdf'] != '' -%} <a href="{{ gan['pdf'] }}">[PDF]</a> {% endif %}
     {%- if ncount.update({'value': (ncount.value + 1)}) -%} {% endif %}
     {%- if gan['Arxiv'] != '-' and gan['Arxiv'] != '' -%} <a href="{{ gan['Arxiv'] }}">[arXiv]</a> {% endif %}
     {%- if gan['Official_Code'] != '-' and gan['Official_Code'] != '' -%} <a href="{{ gan['Official_Code'] }}">[github]</a> {% endif %}
     {%- if gan['Tensorflow'] != '-' and gan['Tensorflow'] != '' -%} <a href="{{ gan['Tensorflow'] }}">[TensorFlow]</a> {% endif %}
     {%- if gan['PyTorch'] != '-' and gan['PyTorch'] != '' -%} <a href="{{ gan['PyTorch'] }}">[PyTorch]</a> {% endif %}
     {%- if gan['KERAS'] != '-' and gan['KERAS'] != '' -%} <a href="{{ gan['KERAS'] }}">[KERAS]</a> {% endif %}
     {%- if gan['Web'] != '-' and gan['Web'] != '' -%} <a href="{{ gan['Web'] }}">[Web]</a> {% endif %}

       - {%- if gan['Citations'] | int > 50  %} :dart: {% endif %}
     {%- if gan['Stars'] | int > 10 %} :octocat: {% endif %} `{{ gan['Year'] }}/{{ gan['Month'] }}` {# #}
     {%- if gan['Medical'] != '-' -%} `Medical: {{ gan['Medical'] }}` {% endif %}
     {%- if gan['Category'] != '-' -%} `{{ gan['Category'] }}` {% endif %}  
     {%- if gan['Abbr.'] != '-' and gan['Abbr.'] != '' %} `{{ gan['Abbr.'] }}`  {% endif %}
     {%- if gan['Citations'] != '0' and gan['Citations'] != '' %} `Citation: {{ gan['Citations'] }}` {% endif %}
     {%- if gan['Stars'] != '-' and gan['Stars'] != '' %} `Stars: {{ gan['Stars'] }}` {% endif %}
     </dd>

    {% endfor %}
      </dl>
<hr>

{% set count = {'value': 1} %}
{% set syear_list = [2018, 2017, 2016, 2015, 2014] %}
{% for syear in syear_list %}
<h2 id="doc-y{{ syear }}">{{ syear }}</h2>
<dl>
{% for gan in gans if gan['Year']|int == syear %}
 <dt><b>{{ gan['Title'] }}</b>  (No: {{ gan['Mnum']}})</dt>
 <dd><a href="http://www.google.com/search?q={{ gan['Title']|urlencode() }})">[Search]</a>  <a href="http://scholar.google.com/scholar?q={{ gan['Title']|urlencode() }})">[Scholar]</a>  
  {%- if gan['pdf'] != '-' and gan['pdf'] != '' -%} <a href="{{ gan['pdf'] }}">[PDF]</a> {% endif %}
  {%- if count.update({'value': (count.value + 1)}) -%} {% endif %}
  {%- if gan['Arxiv'] != '-' and gan['Arxiv'] != '' -%} <a href="{{ gan['Arxiv'] }}">[arXiv]</a> {% endif %}
  {%- if gan['Official_Code'] != '-' and gan['Official_Code'] != '' -%} <a href="{{ gan['Official_Code'] }}">[github]</a> {% endif %}
  {%- if gan['Tensorflow'] != '-' and gan['Tensorflow'] != '' -%} <a href="{{ gan['Tensorflow'] }}">[TensorFlow]</a> {% endif %}
  {%- if gan['PyTorch'] != '-' and gan['PyTorch'] != '' -%} <a href="{{ gan['PyTorch'] }}">[PyTorch]</a> {% endif %}
  {%- if gan['KERAS'] != '-' and gan['KERAS'] != '' -%} <a href="{{ gan['KERAS'] }}">[KERAS]</a> {% endif %}
  {%- if gan['Web'] != '-' and gan['Web'] != '' -%} <a href="{{ gan['Web'] }}">[Web]</a> {% endif %}

    - {%- if gan['Citations'] | int > 50  %} :dart: {% endif %}
  {%- if gan['Stars'] | int > 10 %} :octocat: {% endif %} `{{ gan['Year'] }}/{{ gan['Month'] }}` {# #}
  {%- if gan['Medical'] != '-' -%} `Medical: {{ gan['Medical'] }}` {% endif %}
  {%- if gan['Category'] != '-' -%} `{{ gan['Category'] }}` {% endif %}  
  {%- if gan['Abbr.'] != '-' and gan['Abbr.'] != '' %} `{{ gan['Abbr.'] }}`  {% endif %}
  {%- if gan['Citations'] != '0' and gan['Citations'] != '' %} `Citation: {{ gan['Citations'] }}` {% endif %}
  {%- if gan['Stars'] != '-' and gan['Stars'] != '' %} `Stars: {{ gan['Stars'] }}` {% endif %}
  </dd>
{% endfor %}
  </b></dl>
{% endfor %}
</div>

<hr>

{% set mcount = {'value': 1} %}
<h2 id="doc-medical">Medical</h2>
<dl>
{% for gan in gans if gan['Medical'] != '-' %}
<dt><b>{{ gan['Title'] }}</b>  (No: {{ gan['Mnum']}})</dt>
<dd><a href="http://www.google.com/search?q={{ gan['Title']|urlencode() }})">[Search]</a>  <a href="http://scholar.google.com/scholar?q={{ gan['Title']|urlencode() }})">[Scholar]</a>  
 {%- if gan['pdf'] != '-' and gan['pdf'] != '' -%} <a href="{{ gan['pdf'] }}">[PDF]</a> {% endif %}
 {%- if mcount.update({'value': (mcount.value + 1)}) -%} {% endif %}
 {%- if gan['Arxiv'] != '-' and gan['Arxiv'] != '' -%} <a href="{{ gan['Arxiv'] }}">[arXiv]</a> {% endif %}
 {%- if gan['Official_Code'] != '-' and gan['Official_Code'] != '' -%} <a href="{{ gan['Official_Code'] }}">[github]</a> {% endif %}
 {%- if gan['Tensorflow'] != '-' and gan['Tensorflow'] != '' -%} <a href="{{ gan['Tensorflow'] }}">[TensorFlow]</a> {% endif %}
 {%- if gan['PyTorch'] != '-' and gan['PyTorch'] != '' -%} <a href="{{ gan['PyTorch'] }}">[PyTorch]</a> {% endif %}
 {%- if gan['KERAS'] != '-' and gan['KERAS'] != '' -%} <a href="{{ gan['KERAS'] }}">[KERAS]</a> {% endif %}
 {%- if gan['Web'] != '-' and gan['Web'] != '' -%} <a href="{{ gan['Web'] }}">[Web]</a> {% endif %}

   - {%- if gan['Citations'] | int > 50  %} :dart: {% endif %}
 {%- if gan['Stars'] | int > 10 %} :octocat: {% endif %} `{{ gan['Year'] }}/{{ gan['Month'] }}` {# #}
 {%- if gan['Medical'] != '-' -%} `Medical: {{ gan['Medical'] }}` {% endif %}
 {%- if gan['Category'] != '-' -%} `{{ gan['Category'] }}` {% endif %}  
 {%- if gan['Abbr.'] != '-' and gan['Abbr.'] != '' %} `{{ gan['Abbr.'] }}`  {% endif %}
 {%- if gan['Citations'] != '0' and gan['Citations'] != '' %} `Citation: {{ gan['Citations'] }}` {% endif %}
 {%- if gan['Stars'] != '-' and gan['Stars'] != '' %} `Stars: {{ gan['Stars'] }}` {% endif %}
 </dd>

{% endfor %}
  </dl>

<hr>

<h3>#### GANs counter: {{ count.value-1 }}</h3>

<h3>#### Added new papers: {{ ncount.value-1 }}</h3>

<h3>#### Medical related papers: {{ mcount.value-1 }}</h3>

<h3>#### Modified: {{ nowts.strftime('%A, %b %d %Y / %X') }}</h3>

<h3>MIT (c) 2017, 2018 Jonathan Jeon</h3>

<!-- Label to hide menu -->
<label for="menu" id="exit"></label>

</main>
</body>
</html>
