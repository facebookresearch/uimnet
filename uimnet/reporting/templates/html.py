#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from uimnet import __PROJECT_ROOT__
HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>UIMNET report.</title>

  <meta name="description"
    content="UIMNET report" />

  <link rel="stylesheet" href="/uimnet/reporting/templates/assets/style.css" />
  <link rel="stylesheet" href="/uimnet/reporting/templates/assets/prism/prism.css" />
</head>

<body id="top">
  <header>
    <h1><span class="latex">UIMNET Report</h1>
    <p class="author">
      DLP and MIB<br />
      May 2021
    </p>
  </header>

  <div class="abstract">
    <h2>Abstract</h2>
    <p>
      Uncertanity detection benchmark.
    </p>
  </div>

  <main>
    <article>

      <h2>Experimental Setup </h2>
      <p>User: user</p>
      <p>Date: data</p>
      <p>Commit hash: commit_hash</p>


      <h2>Dataset</h2>
      <h3>Config</h3>
     <p>TODO</p>
      <h3>Partition</h3>
      <h4>Dendrogram</h4>

      <h2>Training</h2>
      <h3>Config</h3>
      <h3>Training curves</h3>
      <h3>Evaluation curves</h3>

      <h2>Calibration</h2>
      <h3>Config</h3>

      <h2>Ensembling</h2>
      <h3>Config</h3>

      <h2>In-domain evaluation</h2>
      {indomain_table}

      <h2>Out-of-domain evaluation</h2>
      {oodomain_table}


    </article>
  </main>

  <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'],],
      }},
    }}
    const toggle = document.getElementById('typeface-toggle')
    const typeface = document.getElementById('typeface')
    toggle.addEventListener('click', () => {{
      document.body.classList.toggle('libertinus')
      typeface.textContent = document.body.classList.contains('libertinus') ? 'Libertinus' : 'Latin Modern'
    }})
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <script async src="assets/prism/prism.js"></script>

  <script async defer data-domain="latex.now.sh" src="https://plausible.io/js/plausible.js"></script>
</body>

</html>
"""
if __name__ == '__main__':
  pass
