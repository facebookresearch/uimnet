#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
LATEX = r"""
\documentclass{{article}}
\usepackage{{booktabs}}

\title{{Report}}
\author{{DLP and MIB}}
\date{{\relax}}

\begin{{document}}
\section{{In-domain evaluation}}
{indomain_table}
\section{{Out-of-domain evaluation}}
{oodomain_table}
\end{{document}}
"""
if __name__ == '__main__':
  pass
