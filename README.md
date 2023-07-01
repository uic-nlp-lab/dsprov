# Provenience of discharge summaries

[![Python 3.10][python310-badge]][python310-link]

This repository contains source code for the paper [Hospital Discharge
Summarization Data Provenance](https://example.com).  If you only want to use
the annotations, see the [Inclusion in Your
Projects](#inclusion-in-your-projects) section.

This repository contains the [annotations used in the
paper](dist/dsprov-annotations.json) and classes for creating discharge summary
*provenience of data* annotations.  This project attempts to give an idea of
from where physicians copy/paste and/or summarize previous medical records when
writing discharge summaries.

The project also used an automated method for [note
matching](https://github.com/plandes/spanmatch) and an automated method for
[note segmentation](https://github.com/uic-nlp-lab/medsecid).


## Inclusion in Your Projects

The purpose of this repository is to reproduce the results in the paper. If you
want to use the annotations and/or use the pretrained model, please refer to
the [zensols.dsprov] repository.  This repository also provides a [Docker
image](https://github.com/plandes/dsprov#docker).  If you use our annotations
and/or code, please [cite](#citation) our paper.


## Reproducing Results

The source annotation files are necessary to reproduce our results.  Those can
be obtained by requesting them from the authors.

**Important**: you must provide proof that you have access to by [requesting
MIMIC-III access](https://mimic.mit.edu/docs/gettingstarted/) in your email
request for the source annotations.

Dependencies:

* A macOS machine.
* Microsoft Word (used to annotate spans across notes).
* GNU make.  What default that comes with macOS should be sufficient.  However,
  brew might be necessary to install the GNU version of some system tools.

Steps to reproducing:

1. Clone this repository and go in to it:
   `git clone https://github.com/uic-nlp-lab/dsprov && cd dsprov`
1. Optionally create a virtual environment: `python -m venv <Python install dir>`
1. Install Python dependencies: `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True make deps`
1. Copy the source annotations compressed file to the current directory.
1. Install the source annotation files in the `corpus/completed` directory:
   `$ unzip dsprov-source-annotations.zip`
1. Load MIMIC-III by following the [Postgres instructions].  Also see the
   [zensols.mimic instructions](https://github.com/plandes/mimic#installation)
   for an SQLite alternative.
1. Edit [etc/db.conf](etc/db.conf) using the parameters of the installed
   database from the previous step.
1. Tell programs where to find the database configuration (assuming Bash):
   `export MIMICSIDRC=./etc/db.conf`
1. Create the corpus and matching statistics (also confirms everything is
   installed and working): `./harness.py excel -o match.xlsx`
1. Check for errors and confirm the data in generated file is sound: `open
   match.xlsx`
1. Run the hyperparameter optimization: `./src/bin/opthyper.py opt -e 500`


## Citation

If you use this project in your research please use the following BibTeX entry:

```bibtex
@inproceedings{landes-etal-2023-dsprov,
    title = "{{Hospital Discharge Summarization Data Provenance}}",
    author = "Landes, Paul  and
      Chaise, Aaron J.  and
      Patel, Kunal P. and
      Huang, Sean S.  and
      Di Eugenio, Barbara",
    booktitle = "Proceedings of the 21st {{Workshop}} on {{Biomedical Language Processing}}",
    month = jul,
    year = "2023",
    day = 13,
    address = "Toronto, Canada",
    publisher = "{{Association for Computational Linguistics}}"
}
```

Also please cite the [Zensols Framework]:

```bibtex
@article{Landes_DiEugenio_Caragea_2021,
  title={DeepZensols: Deep Natural Language Processing Framework},
  url={http://arxiv.org/abs/2109.03383},
  note={arXiv: 2109.03383},
  journal={arXiv:2109.03383 [cs]},
  author={Landes, Paul and Di Eugenio, Barbara and Caragea, Cornelia},
  year={2021},
  month={Sep}
}
```


## License

[MIT License](LICENSE.md)

Copyright (c) 2023 Paul Landes


<!-- links -->
[python310-badge]: https://img.shields.io/badge/python-3.10-blue.svg
[python310-link]: https://www.python.org/downloads/release/python-310
[build-badge]: https://github.com/uic-nlp-lab/dsprov/workflows/CI/badge.svg
[build-link]: https://github.com/uic-nlp-lab/dsprov/actions

[zensols.dsprov]: https://github.com/plandes/dsprov
[Zensols Framework]: https://github.com/plandes/deepnlp
[Postgres instructions]: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/buildmimic/postgres/README.md
