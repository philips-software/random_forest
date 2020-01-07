Random Forests in MPyC
----------------------

![CI Status][ci]

An implementation of machine learning on secure data. We allow a model to be
trained and used on data that is kept private. We use the [MPyC][mpyc] library
to perform a secure multi-party computation (MPC) that trains a forest of
decision trees using an algorithm that is similar to the C4.5 machine learning
algorithm.

### Installation

Install Python 3.7, then invoke:

```bash
pip install -r requirements.txt
```

### Usage

The [spect.py][spect] and [balance.py][balance] files contain examples of how to
specify a dataset and to train a random forest on this data. These examples can
be run as follows:

```bash
python spect.py
python balance.py
```

Please keep in mind that these computations are much slower than their non-MPC
counterparts.

### Tests

Run the test by invoking:

```bash
pytest
```

Run tests in watch mode:

```bash
ptw [-c]
```

(The `-c` flag causes the screen to be cleared before each run.)

### Profiling

```
pip install snakeviz
python -m cProfile -o spect.stats spect.py
snakeviz spect.stats
```

### Thanks

This algorithm was developed as part of the [SODA project][soda]. Many thanks to
Mark Abspoel, Daniel Escudero and Nikolaj Volgushev for designing the decision
tree algorithm for MPC (See chapter 6 of [this SODA document][paper]). Many
thanks to Berry Schoenmakers who developed [MPyC][mpyc] and helped us throughout
the implementation of this algorithm.

[ci]: https://github.com/philips-software/random_forest/workflows/Build%20and%20test/badge.svg
[soda]: https://www.soda-project.eu
[paper]: https://www.soda-project.eu/wp-content/uploads/2019/10/SODA-D2.3-Use-case-specific-algorithms.pdf
[mpyc]: https://github.com/lschoe/mpyc
[spect]: ./spect.py
[balance]: ./balance.py
