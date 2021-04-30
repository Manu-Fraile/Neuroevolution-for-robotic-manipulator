# A high performance Neurevolution algorithm for robotic manipulator control
This repository contains the work developed by Simone Morettini and Manuel Fraile on the course: Il2202 - Research Methodology and Scientific Writing. This work received tha maximum possible grade, an A.

The aim of this course is to produce a research paper as a training or previous step for the Master's Thesis. In our case, we decide to develop a novel technique for controlling a robotic manipulator. For this, we applied an algorithm called Neuroevolution that will learn how to control a robotic manipulator.

Neuroevolution is a form of artificial intelligence that uses evolutionary algorithms to generate artificial neural networks (ANN), parameters, topology and rules.


## Setup

### Linux

``` virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
python evolve-rnn.py
```

### Windows

Create environment:
`py -m venv venv`

Activate environment:
`.\venv\Scripts\activate`
Install dependencies
`pip install -r ./requirements.txt`

https://www.raffaelechiatto.com/modificare-permessi-di-esecuzione-di-script-in-powershell/
