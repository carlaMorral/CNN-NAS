# CNN-NAS

Primer pas: executar NAS en local
Veure com reduir el numero d'epochs, training examples i la mida del search space

Segon pas: incorporar temps d'execucio en el metric

Després fer experiments. Com varia la performance amb:
- diferents funcions al metric
- diferent numero d'epochs/training samples
- possiblement, diferents estratègies de cerca (e.g. carregar-te models dolents abans d'hora)

### How to run
Install the requirements:
`python3 setup.py install`
Run the project:
`python3 main.py`
