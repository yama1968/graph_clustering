# graph_clustering

Pré-requis:

- Python 3.5 / Anaconda
- networkx
- installation metis (http://glaros.dtc.umn.edu/gkhome/views/metis) et interface Python (http://metis.readthedocs.io/en/latest/), attention à compiler metis sur Linux en "make config shared=1"

Test sur deux dataset
- 4k noeuds / 88k arêtes / Facebook
- 81k noeuds / 2.4M arêtes / Twitter

Sous-groupes par vecteurs propres

Sous-groupes par Metis

facebook_graph.ipynb / bicluster -> premières exploration avec SpectralCoclustering sklearn => bof
facebook_cluster -> k-means sur eigevectors du Laplacien => good
facebook_metis -> facebook avec metis

