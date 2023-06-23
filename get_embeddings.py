import networkx as nx
from node2vec import Node2Vec

if __name__ == "__main__":
    # Step 1: Load the GraphML file
    graph = nx.read_graphml('D:\DATA-OSM-Dinasty\DATASET.graphml')

    # Step 2: Preprocess the graph (if needed)
    # ...

    # Step 3: Choose an embedding technique
    # Here, we'll use node2vec

    # Step 4: Generate graph embeddings
    # Set the parameters for node2vec
    p = 1.0  # Return hyperparameter
    q = 1.0  # In-out hyperparameter
    dimensions = 128  # Embedding dimensions
    walk_length = 80  # Length of each random walk
    num_walks = 150  # Number of random walks to generate

    # Create a node2vec object and generate embeddings
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Step 5: Save the node embeddings
    output_file = 'embeddings.txt'  # File path to save the embeddings
    model.wv.save_word2vec_format(output_file)