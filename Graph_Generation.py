import networkx as nx

def create_graph_from_message(message):
    G = nx.Graph()
    for i, word in enumerate(message):
        G.add_node(word)
        neighbors = message[max(0, i - 1): i] + message[i + 1: i + 2]  # Add previous and next words as neighbors
        for neighbor in neighbors:
            if neighbor != word:  # Skip self-loops
                G.add_edge(word, neighbor)
    return G

# Create graphs for the cleaned messages in the DataFrame
df['Graph'] = df['Message_clean'].apply(create_graph_from_message)
# Create a set of unique words in the dataset
unique_words = set(word for message in df['Message_clean'] for word in message)

# Create a dictionary that maps each unique word to a unique integer
word_to_index = {word: idx for idx, word in enumerate(unique_words)}
import numpy as np

def assign_features_to_nodes(graph):
    for node in graph.nodes():
        one_hot = np.zeros(len(unique_words))
        one_hot[word_to_index[node]] = 1
        graph.nodes[node]['feature'] = one_hot

# Apply feature assignment to each graph in the DataFrame
df['Graph'].apply(assign_features_to_nodes)
