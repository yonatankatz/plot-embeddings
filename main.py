import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import gensim.downloader as api

# --- Configuration ---
# This section defines the words and expressions that will be processed.

# Base words are single words whose vectors are fetched directly from the model.
BASE_WORDS = ['cat', 'dog', 'elephant', 'tower', 'building', 'house',
              'plane', 'car', 'ship', 'japan', 'israel', 'rome',
              'jerusalem', 'italy', 'man', 'woman', 'boy', 'girl',
              'king', 'queen', 'nephew', 'niece', 'princess']

# Reference words for PCA fitting.
# A fixed set of words is used to fit the PCA model. This ensures that the
# projection to 3D space is consistent and stable across different runs,
# regardless of the specific words being visualized. This way, the position of
# a word like 'king' will not change if 'queen' is added or removed.
REFERENCE_WORDS = ['cat', 'dog', 'elephant', 'tower', 'building', 'house',
                   'plane', 'car', 'ship', 'japan', 'israel', 'rome',
                   'jerusalem', 'italy', 'man', 'woman', 'boy', 'girl',
                   'king', 'queen', 'nephew', 'niece']

# Lists of labels to be displayed on each of the four graphs.
# Each list corresponds to a separate plot.
PLOT_LABELS_1 = ['cat', 'dog', 'plane', 'car', 'ship', 'king', 'queen', 'princess']
PLOT_LABELS_2 = ['jerusalem', 'italy', 'israel', 'rome']
PLOT_LABELS_3 = ['jerusalem', 'italy', 'israel', 'rome', 'italy - rome', 'israel - jerusalem']
PLOT_LABELS_4 = ['jerusalem', 'italy', 'israel', 'rome', 'italy - rome', 'israel - jerusalem',
                 'italy + jerusalem - israel']


# --- Main Application Logic ---

def create_and_show_plot(labels_to_show, label_to_3d_vector_map, plot_title):
    """
    Generates and displays a single 3D plot for a given set of labels.

    Args:
        labels_to_show (list): A list of strings, where each string is a label to be plotted.
        label_to_3d_vector_map (dict): A dictionary mapping all available labels to their 3D vectors.
        plot_title (str): The title for the generated plot.
    """
    print(f"\n--- Generating Plot: {plot_title} ---")

    # Step 5: Filter the data to get only the points and labels for the current plot.
    points_to_plot = []
    labels_for_points = []
    for label in labels_to_show:
        if label in label_to_3d_vector_map:
            points_to_plot.append(label_to_3d_vector_map[label])
            labels_for_points.append(label)
        else:
            print(f"Warning: Label '{label}' not found and will be skipped.")

    if not points_to_plot:
        print("No valid points to plot. Skipping this graph.")
        return

    points_to_plot = np.array(points_to_plot)
    print(f"Displaying {len(labels_for_points)} labels on the plot.")

    # Step 6: Create and configure the 3D visualization.
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the filtered points.
    colors = plt.cm.get_cmap('tab10', len(points_to_plot))
    for i, (point, label) in enumerate(zip(points_to_plot, labels_for_points)):
        ax.scatter(point[0], point[1], point[2], s=120, color=colors(i), alpha=0.8, label=label)

    # Annotate each point with its corresponding label.
    for i, label in enumerate(labels_for_points):
        ax.text(points_to_plot[i, 0], points_to_plot[i, 1], points_to_plot[i, 2],
                f'  {label}', fontsize=20, ha='left')

    # Configure plot aesthetics.
    ax.set_title(plot_title, fontsize=16, pad=20)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set fixed axis limits for a consistent viewing window across all plots.
    fixed_range = 4.0
    ax.set_xlim(-fixed_range, fixed_range)
    ax.set_ylim(-fixed_range, fixed_range)
    ax.set_zlim(-fixed_range, fixed_range)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Step 1: Load the pre-trained GloVe model from the gensim library.
    # This model contains word vectors trained on a massive text corpus (Wikipedia).
    # The '100' indicates that each word vector has 100 dimensions.
    print("Loading pre-trained GloVe model (glove-wiki-gigaword-100)...")
    model = api.load('glove-wiki-gigaword-100')
    print("Model loaded successfully.")

    # Step 2: Define and compute vectors for arithmetic expressions.
    # These expressions demonstrate vector relationships (e.g., 'king' - 'man' + 'woman' ≈ 'queen').
    computed_expressions = {
        'italy - rome': model['italy'] - model['rome'],
        'israel - jerusalem': model['israel'] - model['jerusalem'],
        'italy + jerusalem - israel': model['italy'] + model['jerusalem'] - model['israel']
    }

    # Step 3: Create a unified dictionary of all available vectors.
    # This includes both single words and the results of vector arithmetic.
    all_vectors_dict = {word: model[word] for word in BASE_WORDS}
    all_vectors_dict.update(computed_expressions)

    # Step 4: Reduce vector dimensionality from 100D to 3D using PCA.
    # The PCA model is fitted *only* on the reference vectors to ensure stability.
    print("Fitting PCA model on reference words...")
    pca = PCA(n_components=3)
    reference_vectors = [model[word] for word in REFERENCE_WORDS]
    pca.fit(reference_vectors)

    # Transform all available vectors (base and computed) into the 3D space.
    all_labels_list = list(all_vectors_dict.keys())
    all_vectors_list = list(all_vectors_dict.values())
    word_vectors_3d_all = pca.transform(all_vectors_list)

    # Create a final mapping from each label to its new 3D vector for easy lookup.
    label_to_3d_vector = {label: vec for label, vec in zip(all_labels_list, word_vectors_3d_all)}

    # Generate and show each of the four requested plots in sequence.
    create_and_show_plot(PLOT_LABELS_1, label_to_3d_vector, "Graph 1: General Word Categories")
    create_and_show_plot(PLOT_LABELS_2, label_to_3d_vector, "Graph 2: Countries and Capitals")
    create_and_show_plot(PLOT_LABELS_3, label_to_3d_vector, "Graph 3: Country-Capital Vector Relationships")
    create_and_show_plot(PLOT_LABELS_4, label_to_3d_vector, "Graph 4: Vector Arithmetic for Analogies")
