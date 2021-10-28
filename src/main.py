"""mnist
Usage:
    mnist make_data <path_raw_data> <path_processed_data>
    mnist make_model <path_processed_data> <learning_rate> <n_tests_console> <network_structure>... 
"""

from docopt import docopt
import pickle
import os
from src.preprocess import read_data, rescale_x, onehot_y, plot_6_random_samples, X_to_1d, rescale_y
from src.neurnet import NeuralNetwork
from src.postprocess import test_random_predictions

def main():

    arguments = docopt(__doc__)

    if arguments['make_data']:
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = read_data(
            path=arguments['<path_raw_data>'],
            filelist=[
                'train-images-idx3-ubyte',
                't10k-images-idx3-ubyte',
                'train-labels-idx1-ubyte',
                't10k-labels-idx1-ubyte'])

        X_train, X_test = rescale_x(X_train_raw), rescale_x(X_test_raw)
        y_train, y_test = onehot_y(y_train_raw), onehot_y(y_test_raw)
        y_train, y_test = rescale_y(y_train), rescale_y(y_test)
        X_train, X_test = X_to_1d(X_train), X_to_1d(X_test)

        path_processed = arguments['<path_processed_data>']
        file_processed = os.path.join(path_processed, 'data_processed.pckl')
        
        if not os.path.isdir(path_processed):
            os.mkdir(path_processed)
        if os.path.isfile(file_processed):
            os.remove(file_processed)
        with open(file_processed, 'wb') as f:
            pickle.dump([X_train, X_test, y_train, y_test, y_test_raw], f)

        plot_6_random_samples(filepath='plots/sample_X_train.png', X=X_train, y=y_train_raw)

    elif arguments['make_model']:
        with open(arguments['<path_processed_data>'], 'rb') as f:
            X_train, X_test, y_train, y_test, y_test_raw = pickle.load(f)

        # handle types
        n_tests_console = int(arguments['<n_tests_console>'])
        network_structure = list(map(int, arguments['<network_structure>']))
        learning_rate = float(arguments['<learning_rate>'])
        n_tests_console = int(arguments['<n_tests_console>'])

        nn = NeuralNetwork(
            network_structure=network_structure, #[784, 50, 10], 
            learning_rate=learning_rate)

        for i in range(len(X_train)):
            nn.train(X_train[i], y_train[i])
            print(f'training sample {i+1}/{len(X_train)}', end='\r', flush=True)

        test_random_predictions(X_test=X_test, y_test=y_test_raw, n=10, model=nn)

        confusion_matrix, precision, recall, corrects, wrongs = nn.quality(X_test, y_test)
#pd.DataFrame({'label': np.arange(0, 10), 'precision': precision, 'recall': recall})


if __name__ == '__main__':
    main()