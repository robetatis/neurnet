
import numpy as np
import src.neurnet

def test_random_predictions(X_test, y_test, n, model):
    
    """
    Function to look at some random predictions with test data in the console
    """
    
    y_pred_label = []
    y_pred_max_activation = []
    index = np.random.randint(low=0, high=len(X_test), size=n)
    for i in index:
        y_pred_activations = model.predict(X_test[i])
        y_pred_label.append(np.argmax(y_pred_activations))
        y_pred_max_activation.append(np.max(y_pred_activations))
    
    print(f'\n==== predictions for {n} random samples ===')
    print('\ny_test\ty_pred\tactivation')
    for i, idx in enumerate(index):
        print(f'{y_test[idx]}\t{y_pred_label[i]}\t{y_pred_max_activation[i]}')

def computesavequality(X, y, model, outfileprefix):
    cm, p, r, corrects, wrongs = model.quality(X, y)
    x = {'cm': cm, 'p': p, 'r': r, 'corrects': corrects, 'wrongs': wrongs}
    for k, v in x.items():
        outfile = outfileprefix + 'quality_' + k + '.txt'
        np.savetxt(outfile, v)

