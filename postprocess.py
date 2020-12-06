
import numpy as np
import neurnet

def computesavequality(X, y, model, outfileprefix):
    cm, p, r, corrects, wrongs = model.quality(X, y)
    x = {'cm': cm, 'p': p, 'r': r, 'corrects': corrects, 'wrongs': wrongs}
    for k, v in x.items():
        outfile = outfileprefix + 'quality_' + k + '.txt'
        np.savetxt(outfile, v)

