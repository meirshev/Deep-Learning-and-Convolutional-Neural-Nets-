
from scipy import signal
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# functions

X_d = 4
W = np.round(5*(np.random.random((X_d,X_d))-0.5))
W[abs(W)>3] = 0
y = {
    0: lambda x: np.sum(np.dot(x,W),axis=1),
    1: lambda x: (x[:,0]*W[0,0]+x[:,1]*W[1,1])*(x[:,2]*W[2,2]+x[:,1]*W[3,3]),
    2: lambda x: np.log(np.sum(np.exp(np.dot(x,W)),axis=1))
}

# data
noise = .01;
X = dict(train=5*(np.random.random((1000,X_d))-.5),
         test=5*(np.random.random((200,X_d))-.5))
Y = {i: {
        'train': y[i](X['train'])*(1+np.random.randn(X['train'].shape[0])*.01),
        'test': y[i](X['test'])*(1+np.random.randn(X['test'].shape[0])*.01)}
     for i in range(len(y))}


# set hyperparameters, and allocate a structure for learning accuracy and models
batch = 100
lamb = 0
beta1 = .3
beta2 = .7
eps = 1e-4
epochs = 100
rate = lambda e: .01
models = {
    'lin':{i: dict(loss=dict(train=[], test=[])) for i in range(len(y))},
    'cnn':{i: dict(loss=dict(train=[], test=[])) for i in range(len(y))}
}

# learn a linear model
for fi in range(len(y)):
    m_t, v_t = 0, 0
    model = models['lin'][fi]
    model['w'] = np.zeros(X_d)
    for ei in range(epochs):
        for bi in range(0, len(Y[fi]['train']), batch):
            idx = np.random.randint(0,len(Y[fi]['train']),batch)
            xx, yy = X['train'][idx,:], Y[fi]['train'][idx]
            p = np.dot(xx,model['w'])
            l = np.sum((p - yy) ** 2)
            dl_dp = np.sum(2 * (p - yy))
            dl_dw = np.array([np.dot(xx[:,0],2*(p-yy)),np.dot(xx[:,1],2*(p-yy)),np.dot(xx[:,2],2*(p-yy)),np.dot(xx[:,3],2*(p-yy))])
            m_t = beta1*m_t + (1-beta1)*dl_dw
            v_t = beta2*v_t + (1-beta2)*(dl_dw)**2
            model['w'] -= rate(ei) * (m_t/(1-beta1)) / (np.sqrt(v_t/(1-beta2))+eps)
            model['loss']['train'].append(l/batch)
            model['loss']['test'].append(np.mean((Y[fi]['test']-np.dot(X['test'],model['w']))**2))


# learn a toy CNN
def forward(model, x):
    """Fill a dict with forward pass variables"""
    fwd = {}
    fwd['x'] = x
    fwd['o1'] = np.maximum(0, signal.convolve2d(x, [model['w1']], mode='same'))
    fwd['o2'] = np.maximum(0, signal.convolve2d(x, [model['w2']], mode='same'))
    fwd['m'] = np.array([np.maximum(fwd['o1'][:,0], fwd['o1'][:,1]), np.maximum(fwd['o1'][:,2], fwd['o1'][:,3]),
                         np.maximum(fwd['o2'][:,0], fwd['o2'][:,1]), np.maximum(fwd['o2'][:,2], fwd['o2'][:,3])])
    fwd['p'] = np.dot(np.transpose(fwd['m']), model['u'])
    return fwd




def backprop(model, yy, fwd):
    """Return the derivative of the loss w.r.t. model"""

    dl_du = np.dot(fwd['m'], 2 * (fwd['p'] - yy))
    dl_dp = 2 * (fwd['p'] - yy)

    # Generic Variables
    os = ['o1', 'o2']
    dl_dws = [[], []]


    blank = np.array([0] * batch)
    for w in range(2):

        # Calculating dl_dw1
        indEqual = np.array([np.float64(fwd[os[w]][:, 0] == fwd['m'][(2 * w), :]),
                             np.float64(fwd[os[w]][:, 1] == fwd['m'][(2 * w), :]),
                             np.float64(fwd[os[w]][:, 2] == fwd['m'][(2 * w + 1), :]), blank])
        indGreater = np.array([np.float64(fwd[os[w]][:, 0] > 0),
                               np.float64(fwd[os[w]][:, 1] > 0),
                               np.float64(fwd[os[w]][:, 2] > 0), blank])
        finalIndicator = np.transpose(indEqual[:] * indGreater[:])
        finalIndicator[:, 0] = finalIndicator[:, 0][:] * fwd['x'][:, 1][:]
        finalIndicator[:, 1] = finalIndicator[:, 1][:] * fwd['x'][:, 2][:]
        finalIndicator[:, 2] = finalIndicator[:, 2][:] * fwd['x'][:, 3][:]
        final = finalIndicator.dot(np.array([model['u'][(2 * w)], model['u'][(2 * w)],
                                             model['u'][(2  *w + 1)], model['u'][(2 * w + 1)]]))
        dl_dws[w].append(dl_dp.dot(final))

        # Calculating dl_dw2
        indEqual = np.array([np.float64(fwd[os[w]][:, 0] == fwd['m'][(2 * w), :]),
                             np.float64(fwd[os[w]][:, 1] == fwd['m'][(2 * w), :]),
                             np.float64(fwd[os[w]][:, 2] == fwd['m'][(2 * w + 1), :]),
                             np.float64(fwd[os[w]][:, 3] == fwd['m'][(2 * w + 1), :])])
        indGreater = np.array([np.float64(fwd[os[w]][:, 0] > 0), np.float64(fwd[os[w]][:, 1] > 0),
                               np.float64(fwd[os[w]][:, 2] > 0), np.float64(fwd[os[w]][:, 3] > 0)])
        finalIndicator = np.transpose(indEqual[:] * indGreater[:])
        final = finalIndicator[:] * fwd['x'][:]
        final = final.dot(np.array([model['u'][(2*w)], model['u'][(2*w)],
                                             model['u'][(2*w +1)], model['u'][(2*w+1)]]))
        dl_dws[w].append(dl_dp.dot(final))

        # Calculating dl_dw3
        indEqual = np.array([blank, np.float64(fwd[os[w]][:, 1] == fwd['m'][(2 * w), :]),
                             np.float64(fwd[os[w]][:, 2] == fwd['m'][(2 * w + 1), :]),
                             np.float64(fwd[os[w]][:, 3] == fwd['m'][(2 * w + 1), :])])
        indGreater = np.array([blank, np.float64(fwd[os[w]][:, 1] > 0),
                               np.float64(fwd[os[w]][:, 2] > 0), np.float64(fwd[os[w]][:, 3] > 0)])
        finalIndicator = np.transpose(indEqual[:] * indGreater[:])
        finalIndicator[:, 1] = finalIndicator[:, 1][:] * fwd['x'][:, 0][:]
        finalIndicator[:, 2] = finalIndicator[:, 2][:] * fwd['x'][:, 1][:]
        finalIndicator[:, 3] = finalIndicator[:, 3][:] * fwd['x'][:, 2][:]
        final = finalIndicator.dot(np.array([model['u'][(2 * w)], model['u'][(2 * w)],
                                             model['u'][(2 * w + 1)], model['u'][(2 * w + 1)]]))
        dl_dws[w].append(dl_dp.dot(final))


    dl_dw1 = np.array(dl_dws[0])
    dl_dw2 = np.array(dl_dws[1])
    dl_dtheta = np.hstack((dl_dw1, dl_dw2, dl_du))
    return dl_dtheta

final_weights = {0:[],1:[],2:[]}
for fi in range(len(y)):

    m_t, v_t = 0, 0
    model = models['cnn'][fi]

    theta = .1 * (np.random.randn(10) - .5)
    model['w1'] = theta[:3]
    model['w2'] = theta[3:6]
    model['u'] = theta[6:]

    for ei in range(epochs):
        for bi in range(0, len(Y[fi]['train']), batch):

            idx = np.random.randint(0, len(Y[fi]['train']), batch)
            xx, yy = X['train'][idx, :], Y[fi]['train'][idx]
            fwd = forward(model, xx)
            l = np.sum((fwd['p'] - yy) ** 2)
            dl_dtheta = backprop(model, yy, fwd) + lamb * theta
            m_t = beta1 * m_t + (1 - beta1) * dl_dtheta
            v_t = beta2 * v_t + (1 - beta2) * (dl_dtheta) ** 2
            theta -= rate(ei) * (m_t / (1 - beta1)) / (np.sqrt(v_t / (1 - beta2)) + eps)
            model['loss']['train'].append(l / batch)
            model['loss']['test'].append(np.mean((Y[fi]['test'] - forward(model, X['test'])['p']) ** 2))



#checkCalc()
# some plots

MODEL = 'cnn'
for i in range(len(y)):

    plt.subplot(3, 2, i*2+1)
    l = len(models[MODEL][i]['loss']['train'])
    plt.plot(np.arange(l), models[MODEL][i]['loss']['train'],
             np.arange(l), models[MODEL][i]['loss']['test'], lw=2)
    plt.ylim([0, 20])
    plt.subplot(3, 2, i*2+2)
    #plt.scatter(X['test'].dot(models[MODEL][i]['w']), Y[i]['test'])
    plt.scatter(forward(models[MODEL][i], X['test'])['p'], Y[i]['test'])
    plt.axis('equal')
plt.show()