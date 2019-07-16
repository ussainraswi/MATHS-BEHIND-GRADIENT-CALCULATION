############# MATHS BEHIND GRADIENT FUNCTION ###########
# What is grad fn ?
# Derivative of a loss fn w.r.t its parameter is known as Grad function.

import torch
# t_u -> Tensor which is used.
# t_c -> Computed tensor.
# t_p -> Prdicted tensor.

# Model which we use for training, weight(w) and bias(b) are the parameters.
def model(t_u,w,b):
    return w*t_u+b
# MSE Loss fn
def loss_fn(t_p, t_c):
    sq_diffs = (t_p-t_c)**2
    return sq_diffs.mean()

# GRAD FUNCTION(grad_fn())
def dloss_fn(t_p, t_c):
    return 2*(t_p-t_c)

# Derivative of model w.r.t weight.
def dmodel_dw(t_u,w,b):
    return t_u

# Derivative of model w.r.t bias.
def dmodel_db(t_c,w,b):
    return 1.0

# Derivative of a loss fn w.r.t its parameter is known as Grad Fn.
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dw = dloss_fn(t_p, t_c) * dmodel_dw(t_u,w,b)
    dloss_db = dloss_fn(t_p, t_c) * dmodel_db(t_u,w,b)
    return torch.stack([dloss_dw.mean(), dloss_db.mean()])

# Step: 1
# learning_rate = 1e-2

# Step: 2
#   If we reduce the learning rate, we are able to pass relevant values to the
# gradient, the parameter updates in a better way, and model convergence
# becomes quicker.
# learning_rate = 1e-4

# Step: 3
#   If we reduce the learning rate a bit, then the process of weight updating
# will be a little slower, which means that the epoch number needs to be
# increased in order to find a stable state for the model.
# t_un = 0.1*t_u
t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0], requires_grad=True)
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 31.9, 21.8, 48.4, 60.4, 68.4])

params = torch.tensor([1.0, 0.0])
t_un = 0.1*t_u
learning_rate = 1e-2
nepochs = 500

for epoch in range(nepochs):
    # w,b = params
    t_p = model(t_un, *params)
    loss = loss_fn(t_p, t_c)
    print('Epoch %d, Loss %f'%(epoch, float(loss)))
    grad = grad_fn(t_un, t_c, t_p, w, b)
    print('Params ', params)
    print('Grad ', grad)
    params = params-learning_rate*grad
params # tensor([ 4.1411, -9.7461])