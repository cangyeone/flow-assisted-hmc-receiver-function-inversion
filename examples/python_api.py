"""CPU example. The short chain checks installation; increase the budget for science."""
import numpy as np
from flowrf import forward,RFData,NeuralModels,RFInverter
layers=[[0,3.6,2,2.5],[2,6.5,6.5/1.78,2.8],[35,8,4.5,3.3]]
p=[.044,.059,.075]
rf=forward(layers,p,gaussians=[1,2.5,4]).detach().numpy()[0]
data=RFData(rf,p,gaussians=(1,2.5,4),sigma=.03)
net=NeuralModels()
candidates=net.propose(data,draws=16,seed=728)
emulated=net.generate(layers,p,draws=16)
result=RFInverter().invert(data,initializer='flow',flow_draws=16,starts=2,steps=30,burn_in=15,maxiter=12,leapfrog_steps=3)
result.save('outputs/python-example');result.plot('outputs/python-example/rf-fit.png')
print(result.summary['status'])
