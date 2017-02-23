from numpy import *
import pandas as pd
import os
import os.path as osp
import cPickle
from chains import load_chain

def like1d(data, weights, bins = 100):
    
    H, xe = histogram(data,bins,weights=weights,normed=False)
    xem = movavg(xe,2)
    fitdata = [[d]*w for d,w in zip(data, weights)]
    fitdata = [item for sublist in fitdata for item in sublist]
    mu, sigma = norm.fit(fitdata)
    maxval = max(norm.pdf(linspace(min(data), max(data), 10000), scale = sigma, loc = mu))
    return H, xem, mu, sigma, maxval


def likelihood(data, weights, bins = 100):
    
    H, xe = histogram(data,bins,weights=weights,normed=False)
    xem = movavg(xe,2)
    fitdata = [[d]*w for d,w in zip(data, weights)]
    fitdata = [item for sublist in fitdata for item in sublist]
    df, mu, sigma = chi2.fit(fitdata)
    maxval = max(chi2.pdf(df, linspace(min(data), max(data), 10000), scale = sigma, loc = mu))
    return H,  df, xem, mu, sigma, maxval

def get_data():
    '''
    Grabs the cepheids and sne from the R16 sample
    returns: 
        cepheids: Cepheid dataframe
        sne: sne dataframe
    '''

    filename = osp.join(os.getcwd(), 'data', 'R16_table4.out')
    sne_start = 40
    sne_end = 59
    sne_lines = arange(sne_start,sne_end)
    sne = pd.DataFrame(columns = ['Host', 'sne', 'm^B_0', 'SN_err'], index = arange(sne_end - sne_start))
    ceph_start = 70
    ceph_end = 2346
    cepheid_lines = arange(ceph_start,ceph_end)
    cepheids = pd.DataFrame(columns = ['Field','RA','DEC','ID','Period','VminusI','m_H','sigma_tot','Z'], 
                            index = arange(ceph_end - ceph_start),
                           dtype = 'float')
    f = file(filename)
    for i, line in enumerate(f):
        if i in sne_lines:
            sne.loc[i-sne_start] = line.lower().split()
        if i in cepheid_lines:
            cepheids.loc[i-ceph_start] = line.lower().split()

    f.close()
    cepheids = cepheids.apply(lambda x: pd.to_numeric(x, errors='ignore') );

    sne = sne.apply(lambda x: pd.to_numeric(x, errors='ignore') );


    parallaxes = {'bgcru': (2.23, 0.30,-0.15), 
                  'dtcyg':(2.19,0.33, -0.18), 
                  'ffaql':(2.64,0.16, -0.03),
                  'rtaur':(2.31, 0.19,-0.06),
                  'sscma':(0.348, 0.038, -0.04),
                  'sucas':(2.57,  0.33, -0.13 ),
                  'syaur':(0.428, 0.054, -0.04),
                  'tvul':(2.06,0.22,-0.09 ),
                  'wsgr':(2.30, 0.19, -0.06),
                  'xsgr':(3.17, 0.14, -0.02),
                  'ysgr':(2.13, 0.29, -0.15),
                  'betador':(3.26, 0.14, -0.02),
                  'delceph':(3.71,0.12,-0.01),
                  'etagem':(2.74,0.12,-0.02),
                  'lcar':(2.03,0.16,-0.05)
                 }
    parallaxes = pd.DataFrame.from_dict(parallaxes, orient = 'index', )
    parallaxes.reset_index(inplace=True)
    parallaxes.columns = ['ID', 'parallax', 'p_err', 'LK']
    cepheids = cepheids.merge(parallaxes, on = 'ID', how = 'left')


    #cepheids.fillna({'parallax':1e-03, 'p_err':0, 'LK':0}, inplace = True);
    #cepheids['err'] = sqrt(cepheids.sigma_tot**2 + (cepheids.p_err / cepheids.parallax * 5/log(10))**2)
    return cepheids, sne



### From Cosmoslik

def load_chain(filename):
    """
    Load a chain produced by the CosmoSlik metropolis_hastings sampler. 
    """
    c = cPickle.load(open(filename,'rb'))
    if isinstance(c,(Chain,Chains)): return c
    elif isinstance(c,tuple): return _load_chain_old(filename)
    else:
        with open(filename) as f:
            names = cPickle.load(f)
            dat = []
            while True:
                try:
                    dat.append(cPickle.load(f))
                except:
                    break
            ii=set(i for i,_ in dat)

            if dat[0][1].dtype.kind=='V':
                return Chains([Chain({n:concatenate([d['f%i'%k] for j,d in dat if i==j]) for k,n in enumerate(names)}) for i in ii])
            else:
                return Chains([Chain(dict(zip(names,vstack([d for j,d in dat if i==j]).T))) for i in ii])



