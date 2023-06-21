import sys

# Access the input number from the command-line argument
number = int(sys.argv[1])

# Use the input number in your Python code
print("The input number is:", number)

import numpy as np 
import camb 
from numpy.fft import fftshift
from tqdm import tqdm
import time

import lensit as li
from lensit.clusterlens import lensingmap, profile 
from lensit.misc.misc_utils import gauss_beam
from lensit.ffs_covs import ffs_cov, ell_mat
#from plancklens.wigners import wigners
#from plancklens import n0s, nhl
#from plancklens.n1 import n1

import os
import os.path as op
import matplotlib as mpl
from matplotlib import pyplot as plt
from lensit.pbs import pbs
from scipy.interpolate import UnivariateSpline as spline

mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['font.size'] = 20
mpl.rcParams['figure.figsize'] = 8.5, 5.5

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rc('text', usetex=True)
mpl.rcParams['errorbar.capsize'] = 4
mpl.rc('legend', fontsize=15)

# We start by intiating CAMB which will give us the relevant cosmology 
cambinifile = 'planck_2018_acc'

pars = camb.read_ini(op.join(op.dirname('/home/users/s/sahasay2/git_repos/camb/inifiles'),  'inifiles', cambinifile + '.ini'))
results = camb.get_results(pars)

# We define here the parameters for the profile of the cluster
M200, z = 4 * 1e14, 0.7
profname = 'nfw'
key = "cluster" # "lss"/"cluster"/"lss_plus_cluster"
profparams={'M200c':M200, 'z':z}
hprofile = profile.profile(results, profname)
xmax = 3*hprofile.get_concentration(M200, z)
# Define here the map square patches
npix = 1024  # Number of pixels
lpix_amin = 0.3 # Physical size of a pixel in arcmin (There is bug when <0.2 amin, due to low precision in Cl_TE at )

print("The size of the data patch is %0.1f X %0.1f arcmin central box"%(npix*lpix_amin, npix*lpix_amin))

# Maximum multipole used to generate the CMB maps from the CMB power spectra
# ellmaxsky = 6000 # (bug when ellmax>6300 because of low precision in Cl_TE of CAMB )
ellmaxsky = 5000 

# Set the maximum ell observed in the CMB data maps
ellmaxdat = 5000
ellmindat = 100

# Number of simulated maps 
nsims = 1000

# Set CMB experiment for noise level and beam
cmb_exp='S4_sayan'

# We will cache things in this directory 
libdir = lensingmap.get_cluster_libdir(cambinifile,  profname, key, npix, lpix_amin, ellmaxsky, M200, z, nsims, cmb_exp)
libdir = op.join(libdir,"trunc")
print(libdir)


lmax = ellmaxsky
cpp_fid = results.get_lens_potential_cls(lmax=lmax, raw_cl=True).T[0]

camb_cls = results.get_unlensed_scalar_cls(CMB_unit='muK', raw_cl=True, lmax=lmax).T
cls_unl_fid = {'tt':camb_cls[0], 'ee':camb_cls[1], 'bb':camb_cls[2], 'te':camb_cls[3], 'pp':cpp_fid}

camb_cls_len = results.get_lensed_scalar_cls(CMB_unit='muK', raw_cl=True, lmax=lmax).T
cls_len_fid = {'tt':camb_cls_len[0], 'ee':camb_cls_len[1], 'bb':camb_cls_len[2], 'te':camb_cls_len[3], 'pp':cpp_fid}

np.random.seed(seed=20)
clustermaps = lensingmap.cluster_maps(libdir, key, npix, lpix_amin, nsims, results, profparams, profilename=profname,  ellmax_sky = ellmaxsky, ellmax_data=ellmaxdat, ellmin_data=ellmindat, cmb_exp=cmb_exp)

def cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

def pp_to_kk(ls):
    return ls ** 2 * (ls+1.) ** 2 * 0.25 

def p_to_k(ls):
    return ls * (ls+1.) * 0.5

def kk_to_pp(ls):
    return cli(pp_to_kk(ls))

def k_to_p(ls):
    return cli(p_to_k(ls))

def th_amin_to_el(th_amin):
    th_rd = (th_amin/60)*(np.pi/180)
    return np.pi/th_rd


ellmax_sky = clustermaps.ellmax_sky
sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = li.get_config(clustermaps.cmb_exp)

cls_noise = {'t': (sN_uKamin * np.pi / 180. / 60.) ** 2 * np.ones(clustermaps.ellmax_sky + 1),
            'q':(sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(clustermaps.ellmax_sky + 1),
            'u':(sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(clustermaps.ellmax_sky + 1)}  # simple flat noise Cls
# cl_transf = gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=ellmax_sky)
# lib_alm = ell_mat.ffs_alm_pyFFTW(get_ellmat(LD_res, HD_res=HD_res),
                    # filt_func=lambda ell: (ell >= ellmin) & (ell <= ellmax), num_threads=pyFFTWthreads)
# lib_skyalm = ell_mat.ffs_alm_pyFFTW(clustermaps.ellmat,
                    # filt_func=lambda ell: (ell <= ellmax_sky), num_threads=clustermaps.num_threads)

cl_transf = clustermaps.cl_transf
lib_skyalm = clustermaps.lib_skyalm

typ = 'T'

lib_dir = op.join(clustermaps.dat_libdir, typ)
# isocov = ffs_cov.ffs_diagcov_alm(lib_dir, clustermaps.lib_datalm, clustermaps.cls_unl, cls_len, cl_transf, cls_noise, lib_skyalm=lib_skyalm)
isocov = ffs_cov.ffs_diagcov_alm(lib_dir, clustermaps.lib_datalm, cls_unl_fid, cls_unl_fid, cl_transf, cls_noise, lib_skyalm=lib_skyalm)

ell, = np.where(lib_skyalm.get_Nell()[:ellmaxsky+1])
cpp_prior =  cpp_fid

H0len =  isocov.get_response(typ, lib_skyalm, use_cls_len=True)[0]
#H0len = phi_var_3
def get_starting_point(idx, typ, clustermaps, keyword=None): 
    """
    This returns initial data for simulation index 'idx' from a CMB-S4 simulation library.
    On first call the simulation library will generate all simulations phases, hence might take a little while.
    """ 

    print(" I will be using data from ell=%s to ell=%s only"%(isocov.lib_datalm.ellmin, isocov.lib_datalm.ellmax))
    print(" The sky band-limit is ell=%s"%(isocov.lib_skyalm.ellmax))
 
    lib_qlm = lib_skyalm #: This means we will reconstruct the lensing potential for all unlensed sky modes.
    ellmax_sky = lib_skyalm.ellmax
    ell = np.arange(ellmax_sky+1)

    if typ=='QU':
        datalms1 = np.array([isocov.lib_datalm.map2alm(m) for m in clustermaps.maps_lib.get_sim_qumap(idx)]) 
    elif typ =='T':
        datalms1 = np.array([isocov.lib_datalm.map2alm(clustermaps.maps_lib.get_sim_tmap(idx))]) 
    elif typ =='TQU':
        datalms1 = np.array([isocov.lib_datalm.map2alm(m) for m in np.array([clustermaps.maps_lib.get_sim_tmap(idx), clustermaps.maps_lib.get_sim_qumap(idx)[0], clustermaps.maps_lib.get_sim_qumap(idx)[1]])]) 
   
    use_cls_len = True
 
    plm1 = 0.5 * isocov.get_qlms(typ,  isocov.get_iblms(typ, datalms1, use_cls_len=use_cls_len)[0], lib_qlm, 
                                 use_cls_len=use_cls_len)[0]
    
    # Normalization and Wiener-filtering:
    plmqe1  = lib_qlm.almxfl(plm1, cli(H0len), inplace=False)
    plm0_1  = lib_qlm.almxfl(plm1, cli(H0len + cli(cpp_prior[:lib_qlm.ellmax+1])), inplace=False)

    # We now build the Wiener-filtered quadratic estimator. We use lensed CMB spectra in the weights.

    if typ=='QU':
        datalms2 = np.array([isocov.lib_datalm.map2alm(m) for m in clustermaps.maps_lib.get_sim_qumap_unl(idx)]) 
    elif typ =='T':
        datalms2 = np.array([isocov.lib_datalm.map2alm(clustermaps.maps_lib.get_sim_tmap_unl(idx))]) 
    elif typ =='TQU':
        datalms2 = np.array([isocov.lib_datalm.map2alm(m) for m in np.array([clustermaps.maps_lib.get_sim_tmap_unl(idx), clustermaps.maps_lib.get_sim_qumap_unl(idx)[0], clustermaps.maps_lib.get_sim_qumap_unl(idx)[1]])]) 


    plm2 = 0.5 * isocov.get_qlms(typ,  isocov.get_iblms(typ, datalms2, use_cls_len=use_cls_len)[0], lib_qlm, 
                                 use_cls_len=use_cls_len)[0]
    
    # Normalization and Wiener-filtering:
    plmqe2  = lib_qlm.almxfl(plm2, cli(H0len), inplace=False)
    plm0_2  = lib_qlm.almxfl(plm2, cli(H0len + cli(cpp_prior[:lib_qlm.ellmax+1])), inplace=False)

    return  plm0_1 , plm0_2, datalms1, datalms2

from lensit.ffs_iterators.ffs_iterator_nufft import ffs_iterator_cstMF
from lensit.misc.misc_utils import gauss_beam
from lensit.qcinv import ffs_ninv_filt_ideal_nufft as ffs_ninv_filt_ideal, chain_samples
from lensit.ffs_covs import ell_mat
from lensit.ffs_deflect import ffs_deflect

lib_qlm = lib_skyalm
lib_datalm = clustermaps.lib_datalm

def   get_itlib(lib_dir, plm0, lib_qlm,  datalms, lib_datalm, H0, typ, beam_fwhmamin, NlevT_filt, NlevP_filt, verbose=True):
    """
    This returns an iterator instance from the input data maps, lensing map starting point,
    likelihood curvature guess and choice of filtering parameters (ideally close to those of the data).
    """

    if not os.path.exists(lib_dir): 
        os.makedirs(lib_dir)
    # Prior on lensing power spectrum, and CMB spectra for the filtering at each iteration step.
    cls_unl = clustermaps.cls_unl
    # cpp_prior = np.copy(cpp_true)
    
    
    # lib_skyalm = ell_mat.ffs_alm_pyFFTW(lib_datalm.ell_mat, filt_func=lambda ell:ell<=ellmaxsky)
                            #: This means we perform here the lensing of CMB skies at the same resolution 
                            #  than the data with the band-limit of 6000.
    lib_skyalm = clustermaps.lib_skyalm
    # transf = gauss_beam(beam_fwhmamin / 180. / 60. * np.pi, lmax=ellmaxsky) #: fiducial beam
    
    # Anisotropic filtering instance, with unlensed CMB spectra as inputs. Delfections will be added by the iterator.
    cl_transf = gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=lib_skyalm.ellmax)
    filt = ffs_ninv_filt_ideal.ffs_ninv_filt(lib_datalm, lib_skyalm, cls_unl_fid, cl_transf, NlevT_filt, NlevP_filt)
    
     # Anisotropic filtering instance, with unlensed CMB spectra as inputs. Delfections will be added by the iterator.
    f_id = ffs_deflect.ffs_id_displacement(lib_skyalm.shape, lib_skyalm.lsides)
    filt = ffs_ninv_filt_ideal.ffs_ninv_filt_wl(lib_datalm, lib_skyalm, cls_unl, cl_transf, NlevT_filt,NlevP_filt, f=f_id)
    
    # Description of the multi-grid chain to use: (here the simplest, diagonal pre-conditioner) 
    chain_descr = chain_samples.get_isomgchain(filt.lib_skyalm.ellmax, filt.lib_datalm.shape,
                                                        tol=1e-6,iter_max=200)
    
    # We assume no primordial B-modes, the E-B filtering will assume all B-modes are either noise or lensing:
    opfilt =  li.qcinv.opfilt_cinv_noBB
    opfilt._type = typ 
    
    # With all this now in place, we can build the iterator instance:
    #iterator = ffs_iterator_pertMF(lib_dir, typ, filt, datalms, lib_qlm, 
    #          plm0, H0, cpp_prior, chain_descr=chain_descr, opfilt=opfilt, verbose=verbose)
    iterator = ffs_iterator_cstMF(lib_dir, typ, filt, datalms, lib_qlm, 
              plm0, H0, plm0 * 0, cpp_prior, chain_descr=chain_descr, opfilt=opfilt, verbose=True, nufft_epsilon=1e-7)
               # We use here an iterator instance that uses an analytical approximation 
               # for the mean-field term at each step.
    return iterator

# lib_dir = os.path.join(os.environ['LENSIT'], 'temp', 'iterator_S4_sim%03d'%0)

itlibdir_1 = lambda idx: op.join(lib_dir, f'iterator_S4_sim{idx:04d}')
itlibdir_2 = lambda idx: op.join(lib_dir, f'iterator_unl_S4_sim{idx:04d}')

nmaps = 50

# number of iteration
nit = 20

fail_1 = []
fail_2 = []
if nsims >1:
    for idx in range(number*nmaps, (number+1)*nmaps):
        print("doing the iterative estimator for map number %i"%idx)
        plm0_1, plm0_2,  datalms_1, datalms_2 = get_starting_point(idx, typ, clustermaps)
        try:
            itlibs_1 = get_itlib(itlibdir_1(idx), plm0_1, lib_qlm, datalms_1, lib_datalm, H0len, typ, beam_fwhmamin=Beam_FWHM_amin, NlevT_filt=sN_uKamin, NlevP_filt=sN_uKaminP, verbose=False)
            itlibs_1.soltn_cond = True
            for i in range(nit):
                itlibs_1.iterate(i, 'p')
        except:
            fail_1 = np.append(fail_1, idx)
        try:
            itlibs_2 = get_itlib(itlibdir_2(idx), plm0_2, lib_qlm, datalms_2, lib_datalm, H0len, typ, beam_fwhmamin=Beam_FWHM_amin, NlevT_filt=sN_uKamin, NlevP_filt=sN_uKaminP, verbose=False)
            itlibs_2.soltn_cond = True
            for i in range(nit):
                itlibs_2.iterate(i, 'p')
        except:
            fail_2 = np.append(fail_2, idx)

        
#if nsims >1:
#    for idx in range(number*nmaps, (number+1)*nmaps):
#        print("doing the iterative estimator for map number %i"%idx)
#        plm0s, plmqes,  klm0, klmqe, H0len, wf_qe, datalms = get_starting_point(idx, typ, clustermaps)
#        itlibs = get_itlib(itlibdir(idx), plm0s, lib_qlm, datalms, lib_datalm, H0len, typ, beam_fwhmamin=Beam_FWHM_amin, NlevT_filt=sN_uKamin, NlevP_filt=sN_uKaminP, verbose=False)
#        itlibs.soltn_cond = True
#        for i in range(10):
#            itlibs.iterate(i, 'p')
            

print(fail_1)
print(fail_2)
end = time.time()