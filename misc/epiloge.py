import numpy as np 
import camb 
from numpy.fft import fftshift
from tqdm import tqdm

import lensit as li
from lensit.clusterlens import lensingmap, profile 
from lensit.misc.misc_utils import gauss_beam
from lensit.ffs_covs import ffs_cov, ell_mat

import os
import os.path as op
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from scipy.interpolate import UnivariateSpline as spline

mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['font.size'] = 20
mpl.rcParams['figure.figsize'] = 17, 11

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

nmaps = 1000
plm0_1 = [None]*nmaps
plm0_2 = [None]*nmaps
datalms_1 = [None]*nmaps
datalms_2 = [None]*nmaps

if nsims >1:
    for idx in range(nmaps):
        print(idx)
        plm0_1[idx], plm0_2[idx], datalms_1[idx], datalms_2[idx] = get_starting_point(idx, typ, clustermaps)
        
plmits_1 = [None]*nmaps
itlibs_1 = [None]*nmaps
plmits_2 = [None]*nmaps
itlibs_2 = [None]*nmaps
nit = 20
fail = []
if nsims >1:
    for idx in range(nmaps):
        try:
            print("doing the iterative estimator for map number %i"%idx)
            itlibs_1[idx] = get_itlib(itlibdir_1(idx), plm0_1[idx], lib_qlm, datalms_1[idx], lib_datalm, H0len, typ, beam_fwhmamin=Beam_FWHM_amin, NlevT_filt=sN_uKamin, NlevP_filt=sN_uKaminP, verbose=False)
            itlibs_1[idx].soltn_cond = True
            for i in range(nit):
                itlibs_1[idx].iterate(i, 'p')
            plmits_1[idx]= itlibs_1[idx].get_Plm(nit-1, 'p')
            # Doing the same for the unlensed maps
            itlibs_2[idx] = get_itlib(itlibdir_2(idx), plm0_2[idx], lib_qlm, datalms_2[idx], lib_datalm, H0len, typ, beam_fwhmamin=Beam_FWHM_amin, NlevT_filt=sN_uKamin, NlevP_filt=sN_uKaminP, verbose=False)
            itlibs_2[idx].soltn_cond = True
            for i in range(nit):
                itlibs_2[idx].iterate(i, 'p')
            plmits_2[idx]= itlibs_2[idx].get_Plm(nit-1, 'p')
            pass
        except:
            fail = np.append(fail, idx)
            continue

print(fail)

wl = np.loadtxt("../files/normalisation_cstMF_lmax5k_lmin100_lmaxout5k_nit20_sims1k.dat")

for idx in range(nmaps):
    plmits_1[idx] = lib_skyalm.almxfl(plmits_1[idx], cli(wl))
    plmits_2[idx] = lib_skyalm.almxfl(plmits_2[idx], cli(wl))
    
plmits = nmaps*[None]
for idx in range(nmaps):
    plmits[idx] = plmits_1[idx] - plmits_2[idx]
    
plmqe_mean = np.mean(plmits, axis=0)

ell = np.arange(ellmax_sky+1)
kappa_l_mean = lib_skyalm.bin_realpart_inell(plmqe_mean)*p_to_k(ell)
kappa_lm_mean = lib_skyalm.almxfl(plmqe_mean, p_to_k(ell), inplace=False)

lbox_amin = npix*lpix_amin #Physical size of the box in arcmin
lbox_rad = (lbox_amin/60)*(np.pi/180)
kappa_map = clustermaps.len_cmbs.kappa_map
from scipy.interpolate import UnivariateSpline

kappa_ell_bin = lib_skyalm.bin_realpart_inell(lib_skyalm.map2alm(kappa_map))[ell]
kappa_ell_lensit = UnivariateSpline(ell, kappa_ell_bin, s=0)
kappa_l = kappa_ell_lensit(ell)

#################### Now the calculation with QE #############################

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
        datalms = np.array([isocov.lib_datalm.map2alm(m) for m in clustermaps.maps_lib.get_sim_qumap(idx)]) 
    elif typ =='T':
        datalms = np.array([isocov.lib_datalm.map2alm(clustermaps.maps_lib.get_sim_tmap(idx))]) 
    elif typ =='TQU':
        datalms = np.array([isocov.lib_datalm.map2alm(m) for m in np.array([clustermaps.maps_lib.get_sim_tmap(idx), clustermaps.maps_lib.get_sim_qumap(idx)[0], clustermaps.maps_lib.get_sim_qumap(idx)[1]])]) 
   
    use_cls_len = True
 
    plm1 = 0.5 * isocov.get_qlms(typ,  isocov.get_iblms(typ, datalms, use_cls_len=use_cls_len)[0], lib_qlm, 
                                 use_cls_len=use_cls_len)[0]
    
    # Normalization and Wiener-filtering:
    plmqe1  = lib_qlm.almxfl(plm1, cli(H0len), inplace=False)

    # We now build the Wiener-filtered quadratic estimator. We use lensed CMB spectra in the weights.

    if typ=='QU':
        datalms = np.array([isocov.lib_datalm.map2alm(m) for m in clustermaps.maps_lib.get_sim_qumap_unl(idx)]) 
    elif typ =='T':
        datalms = np.array([isocov.lib_datalm.map2alm(clustermaps.maps_lib.get_sim_tmap_unl(idx))]) 
    elif typ =='TQU':
        datalms = np.array([isocov.lib_datalm.map2alm(m) for m in np.array([clustermaps.maps_lib.get_sim_tmap_unl(idx), clustermaps.maps_lib.get_sim_qumap_unl(idx)[0], clustermaps.maps_lib.get_sim_qumap_unl(idx)[1]])]) 


    plm2 = 0.5 * isocov.get_qlms(typ,  isocov.get_iblms(typ, datalms, use_cls_len=use_cls_len)[0], lib_qlm, 
                                 use_cls_len=use_cls_len)[0]
    
    # Normalization and Wiener-filtering:
    plmqe2  = lib_qlm.almxfl(plm2, cli(H0len), inplace=False)
    
    plm = 0.5 * isocov.get_qlms(typ,  isocov.get_iblms(typ, datalms, use_cls_len=use_cls_len)[0], lib_qlm, 
                                 use_cls_len=use_cls_len, ellmax_gradleg=2000)[0]

    return  plmqe1 , plmqe2, plm

plmqes1 = [None]*nmaps
plmqes2 = [None]*nmaps
plmqes = [None]*nmaps
plms = [None]*nmaps

if nsims >1:
    for idx in range(nmaps):
        print(idx)
        plmqes1[idx], plmqes2[idx], plms[idx] = get_starting_point(idx, typ, clustermaps)
        plmqes[idx] = plmqes1[idx] - plmqes2[idx]
        
plmqe_mean_1 = np.mean(plmqes, axis=0)
ell = np.arange(ellmax_sky+1)
kappa_l_mean_1 = lib_skyalm.bin_realpart_inell(plmqe_mean_1)*p_to_k(ell)
kappa_lm_mean_1 = lib_skyalm.almxfl(plmqe_mean_1, p_to_k(ell), inplace=False)

phi_l_3 = nmaps*[None]
for idx in tqdm(range(nmaps)):
    phi_l_3[idx] = lib_skyalm.bin_realpart_inell(plms[idx])

num_l = clustermaps.lib_skyalm.get_Nell()
phi_var_3 = num_l*np.var(phi_l_3, axis=0)
H0len_1 = phi_var_3

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
        datalms = np.array([isocov.lib_datalm.map2alm(m) for m in clustermaps.maps_lib.get_sim_qumap(idx)]) 
    elif typ =='T':
        datalms = np.array([isocov.lib_datalm.map2alm(clustermaps.maps_lib.get_sim_tmap(idx))]) 
    elif typ =='TQU':
        datalms = np.array([isocov.lib_datalm.map2alm(m) for m in np.array([clustermaps.maps_lib.get_sim_tmap(idx), clustermaps.maps_lib.get_sim_qumap(idx)[0], clustermaps.maps_lib.get_sim_qumap(idx)[1]])]) 
   
    use_cls_len = True
 
    plm1 = 0.5 * isocov.get_qlms(typ,  isocov.get_iblms(typ, datalms, use_cls_len=use_cls_len)[0], lib_qlm, 
                                 use_cls_len=use_cls_len, ellmax_gradleg=2000)[0]
    
    # Normalization and Wiener-filtering:
    plmqe1  = lib_qlm.almxfl(plm1, cli(H0len_1), inplace=False)

    # We now build the Wiener-filtered quadratic estimator. We use lensed CMB spectra in the weights.

    if typ=='QU':
        datalms = np.array([isocov.lib_datalm.map2alm(m) for m in clustermaps.maps_lib.get_sim_qumap_unl(idx)]) 
    elif typ =='T':
        datalms = np.array([isocov.lib_datalm.map2alm(clustermaps.maps_lib.get_sim_tmap_unl(idx))]) 
    elif typ =='TQU':
        datalms = np.array([isocov.lib_datalm.map2alm(m) for m in np.array([clustermaps.maps_lib.get_sim_tmap_unl(idx), clustermaps.maps_lib.get_sim_qumap_unl(idx)[0], clustermaps.maps_lib.get_sim_qumap_unl(idx)[1]])]) 


    plm2 = 0.5 * isocov.get_qlms(typ,  isocov.get_iblms(typ, datalms, use_cls_len=use_cls_len)[0], lib_qlm, 
                                 use_cls_len=use_cls_len, ellmax_gradleg=2000)[0]
    
    # Normalization and Wiener-filtering:
    plmqe2  = lib_qlm.almxfl(plm2, cli(H0len_1), inplace=False)

    return  plmqe1 , plmqe2

plmqes1 = [None]*nmaps
plmqes2 = [None]*nmaps
plmqes_1 = [None]*nmaps

if nsims >1:
    for idx in range(nmaps):
        print(idx)
        plmqes1[idx], plmqes2[idx]= get_starting_point(idx, typ, clustermaps)
        plmqes_1[idx] = plmqes1[idx] - plmqes2[idx]
        
        
plmqe_mean_2 = np.mean(plmqes_1, axis=0)
ell = np.arange(ellmax_sky+1)
kappa_l_mean_2 = lib_skyalm.bin_realpart_inell(plmqe_mean_2)*p_to_k(ell)
kappa_lm_mean_2 = lib_skyalm.almxfl(plmqe_mean_2, p_to_k(ell), inplace=False)

from scipy.special import jv
def kappa_th_bessel(theta_amin, kappa_l, elmax):
    el_r, = np.where(isocov.lib_skyalm.get_Nell()[:elmax+1])
    el_range = el_r[1:]
    theta_rad = theta_amin * np.pi / 60 / 180
    kappa_el = kappa_l[el_range]
    bessel = jv(0, el_range*theta_rad)
    integrand = el_range*kappa_el*bessel
    kappa_el1 = np.trapz(integrand, el_range)/ 2 / np.pi 
    return kappa_el1

ell = np.arange(ellmax_sky+1)
theta = np.linspace(0, 10, num=100)
nmaps = 1000
kappa_l_ = nmaps*[None]
kappa_l_1 = nmaps*[None]
kappa_l_2 = nmaps*[None]
kappa_prof_th_ = nmaps*[None]
kappa_prof_th_1 = nmaps*[None]
kappa_prof_th_2 = nmaps*[None]
for idx in tqdm(range(nmaps)):
    kappa_l_[idx] = lib_skyalm.bin_realpart_inell(plmits[idx])*p_to_k(ell)
    kappa_l_1[idx] = lib_skyalm.bin_realpart_inell(plmqes[idx])*p_to_k(ell)
    kappa_l_2[idx] = lib_skyalm.bin_realpart_inell(plmqes_1[idx])*p_to_k(ell)
    kappa_th_ = np.zeros((len(theta)))
    kappa_th_1 = np.zeros((len(theta)))
    kappa_th_2 = np.zeros((len(theta)))
    for th in range(len(theta)):
        kappa_th_[th] = kappa_th_bessel(theta[th], kappa_l_[idx], elmax=ellmaxsky)
        kappa_th_1[th] = kappa_th_bessel(theta[th], kappa_l_1[idx], elmax=ellmaxsky)
        kappa_th_2[th] = kappa_th_bessel(theta[th], kappa_l_2[idx], elmax=ellmaxsky)
        
    kappa_prof_th_[idx] = kappa_th_
    kappa_prof_th_1[idx] = kappa_th_1
    kappa_prof_th_2[idx] = kappa_th_2
    
kappa_prof_th_mean = np.mean(kappa_prof_th_, axis=0)*lbox_rad
kappa_prof_th_error = np.std(kappa_prof_th_, axis=0)*lbox_rad/np.sqrt(1000)
kappa_prof_th_mean_1 = np.mean(kappa_prof_th_1, axis=0)*lbox_rad
kappa_prof_th_error_1 = np.std(kappa_prof_th_1, axis=0)*lbox_rad/np.sqrt(1000)
kappa_prof_th_mean_2 = np.mean(kappa_prof_th_2, axis=0)*lbox_rad
kappa_prof_th_error_2 = np.std(kappa_prof_th_2, axis=0)*lbox_rad/np.sqrt(1000)

#################### Plotting ####################

##### kappa_l ########
ell, = np.where(isocov.lib_skyalm.get_Nell()[:ellmax_sky+1])
el = ell[1:]

plt.plot(el, kappa_l_mean[el], label="MAP")
plt.plot(el, kappa_l_mean_2[el], label="QE with cut in the gradient leg")
plt.plot(el, kappa_l_mean_1[el], label="QE without cut")
plt.plot(el, kappa_l[el], c="r", label=r'$\kappa_l^{input}$')
plt.semilogy()
plt.xlabel("$\ell$")
plt.ylabel("$\kappa_\ell$")
plt.legend()
plt.savefig("../plots/kappa_l.pdf")

##### kappa_theta ########
# Define the plot styles
styles = {
    'MAP': {
        'color': 'red',
        'label': 'Without the cut',
        'fill_color': 'red'
    },
    'QE_without_cut': {
        'color': 'blue',
        'label': 'With the cut',
        'fill_color': 'blue'
    },
    'QE_with_cut': {
        'color': 'green',
        'label': 'With the cut',
        'fill_color': 'green'
    },    
    'input_kappa': {
        'color': 'black',
        'label': r'Input $\kappa (\theta)$'
    }
}

fig, ax = plt.subplots()

# Plot with MAP
line1, = ax.plot(theta, kappa_prof_th_mean, color=styles['MAP']['color'])
error1 = ax.fill_between(theta, kappa_prof_th_mean - kappa_prof_th_error, kappa_prof_th_mean + kappa_prof_th_error,
                         alpha=0.2, color=styles['MAP']['fill_color'])

 #Plot with QE with cut
line2, = ax.plot(theta, kappa_prof_th_mean_1, color=styles['QE_without_cut']['color'])
error2 = ax.fill_between(theta, kappa_prof_th_mean_1 - kappa_prof_th_error_1, kappa_prof_th_mean_1 + kappa_prof_th_error_1,
                         alpha=0.2, color=styles['QE_without_cut']['fill_color'])

# Plot with QE without cut
line3, = ax.plot(theta, kappa_prof_th_mean_2, color=styles['QE_with_cut']['color'])
error3 = ax.fill_between(theta, kappa_prof_th_mean_2 - kappa_prof_th_error_2, kappa_prof_th_mean_2 + kappa_prof_th_error_2,
                         alpha=0.2, color=styles['QE_with_cut']['fill_color'])

# Plot input kappa
line4, = ax.plot(theta, kappa_thet_bessel * lbox_rad, color=styles['input_kappa']['color'])

# Set labels and title
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.set_ylabel(r'Stacked $\kappa(\theta)$')
ax.set_title(r'$M_{200} = 4 \times 10^{14}, z=0.7, \ell_{max}^{CMB}=5000, \ell_{min}^{CMB}=100, \ell_{max}^{out}=6000$',
             y=1.08)

# Create custom legend handles for shaded regions with lines
error_patch1 = mpatches.Patch(color=styles['MAP']['fill_color'], alpha=error1.get_alpha(), label='Error')
error_patch2 = mpatches.Patch(color=styles['QE_without_cut']['fill_color'], alpha=error2.get_alpha(), label='Error')
error_patch3 = mpatches.Patch(color=styles['QE_with_cut']['fill_color'], alpha=error3.get_alpha(), label='Error')

# Show the legend with custom handles
ax.legend([(line1, error_patch1), (line2, error_patch2),(line3, error_patch3), line4], ['MAP','QE Without the cut','QE With the cut', r'Input $\kappa (\theta)$'])

# Show the plot
#plt.show()
plt.savefig("../plots/kappa_thet_.pdf")


fig, ax = plt.subplots()

# Plot with MAP
line1, = ax.plot(theta, kappa_prof_th_mean, color=styles['MAP']['color'])
#error1 = ax.fill_between(theta, kappa_prof_th_mean - kappa_prof_th_error, kappa_prof_th_mean + kappa_prof_th_error,
#                         alpha=0.2, color=styles['MAP']['fill_color'])

 #Plot with QE with cut
line2, = ax.plot(theta, kappa_prof_th_mean_1, color=styles['QE_without_cut']['color'])
#error2 = ax.fill_between(theta, kappa_prof_th_mean_1 - kappa_prof_th_error_1, kappa_prof_th_mean_1 + kappa_prof_th_error_1,
#                         alpha=0.2, color=styles['QE_without_cut']['fill_color'])

# Plot with QE without cut
line3, = ax.plot(theta, kappa_prof_th_mean_2, color=styles['QE_with_cut']['color'])
#error3 = ax.fill_between(theta, kappa_prof_th_mean_2 - kappa_prof_th_error_2, kappa_prof_th_mean_2 + kappa_prof_th_error_2,
#                         alpha=0.2, color=styles['QE_with_cut']['fill_color'])

# Plot input kappa
line4, = ax.plot(theta, kappa_thet_bessel * lbox_rad, color=styles['input_kappa']['color'])

# Set labels and title
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.set_ylabel(r'Stacked $\kappa(\theta)$')
ax.set_title(r'$M_{200} = 4 \times 10^{14}, z=0.7, \ell_{max}^{CMB}=5000, \ell_{min}^{CMB}=100, \ell_{max}^{out}=6000$',
             y=1.08)

# Create custom legend handles for shaded regions with lines
error_patch1 = mpatches.Patch(color=styles['MAP']['fill_color'], alpha=error1.get_alpha(), label='Error')
error_patch2 = mpatches.Patch(color=styles['QE_without_cut']['fill_color'], alpha=error2.get_alpha(), label='Error')
error_patch3 = mpatches.Patch(color=styles['QE_with_cut']['fill_color'], alpha=error3.get_alpha(), label='Error')

# Show the legend with custom handles
ax.legend([(line1, error_patch1), (line2, error_patch2),(line3, error_patch3), line4], ['MAP','QE Without the cut','QE With the cut', r'Input $\kappa (\theta)$'])

# Show the plot
#plt.show()
plt.savefig("../plots/kappa_thet_without_error.pdf")