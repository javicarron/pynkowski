import numpy as np
import healpy as hp
from .base_da import DataField
from .utils_da import get_theta, healpix_derivatives, healpix_second_derivatives
from ..stats.minkowski import _MF_prefactor
try:
    from tqdm.auto import tqdm
except:
    tqdm = lambda x: x
    print('tqdm not loaded')



class SO3Healpix(DataField):
    """Class for spherical spin fields in the SO(3) formalism, with Q and U in HEALPix format.

    Parameters
    ----------
    Q : np.array
        Values of the Q field in HEALPix format in RING scheme.
    
    U : np.array
        Values of the U field in HEALPix format in RING scheme. Must have the same shape as `Q`.
        
    normalise : bool, optional
        If `True`, the map is normalised to have unit variance.
        Default: `True`.

    mask : np.array or None, optional
        Sky mask where the field is considered. It is a bool array of the same shape that `Q` and `U`.
        Default: all data is included.
        
    Attributes
    ----------
    field : np.array
        Array of shape `(2, Q.shape)` containing the Q and U values. It can be called as `field(psi)` to obtain the value of $f$ for the given `psi`.
        
    nside : int
        Parameter `nside` of the Q and U maps. The number of pixels is `12*nside**2`.
    
    dim : int
        Dimension of the space where the field is defined. In this case, the space is SO(3) and this is 3.
        
    name : str
        Name of the field. In this case, it is `"SO(3) HEALPix map"`.
    
    first_der : list or None
        First **covariant** derivatives of the field in an orthonormal basis of the space. It a list of size `3`, each entry has the same structure as `field`.
    
    second_der : list or None
        Second **covariant** derivatives of the field in an orthonormal basis of the space. It a list of size `6`, each entry has the same structure as `field`.
        The order of the derivatives is diagonal first, i.e., `11`, `22`, `33`, `12`, `13`, `23`.
        
    mask : np.array
        Sky mask where the field is considered. It is a bool array of the same shape that `Q` and `U`.

    """   
    def __init__(self, Q, U, normalise=True, mask=None):
        assert Q.shape == U.shape, "Q and U must have the same shape"
        field = QUarray(Q, U)
        if mask is None:
            mask = np.ones(Q.shape, dtype=bool)
        super().__init__(field, 3, name="SO(3) HEALPix map", mask=mask)
        self.nside = hp.get_nside(self.field[0])
        if hp.get_nside(self.mask) != self.nside:
            raise ValueError('The map and the mask have different nside')
        
        if normalise:
            σ2 = self.get_variance()
            self.field /= np.sqrt(σ2)
            
    def get_variance(self):
        """Compute the variance of the SO(3) Healpix map within the sky mask. 

        Returns
        -------
        var : float
            The variance of the map within the mask.

        """    
        return (np.mean(self.field[:,self.mask]**2.))
    
    def get_first_der(self, lmax=None):
        """Compute the covariant first derivatives of the SO(3) Healpix map. 
        It stores:
        
        - first covariant derivative wrt e₁ in self.first_der[0]
        - first covariant derivative wrt e₂ in self.first_der[1]
        - first covariant derivative wrt e₃ in self.first_der[2]
        
        Parameters
        ----------
        lmax : int or None, optional
            Maximum multipole used in the computation of the derivatives.
            Default: 3*nside - 1

        """    
        Q_grad = healpix_derivatives(self.field[0], gradient=True, lmax=lmax)  # order θ,ϕ
        U_grad = healpix_derivatives(self.field[1], gradient=True, lmax=lmax)
        theta = get_theta(self.nside)

        self._fphi = QUarray(Q_grad[1], U_grad[1])
        ftheta = QUarray(Q_grad[0], U_grad[0])
        fpsi = self.field.derpsi()

        factor_p = (np.sqrt(1.-np.sin(theta)) + np.sqrt(1.+np.sin(theta))) / np.sqrt(8.)
        factor_m = (np.sqrt(1.-np.sin(theta)) - np.sqrt(1.+np.sin(theta))) / np.sqrt(8.)

        self.first_der = [ factor_p * self._fphi + factor_m * fpsi / np.cos(theta),         # ∂e₁
                           ftheta / np.sqrt(2.),                                            # ∂e₂
                           factor_m * self._fphi + factor_p * fpsi / np.cos(theta) ]        # ∂e₃

        
        # This is the old way, I leave it here for reference and debugging if needed
        # self.grad_theta = Polmap(Q_grad[0]/np.sqrt(2.), U_grad[0]/np.sqrt(2.), normalise=False)
        # self.grad_phi = Polmap(((np.sqrt(1.-np.sin(theta)) + np.sqrt(1.+np.sin(theta)) ) / np.sqrt(8.) ) * Q_grad[1] -
        #                           ((np.sqrt(1.-np.sin(theta)) - np.sqrt(1.+np.sin(theta)) ) / (np.sqrt(8.)*np.cos(theta)) ) * 2.*self.U, 
        #                        ((np.sqrt(1.-np.sin(theta)) + np.sqrt(1.+np.sin(theta)) ) / np.sqrt(8.) ) * U_grad[1] +
        #                           ((np.sqrt(1.-np.sin(theta)) - np.sqrt(1.+np.sin(theta)) ) / (np.sqrt(8.)*np.cos(theta)) ) * 2.*self.Q, normalise=False)
        
        # self.grad_psi = Polmap(((np.sqrt(1.-np.sin(theta)) - np.sqrt(1.+np.sin(theta)) ) / np.sqrt(8.) ) * Q_grad[1] -
        #                           ((np.sqrt(1.-np.sin(theta)) + np.sqrt(1.+np.sin(theta)) ) / (np.sqrt(8.)*np.cos(theta)) ) * 2.*self.U, 
        #                        ((np.sqrt(1.-np.sin(theta)) - np.sqrt(1.+np.sin(theta)) ) / np.sqrt(8.) ) * U_grad[1] +
        #                           ((np.sqrt(1.-np.sin(theta)) + np.sqrt(1.+np.sin(theta)) ) / (np.sqrt(8.)*np.cos(theta)) ) * 2.*self.Q, normalise=False)
        
        # self.der_phi = Polmap(np.cos(theta) * Q_grad[1], np.cos(theta) * U_grad[1], normalise=False)


        # self.first_der = np.array(healpix_derivatives(self.field, gradient=True, lmax=lmax))  # order: θ, ϕ
            
    def get_second_der(self, lmax=None):
        """Compute the covariant second derivatives of the input Healpix scalar map. 
        It stores:
        
        - second covariant derivative wrt θθ in self.second_der[0]
        - second covariant derivative wrt ϕϕ in self.second_der[1]
        - second covariant derivative wrt θϕ in self.second_der[2]

        Parameters
        ----------
        lmax : int or None, optional
            Maximum multipole used in the computation of the derivatives.
            Default: 3*nside - 1

        """    
        if self.first_der is None:
            self.get_first_der()
        theta = get_theta(self.nside)

        # self._fphi  is already computed in get_first_der
        self._fphi *= np.cos(theta)
        ftheta = self.first_der[1] * np.sqrt(2.)
        fpsi = self.field.derpsi()

        Q_der_der = healpix_second_derivatives(ftheta[0], self._fphi[0], lmax=lmax)  # order θθ, ϕϕ, θϕ
        U_der_der = healpix_second_derivatives(ftheta[1], self._fphi[1], lmax=lmax)
        fthetatheta = QUarray(Q_der_der[0], U_der_der[0])
        fphiphi = QUarray(Q_der_der[1], U_der_der[1])
        fthetaphi = QUarray(Q_der_der[2], U_der_der[2])
        del Q_der_der, U_der_der

        factor_P = (np.sqrt(1.-np.sin(theta)) + np.sqrt(1.+np.sin(theta))) 
        factor_M = (np.sqrt(1.-np.sin(theta)) - np.sqrt(1.+np.sin(theta)))

        # order 11, 22, 33, 12, 13, 23
        self.second_der = [ ((1.+np.cos(theta))*fphiphi + (1.-np.cos(theta))*fpsi.derpsi() - 2.*np.sin(theta)*self._fphi.derpsi() - np.sin(2.*theta)*ftheta/2.) / (4.*np.cos(theta)**2.),
                            fthetatheta / 2.,
                            ((1.-np.cos(theta))*fphiphi + (1.+np.cos(theta))*fpsi.derpsi() - 2.*np.sin(theta)*self._fphi.derpsi() - np.sin(2.*theta)*ftheta/2.) / (4.*np.cos(theta)**2.),
                            (2.*np.cos(theta)*(factor_M*ftheta.derpsi() + factor_P*fthetaphi) + (factor_P*np.sin(theta) - factor_M)*self._fphi + (factor_M*np.sin(theta) - factor_P)*fpsi ) / (8.*np.cos(theta)**2.),
                            (-np.sin(theta)*fphiphi - np.sin(theta)*fpsi.derpsi() + 2.*self._fphi.derpsi() + np.cos(theta)*ftheta ) / (4.*np.cos(theta)**2.),
                            (2.*np.cos(theta)*(factor_M*fthetaphi + factor_P*ftheta.derpsi()) + (factor_M*np.sin(theta) - factor_P)*self._fphi + (factor_P*np.sin(theta) - factor_M)*fpsi ) / (8.*np.cos(theta)**2.) ]

        
        # This is the old way, I leave it here for reference and debugging if needed
        # self.der_theta_theta = Polmap(Q_der_der[0]/2., U_der_der[0]/2., normalise=False)
        
        
        # self.der_phi_phi = Polmap( ((1.+np.cos(theta)) * Q_der_der[1]  - (1-np.cos(theta)) * 4. * self.Q  + 4.*np.sin(theta)*U_der[1] - np.sin(2.*theta)*Q_der[0]/2. ) / (4.*np.cos(theta)**2.),
        #                            ((1.+np.cos(theta)) * U_der_der[1]  - (1-np.cos(theta)) * 4. * self.U  - 4.*np.sin(theta)*Q_der[1] - np.sin(2.*theta)*U_der[0]/2. ) / (4.*np.cos(theta)**2.), normalise=False)
            
        # self.der_psi_psi = Polmap( ((1.-np.cos(theta)) * Q_der_der[1]  - (1+np.cos(theta)) * 4. * self.Q  + 4.*np.sin(theta)*U_der[1] - np.sin(2.*theta)*Q_der[0]/2. ) / (4.*np.cos(theta)**2.),
        #                            ((1.-np.cos(theta)) * U_der_der[1]  - (1+np.cos(theta)) * 4. * self.U  - 4.*np.sin(theta)*Q_der[1] - np.sin(2.*theta)*U_der[0]/2. ) / (4.*np.cos(theta)**2.), normalise=False)
            
        # P = np.sqrt(1.-np.sin(theta)) + np.sqrt(1.+np.sin(theta))
        # M = np.sqrt(1.-np.sin(theta)) - np.sqrt(1.+np.sin(theta))
        
        # self.der_theta_phi = Polmap( (2.*np.cos(theta)* (-2.*M*U_der[0] + P*Q_der_der[2]) + (P*np.sin(theta)-M)*Q_der[1] - 2.*(M*np.sin(theta)-P)*self.U ) / (8.*np.cos(theta)**2.),
        #                              (2.*np.cos(theta)* ( 2.*M*Q_der[0] + P*U_der_der[2]) + (P*np.sin(theta)-M)*U_der[1] + 2.*(M*np.sin(theta)-P)*self.Q ) / (8.*np.cos(theta)**2.), normalise=False)
            
        # self.der_theta_psi = Polmap( (2.*np.cos(theta)* (-2.*P*U_der[0] + M*Q_der_der[2]) + (M*np.sin(theta)-P)*Q_der[1] - 2.*(P*np.sin(theta)-M)*self.U ) / (8.*np.cos(theta)**2.),
        #                              (2.*np.cos(theta)* ( 2.*P*Q_der[0] + M*U_der_der[2]) + (M*np.sin(theta)-P)*U_der[1] + 2.*(P*np.sin(theta)-M)*self.Q ) / (8.*np.cos(theta)**2.), normalise=False)
        # # self.der_theta_psi = Polmap( (2.*np.cos(theta)* (-2.*P*U_der[0] + M*Q_der_der[2]) + ( 3.*M*np.sin(theta) - P*(1.+np.cos(theta)) )*Q_der[1] - 2.*(3.*P*np.sin(theta) - M*(1.-np.cos(theta)))*self.U ) / (8.*np.cos(theta)**2.),
        # #                              (2.*np.cos(theta)* ( 2.*P*Q_der[0] + M*U_der_der[2]) + ( 3.*M*np.sin(theta) - P*(1.+np.cos(theta)) )*U_der[1] + 2.*(3.*P*np.sin(theta) - M*(1.-np.cos(theta)))*self.Q ) / (8.*np.cos(theta)**2.), normalise=False)
            
        # self.der_phi_psi = Polmap( (-np.sin(theta) * Q_der_der[1]  + np.sin(theta) * 4. * self.Q  - 4.*U_der[1] + np.cos(theta)*Q_der[0] ) / (4.*np.cos(theta)**2.),
        #                            (-np.sin(theta) * U_der_der[1]  + np.sin(theta) * 4. * self.U  + 4.*Q_der[1] + np.cos(theta)*U_der[0] ) / (4.*np.cos(theta)**2.), normalise=False)

    def _V0(self, us, dus, verbose=True):
        """Compute the first Minkowski Functional, $V_0$, normalized by the volume of the space. Internal interface for `pynkowski.V0`.

        Parameters
        ----------
        us : np.array
            The thresholds where $V_0$ is computed.

        dus : np.array
            The bin sizes of `us`. Ignored.

        verbose : bool, optional
            If True (default), progress bars are shown for the computations on data.
        
        Returns
        -------
        V0 : np.array()
            The values of the first Minkowski Functional at the given thresholds.
        
        """
        stat = np.zeros_like(us)
        modulus = self.field.modulus()

        for ii in tqdm(np.arange(us.shape[0]), disable=not verbose):
            lenghts = np.zeros_like(modulus)
            lenghts[us[ii]<=-modulus] = np.pi
            mask = (us[ii]>-modulus) & (us[ii]<modulus)
            lenghts[mask] = np.arccos(us[ii]/modulus[mask])
            stat[ii] = np.mean(lenghts[self.mask])/np.pi
        return stat
    
    def _V1(self, us, dus, verbose=True):
        """Compute the second Minkowski Functional, $V_1$, normalized by the volume of the space. Internal interface for `pynkowski.V1`.

        Parameters
        ----------
        us : np.array
            The thresholds where $V_1$ is computed.
            
        dus : np.array
            The bin sizes of `us`. Ignored.
            
        verbose : bool, optional
            If True (default), progress bars are shown for the computations on data.
        
        Returns
        -------
        V1 : np.array()
            The values of the second Minkowski Functional at the given thresholds.
        
        """
        if self.first_der is None:
            self.get_first_der()
        modulus = self.field.modulus()
        pol_angle = self.field.pol_angle()
        # theta = get_theta(self.nside)
        stat = np.zeros_like(us)
        
        for ii in tqdm(np.arange(us.shape[0]), disable=not verbose):
            integrand_1 = np.zeros_like(modulus)
            integrand_2 = np.zeros_like(modulus)
            # thetamask = ~(np.isclose(np.cos(theta),0, atol=1.e-2))
            mask = (us[ii]>-modulus) & (us[ii]<modulus)   # & thetamask
            first_der_t = QUarray(*zip(*self.first_der))[:,:,mask]

            psi_1 = pol_angle[mask] + np.arccos(us[ii]/modulus[mask])/2.
            psi_2 = pol_angle[mask] - np.arccos(us[ii]/modulus[mask])/2.
            
            integrand_1[mask] = np.sqrt(np.sum(first_der_t(psi_1)**2., axis=0))
            integrand_2[mask] = np.sqrt(np.sum(first_der_t(psi_2)**2., axis=0))
            
            total_integrand = (integrand_1 + integrand_2)   # /np.mean(thetamask)
            stat[ii] = np.mean(total_integrand[self.mask])/np.pi

        return _MF_prefactor(self.dim, 1) * stat
    
    
    def _V2(self, us, dus, verbose=True):
        """Compute the second Minkowski Functional, $V_2$, normalized by the volume of the space. Internal interface for `pynkowski.V2`.

        Parameters
        ----------
        us : np.array
            The thresholds where $V_2$ is computed.
            
        dus : np.array
            The bin sizes of `us`. Ignored.
            
        verbose : bool, optional
            If True (default), progress bars are shown for the computations on data.
        
        Returns
        -------
        V2 : np.array()
            The values of the second Minkowski Functional at the given thresholds.
        
        """
        if self.first_der is None:
            self.get_first_der()
        if self.second_der is None:
            self.get_second_der()
            
        modulus = self.field.modulus()
        pol_angle = self.field.pol_angle()
        # theta = get_theta(self.nside)
        stat = np.zeros_like(us)

        for ii in tqdm(np.arange(us.shape[0]), disable=not verbose):
            integrand_1 = np.zeros_like(modulus)
            integrand_2 = np.zeros_like(modulus)
            
            # thetamask = ~(np.isclose(np.cos(theta),0, atol=1.e-2))
            mask = (us[ii]>-modulus) & (us[ii]<modulus)  #& thetamask
            # pixs = np.arange(12*self.nside**2)[mask]
            first_der_t = QUarray(*zip(*self.first_der))[:,:,mask]       # This just "transposes" the array without losing QUarray functionality, so that the first index is the Q/U component, the second is the derivative direction, and the third is the pixel
            second_der_t = QUarray(*zip(*self.second_der))[:,:,mask]
            
            psi_1 = pol_angle[mask] + np.arccos(us[ii]/modulus[mask])/2.
            psi_2 = pol_angle[mask] - np.arccos(us[ii]/modulus[mask])/2.

            integrand_1[mask] = self._H(first_der_t, second_der_t, psi_1)
            integrand_2[mask] = self._H(first_der_t, second_der_t, psi_2)
        
            total_integrand = (integrand_1 + integrand_2)  #/np.mean(thetamask)
        
            stat[ii] = np.mean(total_integrand[self.mask])/np.pi

        return _MF_prefactor(self.dim, 2) * stat
    
    def _V3(self, us, dus, verbose=True):
        """Compute the second Minkowski Functional, $V_3$, normalized by the volume of the space. Internal interface for `pynkowski.V3`.

        Parameters
        ----------
        us : np.array
            The thresholds where $V_3$ is computed.
            
        dus : np.array
            The bin sizes of `us`. Ignored.
            
        verbose : bool, optional
            If True (default), progress bars are shown for the computations on data.
        
        Returns
        -------
        V3 : np.array()
            The values of the second Minkowski Functional at the given thresholds.
        
        """
        if self.first_der is None:
            self.get_first_der()
        if self.second_der is None:
            self.get_second_der()
            
        modulus = self.field.modulus()
        pol_angle = self.field.pol_angle()
        # theta = get_theta(self.nside)
        stat = np.zeros_like(us)

        for ii in tqdm(np.arange(us.shape[0]), disable=not verbose):
            integrand_1 = np.zeros_like(modulus)
            integrand_2 = np.zeros_like(modulus)
            
            # thetamask = ~(np.isclose(np.cos(theta),0, atol=1.e-2))
            mask = (us[ii]>-modulus) & (us[ii]<modulus)  #& thetamask
            # pixs = np.arange(12*self.nside**2)[mask]
            first_der_t = QUarray(*zip(*self.first_der))[:,:,mask]
            second_der_t = QUarray(*zip(*self.second_der))[:,:,mask]
            
            psi_1 = pol_angle[mask] + np.arccos(us[ii]/modulus[mask])/2.
            psi_2 = pol_angle[mask] - np.arccos(us[ii]/modulus[mask])/2.

            integrand_1[mask] = self._K(first_der_t, second_der_t, psi_1)
            integrand_2[mask] = self._K(first_der_t, second_der_t, psi_2)
        
            total_integrand = (integrand_1 + integrand_2)  #/np.mean(thetamask)
        
            stat[ii] = np.mean(total_integrand[self.mask])/np.pi

        return _MF_prefactor(self.dim, 3) * stat

    @staticmethod
    def _H(first_der_t, second_der_t, psi):
        """Compute the mean curvature of the field at a given angle.
        """
        first_der_t = first_der_t(psi)
        second_der_t = second_der_t(psi)
        hess = np.array([[second_der_t[0], second_der_t[3], second_der_t[4]],
                        [second_der_t[3], second_der_t[1], second_der_t[5]],
                        [second_der_t[4], second_der_t[5], second_der_t[2]]])
        norm_grad = first_der_t / np.sqrt(np.sum(first_der_t**2., axis=0))
        return np.einsum('j...,jk...,k... -> ...', norm_grad, hess, norm_grad) - np.trace(hess, axis1=0, axis2=1)
    
    @staticmethod
    def _K(first_der_t, second_der_t, psi):
        """Compute the Gaussian curvature of the field at a given angle.
        """
        first_der_t = first_der_t(psi)
        mod_grad = np.sqrt(np.sum(first_der_t**2., axis=0))
        second_der_t = second_der_t(psi) / mod_grad
        norm_grad = first_der_t / mod_grad
        extended_hessian_det = np.linalg.det(np.array([[second_der_t[0], second_der_t[3], second_der_t[4], norm_grad[0]],
                                                        [second_der_t[3], second_der_t[1], second_der_t[5], norm_grad[1]],
                                                        [second_der_t[4], second_der_t[5], second_der_t[2], norm_grad[2]],
                                                        [norm_grad[0], norm_grad[1], norm_grad[2], np.zeros_like(norm_grad[0])]]).T).T
        return -extended_hessian_det*mod_grad


        

__all__ = ["SO3Healpix", "QUarray"]

__docformat__ = "numpy"
