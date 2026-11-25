#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:37:47 2020

@author: robertwi
"""

"""
Script that generates a parameter file for given IC.
"""


def createparamfile(filename,sim,glasssetup,setup_case):
    # A setup may close only SOME axes periodically (powerlawring's cylindrical
    # slab is periodic in z and open in x/y). Gasoline disables an axis with a
    # period of zero (master.c: 0 -> FLOAT_MAXVAL), so the override is a plain
    # per-axis period; everything else keeps using the IC's own box.
    dxP=getattr(sim,'dxPeriod',sim.dxbound)
    dyP=getattr(sim,'dyPeriod',sim.dybound)
    dzP=getattr(sim,'dzPeriod',sim.dzbound)
    periodic=sim.periodic
    delta=sim.deltastep
    nsteps=sim.nsteps
    dmsol=sim.dmsolunit
    dkpc=sim.dkpcunit
    out=sim.freqout
    adi=sim.adi
    molw=sim.molweight
    gamma=sim.gamma
    isothermal_gamma=getattr(sim,'isothermal_gamma',1.0000000001)
    grav=sim.grav
    cosmo=sim.cosmo
    rhoit=sim.rhoit
    ns=sim.ns
    dtheta=getattr(sim,'dtheta',0.6)
    dtheta2=getattr(sim,'dtheta2',0.8)
    hubble0=getattr(sim,'H0',2.894405)
    omega0=getattr(sim,'omega0',1.0)
    omegaLambda=getattr(sim,'omegaLambda',0.0)
    dICdensRsmooth=sim.dICdensRsmooth
    dICdensprofile=sim.dICdensprofile
    dICdensdir=sim.dICdensdir
    dICdensR=sim.dICdensR
    dICdensinner=sim.dICdensinner
    dICdensouter=sim.dICdensouter

    f = open('datafiles/' + filename+'.param', 'w')
    f.write('achInFile \t = '+ filename + '.00000' +' \n')
    f.write('achOutName \t = '+ filename + ' \n')
    if((setup_case=="planet" or setup_case=="planetcollision")and glasssetup == 0  ):
        f.write('achMaterialFile \t = '+ sim.materialfile +'\n')
    if(glasssetup==1):
        periodic=1
    if(periodic==1):
        f.write('bPeriodic \t = 1 \n')
        f.write('dxPeriod \t = '+ str(dxP) + ' \n')
        f.write('dyPeriod \t = '+ str(dyP) + ' \n')
        f.write('dzPeriod \t = '+ str(dzP) + ' \n')
    else:
        f.write('bPeriodic \t = 0 \n')
    f.write('dDelta \t = '+ str(delta) + ' \n')
    f.write('nSteps \t = '+ str(nsteps) + ' \n')
    if(adi!=2):
        f.write('dMsolUnit \t = '+ str(dmsol) + ' \n')
        f.write('dKpcUnit \t = '+ str(dkpc) + ' \n')
    f.write('iCheckInterval \t = '+ str(round(out*10)) + ' \n')
    f.write('iLogInterval \t = 1 \n')
    f.write('iOutInterval \t = '+ str(out) + ' \n')
    f.write('dMeanMolWeight \t = '+ str(molw) + ' \n')
    if(adi==1):
        f.write('bGasAdiabatic \t = 1 \n')
        f.write('dConstGamma \t = '+ str(gamma) + ' \n')
    elif(adi==0):
        f.write('bGasIsothermal \t = 1 \n')
        f.write('dConstGamma \t = '+ str(isothermal_gamma) + ' \n')
    elif(adi==2):
        f.write('bGasAdiabatic \t = 1 \n')
        f.write('dConstGamma \t = '+ str(gamma) + ' \n')
        f.write('dGasConst \t = '+ str(gamma-1) + ' \n')
    elif(adi==3):
        # Runtime cooling. msrSetGasModel ASSERTS if more than one gas model is
        # specified, so this is mutually exclusive with bGasAdiabatic -- it
        # cannot simply be appended to an adiabatic .param.
        f.write('bGasCooling \t = 1 \n')
        f.write('dConstGamma \t = '+ str(gamma) + ' \n')
    if(grav==1):
        f.write('bDoGravity \t = 1 \n')
        f.write('dhMinOverSoft \t = 0.1 \n')
    else:
        f.write('bDoGravity \t = 0 \n')
    if(setup_case=="turbbox" and glasssetup==0):
        # OUTURBDRIVER is a compile-time feature, but all of its physical and
        # reproducibility controls are runtime parameters.  StEnergy is the
        # reference-calibrated input rate per mode; achieved Mach is measured
        # by IC_analysis_turbbox rather than assumed here.
        f.write('# TurbboxTargetMach = ' + str(sim.turbbox_mach) + ' \n')
        f.write('# TurbboxForcing = ' + sim.turbbox_forcing + ' \n')
        f.write('# TurbboxCrossingTime = ' + str(sim.turbbox_tcross) + ' \n')
        f.write('# TurbboxMHD = ' + str(sim.turbbox_mhd) + ' \n')
        f.write('# TurbboxB0 = ' + str(sim.turbbox_B0) + ' \n')
        f.write('bKDK \t = 1 \n')
        for key in ('StDecay', 'StEnergy', 'StDtFreq', 'StKmin', 'StKmax',
                    'StSolWeight', 'StAmplFac', 'StScaleHeight',
                    'StStartTime', 'StSpectForm', 'StSeed',
                    'StCorrectBulkMotion'):
            f.write(key + ' \t = ' + str(getattr(sim, key)) + ' \n')
        if getattr(sim, 'StForcingFile', ''):
            f.write('StForcingFile = ' + sim.StForcingFile + '\n')
        f.write('# TurbboxDriving = ' + getattr(sim, 'turbbox_driving', 'generated') + '\n')
    if(setup_case=="starform" and glasssetup==0):
        # STARFORM/SFBFIELD are compile-time (runtest.sh requests them through
        # GASOLINE_DEFINES); everything that shapes the recipe is runtime, and
        # all of it comes from attributes IC_setup_starform.create() set, so the
        # analytic rate prediction and the .param cannot drift apart.
        # Only a WHOLE-particle conversion reaches the SFBFIELD handoff, and
        # that comes from the dInitStarMass clamp, not from dStarEff (0.5 here,
        # and inert); dMinGasMass > 0 is ASSERTED when dInitStarMass == 0.
        # Names are gasoline's long (.param) spellings, not the -flag aliases.
        # A case that needs the EVALUATED temperature runs bGasCooling = 1 so
        # that CoolSetTime initialises the rate tables (without it the SF
        # criterion indexes uninitialised arrays and segfaults -- blocker B3).
        # bUV = 0 is what keeps that honest: with no photoheating and no metals
        # there is nothing to heat the gas and nothing to cool it below 10^4 K,
        # so the thermal state the test controls stays put (measured: 0.4 per
        # cent drift over four steps, against 164 -> 7300 K with UV on).
        if getattr(sim, 'adi', 1) == 3:
            f.write('bUV \t = 0 \n')
        for key in ('bStarForm', 'bFeedBack', 'dCStar', 'dStarEff',
                    'dInitStarMass', 'dMinGasMass', 'dMinMassFrac',
                    'dPhysDenMin', 'dTempMax', 'dDeltaStarForm',
                    'iRandomSeed', 'bBHForm', 'bDoSphhOutput'):
            f.write(key + ' \t = ' + str(getattr(sim, 'sf_' + key)) + ' \n')
    if getattr(sim, 'fdm_mass_eV', None) is not None and glasssetup==0:
        # Fuzzy-DM quantum pressure on the dark particles (ChaNGa --enable-fdm).
        # A binary built without FDM rejects these keys, so such a run fails
        # loudly instead of silently running plain CDM. runtest.sh passes the
        # estimator variant (and may override the mass) as run flags.
        f.write('bFDM \t = 1 \n')
        f.write('dFDMMass_eV \t = ' + str(sim.fdm_mass_eV) + ' \n')
    if(setup_case=="powerlawring" and glasssetup==0):
        # Isolate the external field: no live sink and no interparticle tree.
        f.write('bDoSelfGravity \t = 0 \n')
        # bBodyForce is a separate legacy vertical-force kernel, not an
        # umbrella switch for external potentials.
        f.write('bBodyForce \t = 0 \n')
        f.write('bPowerLawPotential \t = 1 \n')
        f.write('dPowerLawPotentialBeta \t = ' + str(sim.powerlaw_beta) + ' \n')
        f.write('dPowerLawPotentialSoftening \t = ' +
                str(sim.powerlaw_softening) + ' \n')
        f.write('iPowerLawPotentialGeometry \t = ' +
                str(sim.powerlaw_geometry) + ' \n')
        # The force law: 0 softened power law, 1 bounded Kepler.  The IC and
        # gasoline use the same spelling, so nothing is remapped here.
        f.write('iPowerLawPotentialForm \t = ' +
                str(sim.powerlaw_force_form) + ' \n')
    if(sim.inflow==1 and glasssetup!=1):
        f.write('bInflowOutflow \t = 1 \n')
        f.write('dxInflow \t = ' + str(sim.dxInflow) + ' \n')
        f.write('dxOutflow \t = ' + str(sim.dxOutflow) + ' \n')
        f.write('dyOutflowLeft \t = ' + str(sim.dyOutflowLeft) + ' \n')
        f.write('dyOutflowRight \t = ' + str(sim.dyOutflowRight) + ' \n')
        f.write('dMInflow \t = ' + str(sim.dMInflow) + ' \n')
        f.write('dTInflow \t = ' + str(sim.dTInflow) + ' \n')
        f.write('dVelInflow \t = ' + str(sim.dVelInflow) + ' \n')
        f.write('dAccInflow \t = ' + str(sim.dAccInflow) + ' \n')
        f.write('dCloudPosition \t = ' + str(sim.dCloudPosition) + ' \n')
    if(setup_case=="accdisk" and glasssetup==0):
        # the ISOR EOS (gasoline -DISOR; SPH-EXA's converter reads the same keys): cs = dDiskHr R^-dDiskq
        f.write('dDiskHr \t = ' + str(sim.hr) + ' \n')
        f.write('dDiskq \t = ' + str(sim.q) + ' \n')
    if(setup_case=="blob" and glasssetup!=1):
        f.write('dCloudDensity \t = ' + str(sim.dCloudDensity) + ' \n')
    f.write('#bLogTiming \t = 1 \n')
    f.write('bStandard \t = 1 \n')
    f.write('bVDetails \t = 2 \n')
    if(rhoit==1):
        f.write('bDoSelfGravity \t = 0 \n')
        f.write('bBodyForce \t = 1 \n')
        # Constant-gravity magnitude (pkdBodyForce: a_z = -dBodyForceConst
        # sign(z)), FIXED at 1.0: every stratified setup's hydrostatic profile
        # must integrate with the same g=1.0 (mti and rt do). Changing g is
        # equivalent to rescaling time by 1/sqrt(g), so there is nothing to tune.
        f.write('dBodyForceConst \t = 1.0 \n')
    f.write('nSmooth \t = ' + str(ns) + ' \n')
    f.write('#dConstAlpha \t = 2 \n')
    f.write('#dConstBeta \t = 4 \n')
    f.write('#dThermalDiffusionCoeff \t = 0.03 \n')
    f.write('dEta \t = 0.175 \n')
    f.write('dEtaCourant \t = 0.4 \n')
    f.write('dEtaDiffusion \t = 0.1 \n')
    f.write('iMaxRung \t = 29 \n')
    f.write('iBinaryOutput \t = 1 \n')
    f.write('dTheta \t = '+ str(dtheta) + ' \n')
    f.write('bViscosityLimiter \t = 0 \n')
    f.write('bViscosityLimitdt \t = 0 \n')
    if(cosmo==1):
        f.write('dTheta2 \t = '+ str(dtheta2) + ' \n')
        f.write('dHubble0 \t = '+ str(hubble0) + ' \n')
        f.write('dOmega0 \t = '+ str(omega0) + ' \n')
        f.write('dLambda \t = '+ str(omegaLambda) + ' \n')
        f.write('dRedTo \t = 0.0 \n')
        f.write('bComove \t = 1 \n')
        f.write('dFracNoDomainDecomp \t = 0.1 \n')
        f.write('nTruncateRung \t = 128 \n')
    elif not getattr(sim, 'ewald', 0):
        # A self-gravitating periodic setup that needs the Ewald sum (cubic box) sets ewald = 1.
        f.write('bEwald \t = 0 \n')
    if(glasssetup==1):
        f.write('dICdensRsmooth \t = ' + str(dICdensRsmooth) + ' \n')
        f.write('dICdensprofile \t = ' + str(dICdensprofile) + ' \n')
        f.write('dICdensdir \t = ' + str(dICdensdir) + ' \n')
        f.write('dICdensR \t = ' + str(dICdensR) + ' \n')
        f.write('dICdensinner \t = ' + str(dICdensinner) + ' \n')
        f.write('dICdensouter \t = ' + str(dICdensouter) + ' \n')
    if(setup_case=='mti'):
        f.write('bThermalCondLinearCodeUnits \t = 1  \n')
        f.write('bThermalCondTauConst \t = 1  \n')
        f.write('dThermalCondTau \t = 0.001  \n')
        f.write('dThermalCondCoeff \t = 0.01  \n')
        f.write('dThermalCond2Coeff \t = 0.01  \n')

    f.close()
