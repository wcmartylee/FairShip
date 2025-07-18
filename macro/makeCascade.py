import ROOT,time,os,sys,random,copy,argparse
from array import array
import rootUtils as ut

ROOT.gROOT.LoadMacro("$VMCWORKDIR/gconfig/basiclibs.C")
ROOT.basiclibs()
timer = ROOT.TStopwatch()
timer.Start()

ap = argparse.ArgumentParser(description='Run SHiP makeCascade: generate ccbar or bbar')
ap.add_argument('-s', '--seed', type=int, default=int(time.time() * 100000000 % 900000000), help="Random number seed, integer. If not given, current time will be used")
ap.add_argument('-t', '--Fntuple', default='', help="Name of ntuple output file, default: Cascade{nevgen/1000}k-parp16-MSTP82-1-MSEL{mselcb}-ntuple.root")
ap.add_argument('-n', '--nevgen', type=int, default=100000, help="Number of events to produce, default 100000")
ap.add_argument('-E', '--pbeamh', type=float, default=400.0, help="Energy of beam in GeV, default 400 GeV")
ap.add_argument('-m', '--mselcb', type=int, default=4, help="4 (5): charm (beauty) production, default charm", choices=[4, 5])
ap.add_argument('-P', '--storePrimaries', action=argparse.BooleanOptionalAction, default=False, help="If this argument is used, store all particles produced together with charm.")
ap.add_argument('--target_composition', default='W',
                help="Target composition (to determine the ratio of protons in the material). Default is Tungsten (W). Only other choice is Molybdenum (Mo)", choices=['W', 'Mo'])
ap.add_argument('--pythia_tune', default='PoorE791',
                help="Choices of Pythia tune are PoorE791 (default, settings with default Pythia6 pdf, based on getting <pt> at 500 GeV pi-) or LHCb (settings by LHCb for Pythia 6.427)",
                choices=['PoorE791', 'LHCb'])
# some parameters for generating the chi (sigma(signal)/sigma(total) as a function of momentum
ap.add_argument('--nev', type=int, default=5000, help="Events / momentum")
ap.add_argument('--nrpoints', type=int, default=20, help="Number of momentum points taken to calculate sig/sigtot")

args = ap.parse_args()
if args.Fntuple == '':
    args.Fntuple = f'Cascade{int(args.nevgen/1000)}k-parp16-MSTP82-1-MSEL{args.mselcb}-ntuple.root'

print(f'Generate {args.nevgen}  p.o.t. with msel = {args.mselcb} proton beam {args.pbeamh}GeV')
print(f'Output ntuples written to: {args.Fntuple}')

# cascade beam particles, anti-particles are generated automatically if they exist.
idbeam = [2212, 211, 2112, 321, 130, 310]
target = ['p+','n0']
print(f'Chi generation with {args.nev} events/point, nr points={args.nrpoints}')
print(f'Cascade beam particle: {idbeam}')

# fracp is the fraction of protons in nucleus, used to average chi on p and n target in Pythia.
if args.target_composition == 'W': # target is Tungsten, fracp is 74/(184)
    fracp = 0.40
else: # target would then be Molybdenum, fracp is 42/98
    fracp = 0.43

print(f'Target particles: {target}, fraction of protons in {args.target_composition}={fracp}')

# lower/upper momentum limit for beam, depends on msel..
# signal particles wanted (and their antis), which could decay semi-leptonically.
if args.mselcb == 4:
   pbeaml = 34.
   idsig = [411, 421, 431, 4122, 4132, 4232, 4332, 4412, 4414, 4422, 4424, 4432, 4434, 4444]
elif args.mselcb == 5:
   pbeaml = 130.
   idsig = [511, 521, 531, 541, 5122, 5132, 5142, 5232, 5242, 5332, 5342, 5412, 5414, 5422, 5424, 5432, 5434, 5442, 5444, 5512, 5514, 5522, 5524, 5532, 5534, 5542, 5544, 5554]
else:
   print(f'Error: msel is unknown: {args.mselcb}, STOP program')
   sys.exit('ERROR on input, exit')

PDG = ROOT.TDatabasePDG.Instance()
myPythia = ROOT.TPythia6()
tp = ROOT.tPythia6Generator()

# Pythia6 can only accept names below in pyinit, hence reset PDG table:
PDG.GetParticle(2212).SetName('p+')
PDG.GetParticle(-2212).SetName('pbar-')
PDG.GetParticle(2112).SetName('n0')
PDG.GetParticle(-2112).SetName('nbar0')
PDG.GetParticle(130).SetName('KL0')
PDG.GetParticle(310).SetName('KS0')
# lower lowest sqrt(s) allowed for generating events
myPythia.SetPARP(2, 2.)


def PoorE791_tune(P6):
    # settings with default Pythia6 pdf, based on getting <pt> at 500 GeV pi-
    # same as that of E791: http://arxiv.org/pdf/hep-ex/9906034.pdf
    print(' ')
    print('********** PoorE791_tune **********')
    # change pt of D's to mimic E791 data.
    P6.SetPARP(91, 1.6)
    print(f'PARP(91)={P6.GetPARP(91)}')
    # what PDFs etc.. are being used:
    print('MSTP PDF info, i.e. (51) and (52)=', P6.GetMSTP(51), P6.GetMSTP(52))
    # set multiple interactions equal to Fortran version, i.e.=1, default=4, and adapt parp(89) accordingly
    P6.SetMSTP(82, 1)
    P6.SetPARP(89, 1000.)
    print('And corresponding multiple parton stuff, i.e. MSTP(82),PARP(81,89,90)=', P6.GetMSTP(82), P6.GetPARP(81), P6.GetPARP(89), P6.GetPARP(90))
    print('********** PoorE791_tune **********')
    print(' ')


def LHCb_tune(P6):
    # settings by LHCb for Pythia 6.427
    # https://twiki.cern.ch/twiki/bin/view/LHCb/SettingsSim08
    print(' ')
    print('********** LHCb_tune **********')
    # P6.SetCKIN(41,3.0)
    P6.SetMSTP(2,	2)
    P6.SetMSTP(33,	3)
    # P6.SetMSTP(128,	2)  # store or not store
    P6.SetMSTP(81,	21)
    P6.SetMSTP(82,	3)
    P6.SetMSTP(52,	2)
    P6.SetMSTP(51,	10042)  # (ie CTEQ6 LO fit, with LOrder alpha_S PDF from LHAPDF)
    P6.SetMSTP(142,	0)  # do not weigh events
    P6.SetPARP(67,	1.0)
    P6.SetPARP(82,	4.28)
    P6.SetPARP(89,	14000)
    P6.SetPARP(90,	0.238)
    P6.SetPARP(85,	0.33)
    P6.SetPARP(86,	0.66)
    P6.SetPARP(91,	1.0)
    P6.SetPARP(149,	0.02)
    P6.SetPARP(150,	0.085)
    P6.SetPARJ(11,	0.4)
    P6.SetPARJ(12,	0.4)
    P6.SetPARJ(13,	0.769)
    P6.SetPARJ(14,	0.09)
    P6.SetPARJ(15,	0.018)
    P6.SetPARJ(16,	0.0815)
    P6.SetPARJ(17,	0.0815)
    P6.SetMSTJ(26,	0)
    P6.SetPARJ(33,	0.4)
    print('********** LHCb_tune **********')
    print(' ')


def fillp1(hist):
    # scan filled bins in hist, and fill intermediate bins with linear interpolation
    nb = hist.GetNbinsX()
    i1 = hist.FindBin(pbeaml, 0., 0.)
    y1 = hist.GetBinContent(i1)
    p1 = hist.GetBinCenter(i1)
    for i in range(i1 + 1, nb + 1):
        if hist.GetBinContent(i) > 0.:
            y2 = hist.GetBinContent(i)
            p2 = hist.GetBinCenter(i)
            tg = (y2 - y1) / (p2 - p1)
            for ib in range(i1 + 1, i):
                px = hist.GetBinCenter(ib)
                yx = (px - p1) * tg + y1
                hist.Fill(px, yx)
                i1 = i
                y1 = y2
                p1 = p2


if args.pythia_tune == 'PoorE791':
    PoorE791_tune(myPythia)
    myPythia.OpenFortranFile(11, os.devnull)  # Pythia output to dummy (11) file (to screen use 6)
    myPythia.SetMSTU(11, 11)
else:
    LHCb_tune(myPythia)
    myPythia.OpenFortranFile(6, os.devnull)  # avoid any printing to the screen, only when LHAPDF is used, in LHCb tune

# start with different random number for each run...
print(f'Setting random number seed = {args.seed}')
myPythia.SetMRPY(1, args.seed)

# histogram helper
h = {}

# initialise phase, determine chi=sigma(signal)/sigma(total) for many beam momenta.
# loop over beam particles
id = 0
nb = 400
t0 = time.time()
idhist = len(idbeam) * 4 * [0]
print('Get chi vs momentum for all beam+target particles')
for idp in range(0, len(idbeam)):
    for idpm in [-1, 1]:  # particle or anti-particle
        idw = idbeam[idp] * idpm
        if PDG.GetParticle(idw)!=None:  # if particle exists, book hists etc.
            name = PDG.GetParticle(idw).GetName()
            id = id + 1
            for idnp in range(2):
                idb = id * 10 + idnp * 4
                ut.bookHist(h, str(idb + 1), f'sigma vs p, {name} -> {target[idnp]}', nb, 0.5, args.pbeamh + 0.5)
                ut.bookHist(h, str(idb + 2), f'sigma-tot vs p, {name} -> {target[idnp]}', nb, 0.5, args.pbeamh + 0.5)
                ut.bookHist(h, str(idb + 3), f'chi vs p, {name}-> {target[idnp]}', nb, 0.5, args.pbeamh + 0.5)
                ut.bookHist(h, str(idb + 4), f'Prob(normalised), {name} -> {target[idnp]}', nb, 0.5, args.pbeamh + 0.5)
            # keep track of histogram<->Particle id relation.
            idhist[id] = idw
            for idpn in range(2):  # target is proton or neutron
                idadd = idpn * 4
                print(f'{name}, {target[idpn]}, for  chi, seconds: {time.time()-t0}')
                for ipbeam in range(args.nrpoints):  # loop over beam momentum
                    pbw = ipbeam * (args.pbeamh - pbeaml) / (args.nrpoints - 1) + pbeaml
                    # convert to center of a bin
                    ibin = h[str(id * 10 + 1+ idadd)].FindBin(pbw, 0., 0.)
                    pbw = h[str(id * 10 + 1 + idadd)].GetBinCenter(ibin)
                    # new particle/momentum, init again, first signal run.
                    myPythia.SetMSEL(args.mselcb)  # set forced ccbar or bbbar generation
                    myPythia.Initialize('FIXT', name, target[idpn], pbw)
                    for iev in range(args.nev):
                        myPythia.GenerateEvent()
                    # signal run finished, get cross-section
                    h[str(id * 10 + 1 + idadd)].Fill(pbw, tp.getPyint5_XSEC(2, 0))
                    myPythia.SetMSEL(2)  # now total cross-section, i.e. msel=2
                    myPythia.Initialize('FIXT', name, target[idpn], pbw)
                    for iev in range(args.nev//10):
                    # if iev%100==0: print 'generated mbias events',iev,' seconds:',time.time()-t0
                        myPythia.GenerateEvent()
                    # get cross-section
                    h[str(id * 10 + 2 + idadd)].Fill(pbw, tp.getPyint5_XSEC(2, 0))
                # for this beam particle, do arithmetic to get chi.
                h[str(id * 10 + 3 + idadd)].Divide(h[str(id * 10 + 1 + idadd)], h[str(id * 10 + 2 + idadd)], 1., 1.)
                # fill with linear function between evaluated momentum points.
                fillp1(h[str(id * 10 + 3 + idadd)])

# collected all, now re-scale to make max chi at 400 Gev==1.
chimx = 0.
for i in range(1, id + 1):
    for idpn in range(2):
        idw = i * 10 + idpn * 4 + 3
        ibh = h[str(idw)].FindBin(args.pbeamh)
        print(f'beam id: {i}, {idw}, {idhist[i]}, momentum: {args.pbeamh},chi: {h[str(idw)].GetBinContent(ibh)}, chimx: {chimx}')
        if h[str(idw)].GetBinContent(ibh)>chimx: chimx=h[str(idw)].GetBinContent(ibh)
chimx = 1./chimx
for i in range(1, id + 1):
    for idpn in range(2):
        idw = i * 10 + idpn * 4 + 3
        h[str(idw + 1)].Add(h[str(idw)], h[str(idw)], chimx, 0.)

# switch output printing OFF for generation phase.
# Generate ccbar (or bbbar) pairs according to probability in hists i*10+8.
# some check histos
ut.bookHist(h,str(1), 'E of signals', 100,0., 400.)
ut.bookHist(h,str(2), 'nr signal per cascade depth', 50, 0.5, 50.5)
ut.bookHist(h,str(3), 'D0 pt**2', 40, 0., 4.)
ut.bookHist(h,str(4), 'D0 pt**2', 100, 0., 18.)
ut.bookHist(h,str(5), 'D0 pt', 100, 0., 10.)
ut.bookHist(h,str(6), 'D0 XF', 100, -1., 1.)

ftup = ROOT.TFile.Open(args.Fntuple, 'RECREATE')
Ntup = ROOT.TNtuple("pythia6", "pythia6 heavy flavour",\
       "id:px:py:pz:E:M:mid:mpx:mpy:mpz:mE:mM:k:a0:a1:a2:a3:a4:a5:a6:a7:a8:a9:a10:a11:a12:a13:a14:a15:\
s0:s1:s2:s3:s4:s5:s6:s7:s8:s9:s10:s11:s12:s13:s14:s15")

# make sure all particles for cascade production are stable
for kf in idbeam:
    kc = myPythia.Pycomp(kf)
    myPythia.SetMDCY(kc,1,0)
# make charm or beauty signal hadrons stable
for kf in idsig:
    kc = myPythia.Pycomp(kf)
    myPythia.SetMDCY(kc,1,0)

stack = 1000 * [0]  # declare the stack for the cascade particles
for iev in range(args.nevgen):
    if iev % 1000 == 0: print('Generate event ',iev)
    nstack=0
    # put protons of energy pbeamh on the stack
    # stack: PID, px, py, pz, cascade depth, nstack of mother
    stack[nstack] = [2212, 0., 0., args.pbeamh, 1, 100 * [0], 100 * [0]]
    stack[nstack][5][0] = 2212
    while nstack >= 0:
        # generate a signal based on probabilities in hists i*10+8?
        ptot = ROOT.TMath.Sqrt(stack[nstack][1] ** 2 + stack[nstack][2] ** 2 + stack[nstack][3] ** 2)
        prbsig = 0.
        for i in range(1, id + 1):
            if stack[nstack][0] == idhist[i]:  # get hist id for this beam particle
                idpn = 0
                if random.random() > fracp:  # decide on p or n target
                    idpn = 1
                idw = i * 10 + idpn * 4 + 4
                ib = h[str(idw)].FindBin(ptot, 0., 0.)
                prbsig = h[str(idw)].GetBinContent(ib)
        if prbsig > random.random():
            # last particle on the stack as beam particle
            for k in range(1, 4):
                myPythia.SetP(1, k, stack[nstack][k])
                myPythia.SetP(2, k, 0.)
            # new particle/momentum, init again: signal run.
            myPythia.SetMSEL(args.mselcb)  # set forced ccbar or bbbar generation
            myPythia.Initialize('3MOM', PDG.GetParticle(stack[nstack][0]).GetName(), target[idpn], 0.)
            myPythia.GenerateEvent()
            # look for the signal particles
            charmFound = []
            for itrk in range(1,myPythia.GetN() + 1):
                idabs = ROOT.TMath.Abs(myPythia.GetK(itrk, 2))
                if idabs in idsig:  # if signal is found, store in ntuple
                     vl = array('f')
                     vl.append(float(myPythia.GetK(itrk, 2)))
                     for i in range(1, 6):
                         vl.append(float(myPythia.GetP(itrk, i)))
                     vl.append(float(myPythia.GetK(1, 2)))
                     for i in range(1, 6):
                         vl.append(float(myPythia.GetP(1, i)))
                     vl.append(float(stack[nstack][4]))
                     for i in range(16):
                         vl.append(float(stack[nstack][5][i]))
                     nsub=stack[nstack][4] - 1
                     if nsub > 15: nsub = 15
                     for i in range(nsub):
                         vl.append(float(stack[nstack][6][i]))
                     vl.append(float(myPythia.GetMSTI(1)))
                     for i in range(nsub + 1, 16):
                         vl.append(float(0))
                     Ntup.Fill(vl)
                     charmFound.append(itrk)
                     h['1'].Fill(myPythia.GetP(itrk, 4))
                     h['2'].Fill(stack[nstack][4])
                     if idabs == 421 and stack[nstack][4] == 1 :
                         # some checking hist to monitor pt**2, XF of prompt D^0
                         pt2 = myPythia.GetP(itrk, 1) ** 2 + myPythia.GetP(itrk, 2) ** 2
                         h['3'].Fill(pt2)
                         h['4'].Fill(pt2)
                         h['5'].Fill(ROOT.TMath.Sqrt(pt2))
                         # boost to Cm frame for XF ccalculation of D^0
                         beta = args.pbeamh / (myPythia.GetP(1,4) + myPythia.GetP(2, 5))
                         gamma = (1 - beta ** 2) ** (-0.5)
                         pbcm = -gamma * beta * myPythia.GetP(1, 4) + gamma * myPythia.GetP(1, 3)
                         pDcm = -gamma * beta * myPythia.GetP(itrk, 4) + gamma * myPythia.GetP(itrk, 3)
                         xf = pDcm / pbcm
                         h['6'].Fill(xf)
            if len(charmFound) > 0 and args.storePrimaries:
                for itP in range(1, myPythia.GetN() + 1):
                    if itP in charmFound: continue
                    if myPythia.GetK(itP, 1) == 1:
                        # store only undecayed particle and no charm found
                        # ***WARNING****: with new with new ancestor and process info (a0-15, s0-15) add to ntuple, might not work???
                        Ntup.Fill(float(myPythia.GetK(itP,2)),float(myPythia.GetP(itP, 1)),float(myPythia.GetP(itP, 2)),float(myPythia.GetP(itP, 3)),\
                            float(myPythia.GetP(itP,4)),float(myPythia.GetP(itP, 5)),-1,\
                            float(myPythia.GetV(itP,1)-myPythia.GetV(charmFound[0], 1)),\
                            float(myPythia.GetV(itP,2)-myPythia.GetV(charmFound[0], 2)),\
                            float(myPythia.GetV(itP,3)-myPythia.GetV(charmFound[0], 3)), 0, 0, stack[nstack][4])

        # now generate msel=2 to add new cascade particles to the stack
        for k in range(1, 4):
            myPythia.SetP(1, k, stack[nstack][k])
            myPythia.SetP(2, k, 0.)
        # new particle/momentum, init again, generate total cross-section event
        myPythia.SetMSEL(2)  # mbias event
        idpn = 0
        if random.random() > fracp: idpn = 1
        myPythia.Initialize('3MOM', PDG.GetParticle(stack[nstack][0]).GetName(), target[idpn], 0.)
        myPythia.GenerateEvent()
        # remove used particle from the stack, before adding new
        # first store its history: cascade depth and ancestors-list
        icas = stack[nstack][4] + 1
        if icas > 98: icas = 98
        anclist = copy.copy(stack[nstack][5])
        sublist = copy.copy(stack[nstack][6])
        # fill in interaction process of first proton
        if nstack == 0: sublist[0] = myPythia.GetMSTI(1)
        nstack = nstack - 1
        for itrk in range(1, myPythia.GetN() + 1):
            if myPythia.GetK(itrk, 1) == 1:
                ptot = ROOT.TMath.Sqrt(myPythia.GetP(itrk, 1) ** 2 + myPythia.GetP(itrk, 2) ** 2 + myPythia.GetP(itrk, 3) ** 2)
                if ptot > pbeaml:
                    # produced particle is stable and has enough momentum, is it in the wanted list?
                    for isig in range(len(idbeam)):
                        if ROOT.TMath.Abs(myPythia.GetK(itrk, 2)) == idbeam[isig] and nstack < 999:
                            if nstack < 999: nstack += 1
                            # update ancestor list
                            tmp=copy.copy(anclist)
                            tmp[icas-1] = myPythia.GetK(itrk, 2)
                            stmp = copy.copy(sublist)
                            # Pythia interaction process identifier
                            stmp[icas-1] = myPythia.GetMSTI(1)
                            stack[nstack] = [myPythia.GetK(itrk, 2), myPythia.GetP(itrk, 1), myPythia.GetP(itrk, 2),myPythia.GetP(itrk, 3), icas, tmp, stmp]

print('Now at Ntup.Write()')
Ntup.Write()
for akey in h: h[akey].Write()
ftup.Close()

timer.Stop()
rtime = timer.RealTime()
ctime = timer.CpuTime()
print(' ')
print("Macro finished successfully.")
print(f"Output file is {ftup.GetName()}")
print(f"Real time {rtime} s, CPU time {ctime} s")
