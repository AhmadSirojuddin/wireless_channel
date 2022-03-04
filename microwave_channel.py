"""
This file generates a wideband channel impulse response for the MIMO channel microwave system using the Rician model.
The function can be used for the narrowband channel and the Rayleigh fading channel (see example).
All arguments must be the torch's tensor object.
"""

import torch as tr


def uWave_ch(ch_size, LOS_component=tr.tensor([0]), Rician_fact=tr.tensor([0]), dist=tr.tensor([1]),
             dist_ref=tr.tensor([1]), path_loss_ref=tr.tensor([1]), path_loss_exp=tr.tensor([0]),
             path_num=tr.tensor([1]), multipath_decay=tr.tensor([1])):
    """
    Arguments:
    A. Channel size
        (1) ch_size : 1D tensor with numel=2 representing the number of Rx and Tx antenna.
                      Num Rx antenna->0th element, num Tx antenna->1st element.

    B. Rician Model
        (2) LOS_component : Two dimension tensor of size ch_size representing the LoS channel component.
                            All elements must have absolute value = 1
        (3) Rician_fact : Rician Factor, in linear scale (not in dB).
                          Convert it to linear scale first if the original value is in dB.
                          'Rician_fact' must be real, scalar, and non-negative.
                          'Rician_fact=0' means Rayleigh fading channel is considered.
        Note : If Rayleigh channel is considered, inputs (2) and (3) can be ignored.

    C. PathLoss
        (4) dist : Distance from Tx to Rx.
                   It can either a scalar or matrix of size 'ch_size' representing the distances of each Tx-Rx antenna pairs.
                   All input elements must be real and positive.
        (5) dist_ref : Reference distance at which 'path_loss_ref' is defined.
                       'dist_ref' must be real, scalar, and positive.
        (6) path_loss_ref : Path loss at 'dist_ref', in linear scale (not in dB).
                            'path_loss_ref' must be real, scalar, and positive.
        (7) path_loss_exp : Path loss exponent, must be real, scalar,and non-negative
        Note : If normalized-distance channel model is considered, inputs (4)-(7) can be ignored

    D. Multipath Channel
        (8) path_num : number of channel impulse response for all channel.
                       Can be a scalar or a matrix.
                       All input elements must be real, integer, and positive
        (9) multipath_decay : Decay factor for the multi path channel impulse response model.
                              The underlying model can be found in the paper entitled:
                              'The Effects of Time Delay Spread on Portable Radio Communications Channels
                              with Digital Modulation'.
                              This input must be real, scalar, and positive.
        Note : If narrow-band channel is considered, inputs (8)-(9) can be ignored.

    Output:
    A. For narrow band case (path_num=1), output is a tensor of size [Rx, Tx].
    B. For wide band case (path_num>1), output is a tensor of size  [max(path_num), Rx, Tx]
    """

    # The following code part is for a prevention from wrong input. You can comment them once you are sure with your input
    incr = 10 ** -3  # Just small tolerance
    if tr.is_complex(ch_size) or tr.numel(ch_size) != 2 or tr.any(ch_size < 1) or tr.is_floating_point(ch_size):
        print('WARNING!!! input ch_size must be real, integer, positive, and a vector with two elements')
        print('your input ch_size = ', ch_size)
        raise ValueError('INPUT ERROR')

    if Rician_fact == 0:
        LOS_component = tr.zeros(ch_size.tolist())  # dummy
    else:
        if not tr.equal(ch_size, tr.tensor(LOS_component.size())) or tr.any(
              tr.reshape(LOS_component, (-1,)).abs() < 1 - incr) \
              or tr.any(tr.reshape(LOS_component, (-1,)).abs() > 1 + incr):
            print(
                'WARNING!!! input LOS_component must be a size of ch_size, and all absolute value of element must be one')
            print('your input LOS_component = ', LOS_component)
            print('LOS_component size = ', LOS_component.size())
            print('absolute value of LOS_component = ', tr.abs(LOS_component))
            raise ValueError('INPUT ERROR')
    if tr.is_complex(Rician_fact) or tr.numel(Rician_fact) != 1 or Rician_fact < 0:
        print('WARNING!!! input Rician_fact must be real scalar non-negative')
        print('your input Rician_fact = ', Rician_fact)
        raise ValueError('INPUT ERROR')

    if tr.is_complex(dist) or (tr.numel(dist) != 1 and not tr.equal(ch_size, tr.tensor(dist.size()))) \
          or tr.any(tr.reshape(dist, (-1,)) <= 0):
        print(
            'WARNING!!! input dist must be real, positive, and a scalar or matrix whose size is the same with ch_size')
        print('your input dist = ', dist)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(dist_ref) or tr.numel(dist_ref) != 1 or dist_ref <= 0:
        print('WARNING!!! input dist_ref must be a scalar, real, positive')
        print('your input dist_ref = ', dist_ref)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(path_loss_ref) or tr.numel(path_loss_ref) != 1 or path_loss_ref <= 0:
        print('WARNING!!! input path_loss_ref must be a scalar, real, positive')
        print('your input path_loss_ref = ', path_loss_ref)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(path_loss_exp) or tr.numel(path_loss_exp) != 1 or path_loss_exp < 0:
        print('WARNING!!! input path_loss_exp must be a scalar, real, positive')
        print('your input path_loss_exp = ', path_loss_exp)
        raise ValueError('INPUT ERROR')

    if tr.is_complex(path_num) or (tr.numel(path_num) != 1 and not tr.equal(ch_size, tr.tensor(path_num.size()))) \
          or tr.is_floating_point(path_num) or tr.any(tr.reshape(path_num, (-1,)) <= 0):
        print('WARNING!!! input path_num must be real, scalar, integer, and positive')
        print('your input path_num = ', path_num)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(multipath_decay) or tr.numel(multipath_decay) != 1 or multipath_decay <= 0:
        print('WARNING!!! input multipath_decay must be real, scalar, and positive')
        print('your input multipath_decay = ', multipath_decay)
        raise ValueError('INPUT ERROR')

    # The main code part begins here
    nPath = tr.reshape(path_num, (-1,)).max()  # ; print('nPath = ', nPath)
    if tr.numel(dist) == 1:
        dist = tr.ones(ch_size.tolist()) * dist  # ; print('dist = ', dist)
    if tr.numel(path_num):
        path_num = tr.ones(ch_size.tolist()) * path_num  # ; print('path_num = ', path_num)

    # ----------- First path channel ----------- #
    first_pow_profile = path_loss_ref * ((dist / dist_ref) ** -path_loss_exp)
    NLOS_component = (tr.randn(ch_size.tolist()) + 1j * tr.randn(ch_size.tolist())) / (
          2 ** 0.5)  # ; print('NLOS_component = ', NLOS_component)
    first_tap_ch = (first_pow_profile ** 0.5) * ((Rician_fact / (1 + Rician_fact)) ** 0.5 * LOS_component + (
          1 / (1 + Rician_fact)) ** 0.5 * NLOS_component)
    # print('first_tap_ch = ', first_tap_ch)

    if nPath == 1:
        return first_tap_ch
    else:
        # ----------- Multi path channel ----------- #
        ch = tr.zeros([nPath, ch_size[0].tolist(), ch_size[1].tolist()], dtype=tr.complex128)
        ch[0, :, :] = first_tap_ch
        for ch_th in range(1, nPath):
            # print('------- ch_th = ', ch_th, ' -------')
            pow_profile = first_pow_profile * tr.exp(-ch_th / multipath_decay)  # ; print('pow_profile = ', pow_profile)
            pow_profile = pow_profile * (path_num - ch_th > 0)  # ; print('pow_profile = ', pow_profile)
            ch[ch_th, :, :] = (pow_profile ** 0.5) * (tr.randn(ch_size.tolist()) + 1j * tr.randn(ch_size.tolist())) / (
                  2 ** 0.5)
        return ch

# ---------- EXAMPLES ----------- #

def dB_to_lin(input):
    return 10 ** (input / 10)

def example1():
    """
    Given 4 Tx and 3 Rx antenna, create a random Rayleigh narrow-band MIMO channel with normalized distance
    i.e., H~CN(0, 1)
    """
    Tx = 4
    Rx = 3
    ch_size = tr.tensor([Rx, Tx])
    smpl_ch = uWave_ch(ch_size)
    print("----------- EXAMPLE 1 ----------- ")
    print('sampled channel = ', smpl_ch)

def check1():
    """Check if the generated channel for Rayleigh is CN(0, 1)"""
    Tx = 4; Rx = 3
    ch_size = tr.tensor([Rx, Tx])
    smpl_num = 10000
    ch = tr.zeros([smpl_num, Rx, Tx], dtype=tr.complex128)  # Buffer
    for smpl_th in range(smpl_num):
        ch[smpl_th, :, :] = uWave_ch(ch_size)

    print(" ")
    print("----------- CHECK 1 ----------- ")
    print('mean of the generated channel = ', tr.mean(ch, dim=0))
    print('variance of the generated channel = ', tr.var(ch, dim=0, unbiased=False))

def example2():
    """
    Given 4 Tx and 3 Rx antenna, create a random Rician narrow-band MIMO channel with normalize distance.
    # Rician factor = 10 dB
    # LoS component = random complex with unity absolute value
    """
    Tx = 4; Rx = 3
    ch_size = tr.tensor([Rx, Tx])
    Rician_fact = dB_to_lin(tr.tensor([10.]))
    angles = 2*tr.pi*(tr.rand(size=ch_size.tolist())-0.5)  # Uniform number from -pi to pi
    LOS_component = tr.exp(1j*angles)
    smpl_ch = uWave_ch(ch_size, LOS_component, Rician_fact)
    print(" ")
    print("----------- EXAMPLE 2 ----------- ")
    print('sampled channel = ', smpl_ch)

def check2():
    """
    As the Rician factor is higher, the channel tends to become deterministic. Hence, its variance decreases.
    Theoretically, it follows the following equation: variance = 1/(1+Rician_factor)*NLOS_variance.
    NLOS_variance, in this case, is one since the distances are normalized.
    Given Rician_factor = [-10, 0, 6.0206, 10] dB = [0.1, 1, 4, 10] in linear,
    then the variance of the generated Rician channel are [1/1.1, 1/2, 1/5, 1/11]
    = [0.90909, 0.50000, 0.20000, 0.09090].
    """
    Tx = 3
    Rx = 3
    ch_size = tr.tensor([Rx, Tx])
    angles = 2 * tr.pi * (tr.rand(size=ch_size.tolist()) - 0.5)  # Uniform number from -pi to pi
    LOS_component = tr.exp(1j * angles)
    Rician_fact_set = dB_to_lin(tr.tensor([-10, 0, 6.0206, 10]))
    smpl_num = 10000
    ch = tr.zeros([smpl_num, Rx, Tx], dtype=tr.complex128)  # Buffer
    var_theory = [0.90909, 0.50000, 0.20000, 0.09090]
    print(" ")
    print("----------- CHECK 2 ----------- ")
    for ric_th in range(Rician_fact_set.numel()):
        Rician_fact = Rician_fact_set[ric_th]; print('Rician_fact = ', Rician_fact)
        print('Theoretical Variance = ', var_theory[ric_th])
        for smpl_th in range(smpl_num):
            ch[smpl_th, :, :] = uWave_ch(ch_size, LOS_component, Rician_fact)
        print('Numerical Variance = ', tr.var(ch, dim=0, unbiased=True))
        print('-------')

def example3():
    """
    Given 4 Tx and 3 Rx antenna, create a random Rician narrow-band MIMO channel with the following specifications.
    # Rician factor = 6.0206 dB
    # Los component = random complex with unity absolute value
    # distance = 100 m
    # path loss exponent = 2
    # reference distance = 1 m
    # path loss at the reference distance = -20 dB
    """
    Tx = 4; Rx = 3
    ch_size = tr.tensor([Rx, Tx])
    Rician_fact = dB_to_lin(tr.tensor([6.0206]))
    angles = 2 * tr.pi * (tr.rand(size=ch_size.tolist()) - 0.5)  # Uniform number from -pi to pi
    LOS_component = tr.exp(1j * angles)
    dist = tr.tensor([100.])
    dist_ref = tr.tensor([1.])
    path_loss_ref = dB_to_lin(tr.tensor([-20.]))
    path_loss_exp = tr.tensor([2.])
    smpl_ch = uWave_ch(ch_size, LOS_component, Rician_fact, dist, dist_ref, path_loss_ref, path_loss_exp)
    print(" ")
    print("----------- EXAMPLE 3 ----------- ")
    print('sampled channel = ', smpl_ch)

def check3():
    """
    The path loss at distance d can be expressed as P = Pl0 * (d/d0)**a, where d0 is the reference distance,
    Pl0 is the path loss at d0, d is the distance, and a is the path loss exponent. Suppose Pl0 = -30 dB, d0 = 1 m, and
    a = 2. Then, the path loss at:
    d = 1 -> P = 1e-3
    d = 2 -> P = 2.5e-4
    d = 3 -> P = 1.111e-4
    d = 4 -> P = 6.25e-5
    d = 5 -> P = 4e-5
    d = 6 -> P = 2.777e-5
    """
    Tx = 3; Rx = 2
    ch_size = tr.tensor([Rx, Tx])
    Rician_fact = dB_to_lin(tr.tensor([6.0206]))
    angles = 2 * tr.pi * (tr.rand(size=ch_size.tolist()) - 0.5)  # Uniform number from -pi to pi
    LOS_component = tr.exp(1j * angles)
    dist = tr.tensor([[1, 2, 3], [4, 5, 6]])  # we can input different distance for each antenna-pair
    dist_ref = tr.tensor([1.])
    path_loss_ref = dB_to_lin(tr.tensor([-30.]))
    path_loss_exp = tr.tensor([2.])
    smpl_num = 10000
    path_loss_theory = [[1e-3, 2.5e-4, 1.111e-4], [6.25e-5, 4e-5, 2.777e-5]]
    ch = tr.zeros([smpl_num, Rx, Tx], dtype=tr.complex128)
    for smpl_th in range(smpl_num):
        ch[smpl_th, :, :] = uWave_ch(ch_size, LOS_component, Rician_fact, dist, dist_ref, path_loss_ref, path_loss_exp)
    print(" ")
    print("----------- CHECK 3 ----------- ")
    print('average channel power = ', tr.mean(tr.abs(ch)**2, dim=0))
    print('path loss theory = ', path_loss_theory)

def example4():
    """
    Given 2 Tx and 3 Rx, create create a random wide-band MIMO channel (multipath impulse response) with the following
    specifications:
    # Rician factor (of the first tap impulse response) = 6.0206 dB
    # Los component = random complex with unity absolute value
    # distances = [[1, 2, 3][4, 5, 6]]
    # path loss exponent = 2
    # reference distance = 1 m
    # path loss at the reference distance = -30 dB
    # number of delay tap = [[3, 3, 4], [4, 5, 6]] (each antenna pair has different number of delay tap)
    # multipath decay factor = 2
    """
    Tx = 3; Rx = 2
    ch_size = tr.tensor([Rx, Tx])
    Rician_fact = dB_to_lin(tr.tensor([6.0206]))
    angles = 2 * tr.pi * (tr.rand(size=ch_size.tolist()) - 0.5)  # Uniform number from -pi to pi
    LOS_component = tr.exp(1j * angles)
    dist = tr.tensor([[1., 2., 3.], [4., 5., 6.]])
    dist_ref = tr.tensor([1.])
    path_loss_ref = dB_to_lin(tr.tensor([-30.]))
    path_loss_exp = tr.tensor([2.])
    path_num = tr.tensor([[3, 3, 4], [4, 5, 5]])
    multipath_decay = tr.tensor([2.])
    smpl_ch = uWave_ch(ch_size, LOS_component, Rician_fact, dist, dist_ref, path_loss_ref, path_loss_exp, path_num, multipath_decay)
    print(" ")
    print("----------- EXAMPLE 4 ----------- ")
    print('sampled channel = ', smpl_ch)

def check4():
    """
    With the settings the same as in check3(), but now we consider multipath wide band channel.
    The channel power at the t-th tap is expressed as: p(t) = p0 * (1/decay_fact) * exp(-t/decay_fact),
    where p0 is the power at the zero-th path. Suppose the maximum number of tap channel is 5.
    Hence, the average power delay profile of the MIMO channel for all tap channels are:
    tap_th        =     1          2           3           4           5
    Rx1-Tx1 (1 m) = 1.0000e-3, 6.0653e-04, 3.6788e-04, 2.2313e-04, 1.3534e-04
    Rx1-Tx2 (2 m) = 2.5000e-4, 1.5163e-04, 9.1970e-05, 5.5783e-05, 3.3834e-05
    Rx1-Tx3 (3 m) = 1.1111e-4, 6.7392e-05, 4.0875e-05, 2.4792e-05, 1.5037e-05
    Rx2-Tx1 (4 m) = 6.2500e-5, 3.7908e-05, 2.2992e-05, 1.3946e-05, 8.4585e-06
    Rx2-Tx2 (5 m) = 4.0000e-5, 2.4261e-05, 1.4715e-05, 8.9252e-06, 5.4134e-06
    Rx2-Tx3 (6 m) = 2.7776e-5, 1.6848e-05, 1.0219e-05, 6.1981e-06, 3.7593e-06
    """
    Tx = 3; Rx = 2
    ch_size = tr.tensor([Rx, Tx])
    Rician_fact = dB_to_lin(tr.tensor([6.0206]))
    angles = 2 * tr.pi * (tr.rand(size=ch_size.tolist()) - 0.5)  # Uniform number from -pi to pi
    LOS_component = tr.exp(1j * angles)
    dist = tr.tensor([[1., 2., 3.], [4., 5., 6.]])
    dist_ref = tr.tensor([1.])
    path_loss_ref = dB_to_lin(tr.tensor([-30.]))
    path_loss_exp = tr.tensor([2.])
    path_num = tr.tensor([[5, 5, 4], [4, 3, 3]])
    multipath_decay = tr.tensor([2.])
    smpl_num = 10000
    max_path_number = tr.max(tr.reshape(path_num, (-1,)))  # 5
    ch = tr.zeros([smpl_num, max_path_number, Rx, Tx], dtype=tr.complex128)
    for smpl_th in range(smpl_num):
        ch[smpl_th, :, :, :] = uWave_ch(ch_size, LOS_component, Rician_fact, dist, dist_ref, path_loss_ref, path_loss_exp, path_num, multipath_decay)
    print(" ")
    print("----------- CHECK 4 ----------- ")
    print('average channel power = ', tr.mean(tr.abs(ch)**2, dim=0))
    print("Note that some tap power profiles are zero according to 'path_num' variables")


example1()
check1()
example2()
check2()
example3()
check3()
example4()
check4()
