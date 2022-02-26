import torch
import wireless_channel as wc

def example1():
    """
    Given 4 Tx and 3 Rx antenna, create a random Rayleigh narrow-band MIMO channel with normalized distance
    i.e., H~CN(0, 1)
    """
    Tx = 4
    Rx = 3
    ch_size = torch.tensor([Rx, Tx])
    smpl_ch = wc.uWave_ch(ch_size)
    print('sampled channel = ', smpl_ch)
def check1():
    """Check if the generated channel for Rayleigh is CN(0, 1)"""
    Tx = 4; Rx = 3
    ch_size = torch.tensor([Rx, Tx])
    smpl_num = 10000
    ch = torch.zeros([smpl_num, Rx, Tx], dtype=torch.complex128)  # Buffer
    for smpl_th in range(smpl_num):
        ch[smpl_th, :, :] = wc.uWave_ch(ch_size)

    print('mean of the generated channel = ', torch.mean(ch, dim=0))
    print('variance of the generated channel = ', torch.var(ch, dim=0, unbiased=False))
def example2():
    """
    Given 4 Tx and 3 Rx antenna, create a random Rician narrow-band MIMO channel with normalize distance.
    # Rician factor = 10 dB
    # LoS component = random complex with unity absolute value
    """
    Tx = 4; Rx = 3
    ch_size = torch.tensor([Rx, Tx])
    Rician_fact = wc.dB_to_lin(torch.tensor([10.]))
    angles = 2*torch.pi*(torch.rand(size=ch_size.tolist())-0.5)  # Uniform number from -pi to pi
    LOS_component = torch.exp(1j*angles)
    smpl_ch = wc.uWave_ch(ch_size, LOS_component, Rician_fact)
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
    ch_size = torch.tensor([Rx, Tx])
    angles = 2 * torch.pi * (torch.rand(size=ch_size.tolist()) - 0.5)  # Uniform number from -pi to pi
    LOS_component = torch.exp(1j * angles)
    Rician_fact_set = wc.dB_to_lin(torch.tensor([-10, 0, 6.0206, 10]))
    smpl_num = 10000
    ch = torch.zeros([smpl_num, Rx, Tx], dtype=torch.complex128)  # Buffer
    var_theory = [0.90909, 0.50000, 0.20000, 0.09090]
    for ric_th in range(Rician_fact_set.numel()):
        print('-------------------------------')
        Rician_fact = Rician_fact_set[ric_th]; print('Rician_fact = ', Rician_fact)
        print('Theoretical Variance = ', var_theory[ric_th])
        for smpl_th in range(smpl_num):
            ch[smpl_th, :, :] = wc.uWave_ch(ch_size, LOS_component, Rician_fact)
        print('Numerical Variance = ', torch.var(ch, dim=0, unbiased=True))
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
    ch_size = torch.tensor([Rx, Tx])
    Rician_fact = wc.dB_to_lin(torch.tensor([6.0206]))
    angles = 2 * torch.pi * (torch.rand(size=ch_size.tolist()) - 0.5)  # Uniform number from -pi to pi
    LOS_component = torch.exp(1j * angles)
    dist = torch.tensor([100.])
    dist_ref = torch.tensor([1.])
    path_loss_ref = wc.dB_to_lin(torch.tensor([-20.]))
    path_loss_exp = torch.tensor([2.])
    smpl_ch = wc.uWave_ch(ch_size, LOS_component, Rician_fact, dist, dist_ref, path_loss_ref, path_loss_exp)
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
    ch_size = torch.tensor([Rx, Tx])
    Rician_fact = wc.dB_to_lin(torch.tensor([6.0206]))
    angles = 2 * torch.pi * (torch.rand(size=ch_size.tolist()) - 0.5)  # Uniform number from -pi to pi
    LOS_component = torch.exp(1j * angles)
    dist = torch.tensor([[1, 2, 3], [4, 5, 6]])  # we can input different distance for each antenna-pair
    dist_ref = torch.tensor([1.])
    path_loss_ref = wc.dB_to_lin(torch.tensor([-30.]))
    path_loss_exp = torch.tensor([2.])
    smpl_num = 10000
    path_loss_theory = [[1e-3, 2.5e-4, 1.111e-4], [6.25e-5, 4e-5, 2.777e-5]]
    ch = torch.zeros([smpl_num, Rx, Tx], dtype=torch.complex128)
    for smpl_th in range(smpl_num):
        ch[smpl_th, :, :] = wc.uWave_ch(ch_size, LOS_component, Rician_fact, dist, dist_ref, path_loss_ref, path_loss_exp)
    print('average channel power = ', torch.mean(torch.abs(ch)**2, dim=0))
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
    ch_size = torch.tensor([Rx, Tx])
    Rician_fact = wc.dB_to_lin(torch.tensor([6.0206]))
    angles = 2 * torch.pi * (torch.rand(size=ch_size.tolist()) - 0.5)  # Uniform number from -pi to pi
    LOS_component = torch.exp(1j * angles)
    dist = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    dist_ref = torch.tensor([1.])
    path_loss_ref = wc.dB_to_lin(torch.tensor([-30.]))
    path_loss_exp = torch.tensor([2.])
    path_num = torch.tensor([[3, 3, 4], [4, 5, 5]])
    multipath_decay = torch.tensor([2.])
    smpl_ch = wc.uWave_ch(ch_size, LOS_component, Rician_fact, dist, dist_ref, path_loss_ref, path_loss_exp, path_num, multipath_decay)
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
    ch_size = torch.tensor([Rx, Tx])
    Rician_fact = wc.dB_to_lin(torch.tensor([6.0206]))
    angles = 2 * torch.pi * (torch.rand(size=ch_size.tolist()) - 0.5)  # Uniform number from -pi to pi
    LOS_component = torch.exp(1j * angles)
    dist = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    dist_ref = torch.tensor([1.])
    path_loss_ref = wc.dB_to_lin(torch.tensor([-30.]))
    path_loss_exp = torch.tensor([2.])
    path_num = torch.tensor([[5, 5, 4], [4, 3, 3]])
    multipath_decay = torch.tensor([2.])
    smpl_num = 10000
    max_path_number = torch.max(torch.reshape(path_num, (-1,)))  # 5
    ch = torch.zeros([smpl_num, max_path_number, Rx, Tx], dtype=torch.complex128)
    for smpl_th in range(smpl_num):
        ch[smpl_th, :, :, :] = wc.uWave_ch(ch_size, LOS_component, Rician_fact, dist, dist_ref, path_loss_ref, path_loss_exp, path_num, multipath_decay)
    print('average channel power = ', torch.mean(torch.abs(ch)**2, dim=0))
    print("Note that some tap power profiles are zero according to 'path_num' variables")


example1()
check1()
example2()
check2()
example3()
check3()
example4()
check4()
