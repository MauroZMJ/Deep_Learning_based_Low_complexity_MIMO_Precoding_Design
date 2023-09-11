import numpy as np
from sympy import Symbol

Nt = Symbol('Nt')
Nr = Symbol('Nr')
K = Symbol('K')
dk = Symbol('dk')
RB = Symbol('RB')
It = Symbol('It')

def MM(M,N,T,triangle_a=False,triangle_b=False):
    if triangle_a:
        return N*T
    else:
        if triangle_b:
            return M*N
        else:
            return 2*M*N*T

def MI(M,hermitian):
    if hermitian:
        return 2*M**3/3 + 3*M**2/2 + 5*M/6
    else:
        return 2*M**3
    
def MA(M,N): # matrix addition
    return M*N

def MC(M,N): # matrix x constant
    return M*N

def SVD(M,N,dk):
#    assert M>=N # for numerical 
    return 2*M*N**2 + 4*N**3 + dk * ( 2 * M * N + M )
    #return 2*M*N**2 + 11*N**3

def Tr(M):
    return M-1
def MN(M):
    #vector L2 norm 
    return 2*M-1

def compute_EZF_flops(Nt,Nr,K,dk,RB):
    # RB信道求和
    flops1 = K*(RB*MM(Nr,Nt,Nr)+(RB-1)*MA(Nt,Nt))
    # 信道SVD分解
    flops2 = K*SVD(Nt,Nt,dk)
    # 最终预编码矩阵
    flops3 = MM(K*dk,Nt,K*dk)+MI(K*dk,True)+MM(Nt,K*dk,K*dk)
    flops = flops1 + flops2 + flops3
    return flops*4 # x4 for real flops
def compute_EZF_flops_single_RB(Nt,Nr,K,dk):
    flops1 = K*SVD(Nt,Nr,dk)
    flops2 = MM(K*dk,Nt,K*dk)+MI(K*dk,True)+MM(Nt,K*dk,K*dk)
    flops = flops1 + flops2
    return flops*4

EZF_flops = compute_EZF_flops(Nt,Nr,K,dk,RB)


def compute_WMMSE_flops(It,Nt,Nr,K,dk,RB):
    #更新U
    flops1 = K*(MM(Nt,dk,Nt)+Tr(Nt)) + MC(Nr,Nr) + MA(Nr,Nr)
    flops2 = K*(MM(Nr,Nt,dk)+MM(Nr,dk,Nr)) + (K-1)*MA(Nr,Nr)
    flops3 = MI(Nr,True)+MM(Nr,Nr,Nt)+MM(Nr,Nt,dk)
    flops_U = K*RB*(flops1+flops2+flops3)
    #更新W
    flops1 = MM(dk,Nr,Nt)+MM(dk,Nt,dk)+MA(dk,dk)
    flops2 = MI(dk,False)
    flops_W = K*RB*(flops1+flops2)
    #更新V
    flops1 = K*RB*(MM(Nr,dk,dk)+MM(Nr,dk,Nr)+Tr(Nr))+MC(Nt,Nt)
    flops2 = K*RB*(MM(Nt,Nr,dk)+MM(Nt,dk,dk)+MM(Nt,dk,Nr)+MM(Nt,Nr,Nt)) + (K*RB-1)*MA(Nt,Nt)
    flops3 = RB*(MM(Nt,Nr,dk)+MM(Nt,dk,dk))+(RB-1)*MA(Nt,dk)
    flops4 = MI(Nt,False)+MM(Nt,Nt,dk)
    flops_V = K*(flops1+flops2+flops3+flops4)
    
    flops_per_it = flops_U + flops_W + flops_V
    flops = It*flops_per_it
    return flops*4 # x4 for real flops
def compute_lcp_restore_flops(Nt,Nr,K,dk,RB):
    flops1 = (1+(MM(Nt,1,Nt)+MC(Nt,Nt))*K*dk*RB+(K*dk*RB-1)*MA(Nt,Nt)) + MI(Nt,hermitian=True) + K*dk*(MC(Nt,1)+(RB-1)*MA(Nt,1)+MM(Nt,Nt,1)+MN(Nt)+MC(Nt,1))
    flops2 = K*RB*(SVD(Nt,Nr,dk)+dk*MC(Nt,1))
    return (flops1+flops2)*4
def compute_lcp_restore_flops_single_RB(Nt,Nr,K,dk):
    M = K * dk
    #MIMO2MISO
    flops1 = K*(SVD(Nt,Nr,dk)+dk*MC(Nt,1))
    #input preprocessing
    flops2 = MM(M,Nt,M)
    #restore module
    flops3 = (1+MM(Nt,M,M,triangle_b=True)+MM(M,Nt,M)+M+MI(M,hermitian=True)+MM(Nt,M,M)+MM(Nt,M,M,triangle_b=True)+3*Nt*M+MM(Nt,M,M,triangle_b=True)+2*M)
    flops = flops1 + flops2+ flops3
    return 4*flops
def LUW_restore_flops(Nt,Nr,K,dk,RB):
    #input preprocessing
    flops_input_pre = MM(K*RB*Nr,Nt,K*RB*Nr)
    #output restore
    flops1 = K*RB*(MM(Nr,dk,dk)+MM(Nr,dk,Nr)+Tr(Nr))+MC(Nt,Nt)
    flops2 = K*RB*(MM(Nt,Nr,dk)+MM(Nt,dk,dk)+MM(Nt,dk,Nr)+MM(Nt,Nr,Nt)) + (K*RB-1)*MA(Nt,Nt)
    flops3 = RB*(MM(Nt,Nr,dk)+MM(Nt,dk,dk))+(RB-1)*MA(Nt,dk)
    flops4 = MI(Nt,False)+MM(Nt,Nt,dk)
    flops_V = K*(flops1+flops2+flops3+flops4)
    flops = flops_input_pre + flops_V
    return flops*4

EZF_flops_multi_RB = compute_EZF_flops(Nt,Nr,K,dk,RB)
EZF_flops_single_RB = compute_EZF_flops_single_RB(Nt,Nr,K,dk)
WMMSE_flops = compute_WMMSE_flops(It,Nt,Nr,K,dk,RB)
lcp_restore_flops_multi_RB = compute_lcp_restore_flops(Nt,Nr,K,dk,RB)
lcp_restore_flops_single_RB = compute_lcp_restore_flops_single_RB(Nt,Nr,K,dk)
luw_restore_flops = LUW_restore_flops(Nt,Nr,K,dk,RB)

# substitute typical system parameters
Nt = 64
Nr = 4
# dk = 2
# K = 10
RB = 2
It = 20

# # 手动复制上面的symbolic表达式
# EZF_flops_multi_RB_numerical = 8*K**3*dk**3/3 + 16*K**2*Nt*dk**2 + 6*K**2*dk**2 + 10*K*dk/3 + 4*K*(6*Nt**3 + dk*(2*Nt**2 + Nt)) + 4*K*(2*Nr**2*Nt*RB + Nt**2*(RB - 1))
# EZF_flops_single_RB_numerical = 8*K**3*dk**3/3 + 16*K**2*Nt*dk**2 + 6*K**2*dk**2 + 10*K*dk/3 + 4*K*(4*Nr**3 + 2*Nr**2*Nt + dk*(2*Nr*Nt + Nt))
# WMMSE_flops_numerical = 4*It*(K*RB*(2*Nr*Nt*dk + 2*Nt*dk**2 + 2*dk**3 + dk**2) + K*RB*(K*(2*Nr**2*dk + 2*Nr*Nt*dk) + K*(2*Nt**2*dk + Nt - 1) + 2*Nr**3/3 + 2*Nr**2*Nt + Nr**2*(K - 1) + 7*Nr**2/2 + 2*Nr*Nt*dk + 5*Nr/6) + K*(K*RB*(2*Nr*Nt**2 + 4*Nr*Nt*dk + 2*Nt*dk**2) + K*RB*(2*Nr**2*dk + 2*Nr*dk**2 + Nr - 1) + 2*Nt**3 + 2*Nt**2*dk + Nt**2*(K*RB - 1) + Nt**2 + Nt*dk*(RB - 1) + RB*(2*Nr*Nt*dk + 2*Nt*dk**2)))
# lcp_restore_flops_multi_RB_numerical = 12*K*Nt**2*RB*dk + 4*K*RB*(4*Nr**3 + 2*Nr**2*Nt + Nt*dk + dk*(2*Nr*Nt + Nt)) + 4*K*dk*(2*Nt**2 + Nt*(RB - 1) + 4*Nt - 1) + 8*Nt**3/3 + 4*Nt**2*(K*RB*dk - 1) + 6*Nt**2 + 10*Nt/3 + 4
# lcp_restore_flops_single_RB_numerical = 8*K**3*dk**3/3 + 24*K**2*Nt*dk**2 + 6*K**2*dk**2 + 24*K*Nt*dk + 46*K*dk/3 + 4*K*(4*Nr**3 + 2*Nr**2*Nt + Nt*dk + dk*(2*Nr*Nt + Nt)) + 4
# luw_restore_flops_numerical = 8*K**2*Nr**2*Nt*RB**2 + 4*K*(K*RB*(2*Nr*Nt**2 + 4*Nr*Nt*dk + 2*Nt*dk**2) + K*RB*(2*Nr**2*dk + 2*Nr*dk**2 + Nr - 1) + 2*Nt**3 + 2*Nt**2*dk + Nt**2*(K*RB - 1) + Nt**2 + Nt*dk*(RB - 1) + RB*(2*Nr*Nt*dk + 2*Nt*dk**2))
EZF_flops_list = []
WMMSE_flops_list = []
WMMSE_flops_1_round_list = []
LCP_restore_flops_list = []
LUW_restore_flops_list = []
#dk_list = [1,2,3,4]
dk_list = [2]
K_list = [10]
K_list = [8,9,10,11,12,13,14,15,16]
It = 50
if RB == 1:
    for dk in dk_list:
        for K in K_list:
            EZF_flops_list.append(8*K**3*dk**3/3 + 16*K**2*Nt*dk**2 + 6*K**2*dk**2 + 10*K*dk/3 + 4*K*(4*Nr**3 + 2*Nr**2*Nt + dk*(2*Nr*Nt + Nt)))
            WMMSE_flops_list.append(4*It*(K*RB*(2*Nr*Nt*dk + 2*Nt*dk**2 + 2*dk**3 + dk**2) + K*RB*(K*(2*Nr**2*dk + 2*Nr*Nt*dk) + K*(2*Nt**2*dk + Nt - 1) + 2*Nr**3/3 + 2*Nr**2*Nt + Nr**2*(K - 1) + 7*Nr**2/2 + 2*Nr*Nt*dk + 5*Nr/6) + K*(K*RB*(2*Nr*Nt**2 + 4*Nr*Nt*dk + 2*Nt*dk**2) + K*RB*(2*Nr**2*dk + 2*Nr*dk**2 + Nr - 1) + 2*Nt**3 + 2*Nt**2*dk + Nt**2*(K*RB - 1) + Nt**2 + Nt*dk*(RB - 1) + RB*(2*Nr*Nt*dk + 2*Nt*dk**2))))
            WMMSE_flops_1_round_list.append(4*(K*RB*(2*Nr*Nt*dk + 2*Nt*dk**2 + 2*dk**3 + dk**2) + K*RB*(K*(2*Nr**2*dk + 2*Nr*Nt*dk) + K*(2*Nt**2*dk + Nt - 1) + 2*Nr**3/3 + 2*Nr**2*Nt + Nr**2*(K - 1) + 7*Nr**2/2 + 2*Nr*Nt*dk + 5*Nr/6) + K*(K*RB*(2*Nr*Nt**2 + 4*Nr*Nt*dk + 2*Nt*dk**2) + K*RB*(2*Nr**2*dk + 2*Nr*dk**2 + Nr - 1) + 2*Nt**3 + 2*Nt**2*dk + Nt**2*(K*RB - 1) + Nt**2 + Nt*dk*(RB - 1) + RB*(2*Nr*Nt*dk + 2*Nt*dk**2))))

            LUW_restore_flops_list.append(8*K**2*Nr**2*Nt*RB**2 + 4*K*(K*RB*(2*Nr*Nt**2 + 4*Nr*Nt*dk + 2*Nt*dk**2) + K*RB*(2*Nr**2*dk + 2*Nr*dk**2 + Nr - 1) + 2*Nt**3 + 2*Nt**2*dk + Nt**2*(K*RB - 1) + Nt**2 + Nt*dk*(RB - 1) + RB*(2*Nr*Nt*dk + 2*Nt*dk**2)))
            LCP_restore_flops_list.append(8*K**3*dk**3/3 + 24*K**2*Nt*dk**2 + 6*K**2*dk**2 + 24*K*Nt*dk + 46*K*dk/3 + 4*K*(4*Nr**3 + 2*Nr**2*Nt + Nt*dk + dk*(2*Nr*Nt + Nt)) + 4)
else:
    for dk in dk_list:
        for K in K_list:
            EZF_flops_list.append(8*K**3*dk**3/3 + 16*K**2*Nt*dk**2 + 6*K**2*dk**2 + 10*K*dk/3 + 4*K*(6*Nt**3 + dk*(2*Nt**2 + Nt)) + 4*K*(2*Nr**2*Nt*RB + Nt**2*(RB - 1)))
            WMMSE_flops_list.append(4*It*(K*RB*(2*Nr*Nt*dk + 2*Nt*dk**2 + 2*dk**3 + dk**2) + K*RB*(K*(2*Nr**2*dk + 2*Nr*Nt*dk) + K*(2*Nt**2*dk + Nt - 1) + 2*Nr**3/3 + 2*Nr**2*Nt + Nr**2*(K - 1) + 7*Nr**2/2 + 2*Nr*Nt*dk + 5*Nr/6) + K*(K*RB*(2*Nr*Nt**2 + 4*Nr*Nt*dk + 2*Nt*dk**2) + K*RB*(2*Nr**2*dk + 2*Nr*dk**2 + Nr - 1) + 2*Nt**3 + 2*Nt**2*dk + Nt**2*(K*RB - 1) + Nt**2 + Nt*dk*(RB - 1) + RB*(2*Nr*Nt*dk + 2*Nt*dk**2))))
            WMMSE_flops_1_round_list.append(4*(K*RB*(2*Nr*Nt*dk + 2*Nt*dk**2 + 2*dk**3 + dk**2) + K*RB*(K*(2*Nr**2*dk + 2*Nr*Nt*dk) + K*(2*Nt**2*dk + Nt - 1) + 2*Nr**3/3 + 2*Nr**2*Nt + Nr**2*(K - 1) + 7*Nr**2/2 + 2*Nr*Nt*dk + 5*Nr/6) + K*(K*RB*(2*Nr*Nt**2 + 4*Nr*Nt*dk + 2*Nt*dk**2) + K*RB*(2*Nr**2*dk + 2*Nr*dk**2 + Nr - 1) + 2*Nt**3 + 2*Nt**2*dk + Nt**2*(K*RB - 1) + Nt**2 + Nt*dk*(RB - 1) + RB*(2*Nr*Nt*dk + 2*Nt*dk**2))))

            LUW_restore_flops_list.append(8*K**2*Nr**2*Nt*RB**2 + 4*K*(K*RB*(2*Nr*Nt**2 + 4*Nr*Nt*dk + 2*Nt*dk**2) + K*RB*(2*Nr**2*dk + 2*Nr*dk**2 + Nr - 1) + 2*Nt**3 + 2*Nt**2*dk + Nt**2*(K*RB - 1) + Nt**2 + Nt*dk*(RB - 1) + RB*(2*Nr*Nt*dk + 2*Nt*dk**2)))
            LCP_restore_flops_list.append(12*K*Nt**2*RB*dk + 4*K*RB*(4*Nr**3 + 2*Nr**2*Nt + Nt*dk + dk*(2*Nr*Nt + Nt)) + 4*K*dk*(2*Nt**2 + Nt*(RB - 1) + 4*Nt - 1) + 8*Nt**3/3 + 4*Nt**2*(K*RB*dk - 1) + 6*Nt**2 + 10*Nt/3 + 4)

print(EZF_flops_list)
print(WMMSE_flops_1_round_list)
print(WMMSE_flops_list)
print(LUW_restore_flops_list)
print(LCP_restore_flops_list)

# flops_ratio = WMMSE_flops_numerical/EZF_flops_numerical
# flops_ratio_per_iter = flops_ratio/It
# restore_flops_ratio = restore_flops_numerical/EZF_flops_numerical
# print(flops_ratio_per_iter)
# print(flops_ratio)
# import ipdb;ipdb.set_trace()
