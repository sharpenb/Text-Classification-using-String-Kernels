import numpy as np

def ssk(s, t, n, m_lambda):
    s_len = len(s)
    t_len = len(t)

    # Initializations
    Kp = np.zeros((n+1, s_len, t_len))
    Kp[0] = np.ones((s_len, t_len))

    
    # Compute the K' and K'' helper kernels
    for i in range(n):
        for j in range(s_len - 1):
            Kpp = 0
            for k in range(t_len - 1):
                # if s[j] == t[k]:
                #     Kpp = m_lambda * (Kpp + m_lambda * Kp[i, j, k])
                # else:
                #     Kpp *= m_lambda
                Kpp = m_lambda * (Kpp + (s[j] == t[k]) * m_lambda * Kp[i, j, k])
                Kp[i+1, j+1, k+1] = m_lambda * Kp[i+1, j, k+1] + Kpp

    # Compute the real kernel K
    K = 0
    for j in range(s_len):
        for k in range(t_len):
            K += (s[j] == t[k]) * m_lambda * m_lambda * Kp[n, j, k]

    ## Kernel computation for varying string length
    # K = 0
    # for i in range(n):
    #     for j in range(s_len):
    #         for k in range(t_len):
    #             if s[j] == t[k]:
    #                 K += m_lambda * m_lambda * Kp[i, j, k]
    #             K += (s[j] == t[k]) * m_lambda * m_lambda * Kp[i, j, k]

    return K
            

string1 = 'a'
string2 = 'abcde'
lambda1 = 0.1
for i in range(0, 6):
    print(ssk(string1, string2, i, lambda1))
