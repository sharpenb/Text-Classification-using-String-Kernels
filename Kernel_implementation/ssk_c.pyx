import cython
import numpy as np
cimport numpy as np


@cython.nonecheck(False)
cpdef float ssk(s, t, int n, float m_lambda):
    cdef int s_len = len(s)
    cdef int t_len = len(t)

    # Initializations
    cdef np.ndarray[np.double_t, ndim=3] Kp = np.zeros((n+1, s_len, t_len))
    Kp[0] = np.ones((s_len, t_len))

    
    # Compute the K' and K'' helper kernels
    cdef int i, j, k
    cdef float Kpp = 0
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
    cdef float K = 0
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
            

# string1 = 'a'
# string2 = 'abcde'
# lambda1 = 0.1
# for i in range(0, 6):
#     print(ssk(string1, string2, i, lambda1))
