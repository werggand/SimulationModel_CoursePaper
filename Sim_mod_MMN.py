import matplotlib.pyplot as plt
import math
import random
import decimal

decimal.getcontext().prec = 100

X = [2, 1]
R = 250
F = 2
N = [math.floor(R / X[0]), math.floor(R / X[1])]


def fact_2(n):
    p = 1
    for i in range(2, n + 1):
        p *= i
    return p


def P(C, x, ro):
    result = C
    for f in range(F):
        result *= (ro[f] ** x[f]) / fact_2(x[f])
    return result


def main(ro):
    ''' Множество состояний системы '''
    NN = []
    for n1 in range(N[0]):
        for n2 in range(N[1]):
            n = [n1, n2]
            if n[0] * X[0] + n[1] * X[1] <= R:
                NN.append(n)
    ''' Множество состояний сброса'''
    B = [0 for f in range(F)]
    for f in range(F):
        B[f] = []
        for n in NN:
            test_NN = []
            e = [0 for f in range(F)]
            e[f] = 1
            for i in range(F):
                test_NN.append(n[i] + e[i])
            if test_NN not in NN:
                B[f].append(n)

    ''' Константа'''
    const = 0
    for n in NN:
        s = 1
        for f in range(F):
            s *= (ro[f] ** n[f]) / math.factorial(n[f])

        const += s
    const = 1 / const

    '''Распределение вероятностей'''
    result = []
    for n in NN:
        result.append(P(const, n, ro))

    '''Вероятность Сброса'''
    pB = [0 for i in range(F)]
    for f in range(F):
        for n in B[f]:
            pB[f] += P(const, n, ro)

    ''' Мат ожидание числа заявок на обслуживании'''
    M = [0 for f in range(F)]
    for f in range(F):
        for n in NN:
            M[f] += n[f] * P(const, n, ro)

    ''' Мат ожидание кол-ва занятого ресурса '''
    MR = [0 for f in range(F)]
    for f in range(F):
        for n in NN:
            MR[f] += n[f] * X[f] * P(const, n, ro)

    '''Коэф. использовния ресурса'''
    MR_S = 0
    for f in range(F):
        MR_S += MR[f]
    k = MR_S / R

    pn1 = [0 for i in range(N[0] + 1)]
    for n1 in range(N[0] + 1):  # распр вер одн случ
        for n in NN:
            if n1 == n[0]:
                pn1[n1] += P(const, n, ro)

    oX = [i for i in range(N[0] + 1)]

    return oX, pn1, pB[0], pB[1], M[0], M[1], MR[0], MR[1], k


NumbOfExp = 100  #Количетсво эксперимнтов

def Exp(parametr):
    X = -(1 / parametr) * decimal.Decimal(math.log(random.uniform(0, 1)))
    return X


def Simulation(lmb, mu):
    count_n1 = 0
    count_n2 = 0
    done_n1 = 0
    done_n2 = 0
    pB1 = 0
    pB2 = 0
    t0 = 0
    n1 = 0
    n2 = 0
    drop_n1 = 0
    drop_n2 = 0
    n = [n1, n2]
    period1 = [0 for i in range(N[0] + 1)]
    period2 = [0 for i in range(N[1] + 1)]

    while min(done_n1, done_n2) <= NumbOfExp:
        Par = [decimal.Decimal(0) for i in range(4)]
        parameters = []
        Par[0] = Exp(decimal.Decimal(lmb[0]))
        Par[1] = Exp(decimal.Decimal(lmb[1]))
        if n1 != 0:
            Par[2] = Exp(decimal.Decimal(mu[0]) * n1)
        if n2 != 0:
            Par[3] = Exp(decimal.Decimal(mu[1]) * n2)
        for k in range (4):
          if Par[k] != 0:
            parameters.append(Par[k])
        Action = min(parameters)
        if Action == Par[0]:
            if (n1 + 1) * X[0] + n2 * X[1] <= R:
                period1[n1] += Action
                period2[n2] += Action
                t0 += Action
                n1 += 1
                count_n1 += 1
            else:
                period1[n1] += Action
                period2[n2] += Action
                t0 += Action
                drop_n1 += 1
                count_n1 += 1
        if Action == Par[1]:
            if n1 * X[0] + (n2 + 1) * X[1] <= R:
                period1[n1] += Action
                period2[n2] += Action
                t0 += Action
                n2 += 1
                count_n2 += 1
            else:
                period1[n1] += Action
                period2[n2] += Action
                t0 += Action
                drop_n2 += 1
                count_n2 += 1
        if Action == Par[2]:
            if n1 != 0:
                period1[n1] += Action
                period2[n2] += Action
                t0 += Action
                n1 -= 1
                done_n1 += 1
        if Action == Par[3]:
            if n2 != 0:
                period1[n1] += Action
                period2[n2] += Action
                t0 += Action
                n2 -= 1
                done_n2 += 1
        n = [n1, n2]

    freq1 = [decimal.Decimal(0) for g in range(N[0] + 1)]
    freq2 = [decimal.Decimal(0) for g in range(N[1] + 1)]

    '''Считаем Частоты'''
    for j in range(N[0] + 1):
        freq1[j] = period1[j] / t0
    for j in range(N[1] + 1):
        freq2[j] = period2[j] / t0
    '''Вероятность сброса'''
    pB1 = drop_n1 / count_n1
    pB2 = drop_n2 / count_n2

    ''' Мат ожидание числа заявок на обслуживании'''
    ''' Мат ожидание числа заявок кол-ва занятого ресурса'''
    M1 = float(0)
    M2 = float(0)
    MR1 = float(0)
    MR2 = float(0)
    for n in range(N[0] + 1):
        M1 += float(n * freq1[n])
        MR1 += float(n * X[0] * freq1[n])
    for n in range(N[1] + 1):
        M2 += float(n * freq2[n])
        MR2 += float(n * X[1] * freq2[n])
    '''Коэф. использовния ресурса'''
    MR_S = MR2 + MR1
    k = MR_S / R
    return freq1, pB1, pB2, M1, M2, MR1, MR2, k


#i for i in range(1, 10, 1)
lmb_list = []
pb1_list = []
pb2_list = []
sim_pB1_list = []
sim_pB2_list = []
sim_M1_list = []
sim_M2_list = []
M1_list = []
M2_list = []
sim_MR1_list = []
sim_MR2_list = []
MR1_list = []
MR2_list = []
k_list = []
sim_k_list = []
lm = 190
for i in range (5):
    lmb_list.append(lm)
    lm += 2
def res():
    counter = 0
    for l in lmb_list:
        lmb = [10, l]
        mu = [0.033, 5.26]
        ro = [decimal.Decimal(lmb[0] / mu[0]), decimal.Decimal(lmb[1] / mu[1])]
        oX, pn1, pB1, pB2, M1, M2, MR1, MR2, k = main(ro)  # Теор
        freq, sim_pB1, sim_pB2, sim_M1, sim_M2, sim_MR1, sim_MR2, sim_k = Simulation(lmb, mu)
        pb1_list.append(pB1)
        pb2_list.append(pB2)
        sim_pB1_list.append(sim_pB1)
        sim_pB2_list.append(sim_pB2)
        M1_list.append(M1)
        M2_list.append(M2)
        sim_M1_list.append(sim_M1)
        sim_M2_list.append(sim_M2)
        MR1_list.append(MR1)
        MR2_list.append(MR2)
        sim_MR1_list.append(sim_MR1)
        sim_MR2_list.append(sim_MR2)
        k_list.append(k)
        sim_k_list.append(sim_k)
        counter += 1
        print(counter,'/',len(lmb_list))



    freq_sum = [0 for i in range(N[0] + 1)]
    freq_sum[0] = freq[0]

    for i in range(1, N[0] + 1):
        freq_sum[i] += freq[i - 1]
    max = 0
    for i in range(N[0] + 1):
        test = (math.fabs(pn1[i] - freq_sum[i]))
        if test > max:
            max = test

    plt.plot(lmb_list, pb1_list, label='Аналитика - 4G', c='b')
    plt.plot(lmb_list, pb2_list, label='Аналитика - 5G', linestyle='--', c='orange')
    plt.scatter(lmb_list, sim_pB1_list, label='Симуляция - 4G', marker='o', c='b')
    plt.scatter(lmb_list, sim_pB2_list, label='Симуляция - 5G', marker='^', c='orange')
    plt.xlabel('Интенсивность поступления заявок 5G, $\lambda_2$')
    plt.ylabel('Вероятность сброса')
    plt.legend()
    plt.grid(True, linestyle='--')

    plt.show()

    plt.plot(lmb_list, M1_list, label='Аналитика - 4G', c='b')
    plt.plot(lmb_list, M2_list, label='Аналитика - 5G', linestyle='--', c='orange')
    plt.scatter(lmb_list, sim_M1_list, label='Симуляция - 4G', marker='o', c='b')
    plt.scatter(lmb_list, sim_M2_list, label='Симуляция - 5G', marker='^', c='orange')
    plt.xlabel('Интенсивность поступления заявок 5G, $\lambda_2$')
    plt.ylabel('Среднее количество заявок на обслуживании')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()

    plt.plot(lmb_list, MR1_list, label='Аналитика - 4G', c='b')
    plt.plot(lmb_list, MR2_list, label='Аналитика - 5G', linestyle='--', c='orange')
    plt.scatter(lmb_list, sim_MR1_list, label='Симуляция - 4G', marker='o', c='b')
    plt.scatter(lmb_list, sim_MR2_list, label='Симуляция - 5G', marker='^', c='orange')
    plt.xlabel('Интенсивность поступления заявок 5G, $\lambda_2$')
    plt.ylabel('Среднее количесвто занятого ресурса')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()

    plt.plot(lmb_list, k_list, label='Аналитика', c='green')
    plt.xlabel('Интенсивность поступления заявок 5G, $\lambda_2$')
    plt.scatter(lmb_list, sim_k_list, label='Симуляция', marker='s', c='green')
    plt.ylabel('Коэф. использования ресурса')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.axis([190,198.2,0.9,1])
    plt.show()

def Kol():
    lmb = [198.315, 5]
    mu = [5.26, 0.033]
    ro = [decimal.Decimal(lmb[0] / mu[0]), decimal.Decimal(lmb[1] / mu[1])]
    oX, pn1, pB1, pB2, M1, M2, MR1, MR2, k = main(ro)  # Теор
    freq, sim_pB1, sim_pB2, sim_M1, sim_M2, sim_MR1, sim_MR2, sim_k = Simulation(lmb, mu)


    freq_sum = [0 for i in range(N[0] + 1)]

    freq_sum[0] = freq[0]
    for i in range(1, N[0] + 1):
        freq_sum[i] += freq[i - 1]
    max = 0
    for i in range(N[0] + 1):
        test = (math.fabs(pn1[i] - freq_sum[i]))
        if test > max:
            max = test
    max1 = 0
    for i in range(N[0] + 1):
        test = (math.fabs(pn1[i] - freq[i]))
        if test > max1:
            max1 = test

    print('Критерий Колмогорова', max)
    print('Критерий Колмогорова', max1)
    plt.plot(oX, pn1, label='Теоретич.')
    plt.plot(oX, freq, label='Имитацион.', linestyle='--', color='green')
    plt.xlabel('Количество заявок первого типа в системе ')
    plt.ylabel('Вероятность ')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.axis([ 15, 55, 0, 0.07])
    plt.show()

def TwoSimKol():
    lmb = [198.315, 5]
    mu = [5.26, 0.033]
    ro = [decimal.Decimal(lmb[0] / mu[0]), decimal.Decimal(lmb[1] / mu[1])]
    freq1, sim_pB1, sim_pB2, sim_M1, sim_M2, sim_MR1, sim_MR2, sim_k = Simulation(lmb, mu)
    freq2, sim_pB1, sim_pB2, sim_M1, sim_M2, sim_MR1, sim_MR2, sim_k = Simulation(lmb, mu)
    oX, pn1, pB1, pB2, M1, M2, MR1, MR2, k = main(ro)  # Теор

    freq_sum1 = [0 for i in range(N[0] + 1)]
    freq_sum2 = [0 for i in range(N[0] + 1)]

    freq_sum1[0] = freq1[0]
    freq_sum2[0] = freq2[0]
    for i in range(1, N[0] + 1):
        freq_sum1[i] += freq1[i - 1]
        freq_sum2[i] += freq2[i - 1]
    max = 0
    for i in range(N[0] + 1):
        test = (math.fabs(freq_sum2[i] - freq_sum1[i]))
        if test > max:
            max = test
    max1 = 0
    for i in range(N[0] + 1):
        test = (math.fabs(freq2[i] - freq1[i]))
        if test > max1:
            max1 = test
    print('Критерий Колмогорова', max)
    print ('Критерий Колмогорова', max1)
    plt.plot(oX, freq1, label='Имитация - 1.')
    plt.plot(oX, freq2, label='Имитация - 2', linestyle='--', color='green')
    plt.xlabel('Количество заявок первого типа в системе ')
    plt.ylabel('Вероятность ')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.axis([ 15, 55, 0, 0.07])
    plt.show()

res()
