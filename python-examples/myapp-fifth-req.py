# coding: utf-8

'''
from:	examples/tutorial/fifth.cc
to:	fifth.py
time:	20101110.1948.

//    
//         node 0                 node 1
//   +----------------+    +----------------+
//   |    ns-3 TCP    |    |    ns-3 TCP    |
//   +----------------+    +----------------+
//   |    10.1.1.1    |    |    10.1.1.2    |
//   +----------------+    +----------------+
//   | point-to-point |    | point-to-point |
//   +----------------+    +----------------+
//           |                     |
//           +---------------------+
//                5 Mbps, 2 ms
//
//
// We want to look at changes in the ns-3 TCP congestion window.  We need
// to crank up a flow and hook the CongestionWindow attribute on the socket
// of the sender.  Normally one would use an on-off application to generate a
// flow, but this has a couple of problems.  First, the socket of the on-off 
// application is not created until Application Start time, so we wouldn't be 
// able to hook the socket (now) at configuration time.  Second, even if we 
// could arrange a call after start time, the socket is not public so we 
// couldn't get at it.
//
// So, we can cook up a simple version of the on-off application that does what
// we want.  On the plus side we don't need all of the complexity of the on-off
// application.  On the minus side, we don't have a helper, so we have to get
// a little more involved in the details, but this is trivial.
//
// So first, we create a socket and do the trace connect on it; then we pass 
// this socket into the constructor of our simple application which we then 
// install in the source node.
'''

import sys
import ns.applications
import ns.core
import ns.internet
import ns.network
import ns.point_to_point
# import ns.gnuplot
import ns3 

import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels as sm
import scipy.stats as stats
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import os
import io
import subprocess
import statsmodels.distributions.empirical_distribution as edf
from scipy.interpolate import interp1d
from scipy.stats.distributions import chi2
import random
from matplotlib.ticker import FixedLocator, FixedFormatter

# Desligando avisos
import warnings
warnings.filterwarnings("ignore")

# Definindo variáveis globais
# Opções de geração de números aleatórios por "tcdf" ou "ecdf" ou "PD"(Probability Distribution)
mt_RG = ""
IC = ""

validation = False

# Definindo se o trace é ".txt" ou "xml"
reader = "txt"

traffic = ""
# time_request = False
# size_request = False
const_Size = ""
app_protocol = ""
size_xml = 0
stop_xml = 0

nRequests = 0

plot_graph = False
# plot = "show"
plot = "save"

# Define o parametro de rede utilizado nas funções
parameter = ""

# Armazena em np.arrays() os dados dos traces
t_time = np.empty(1)
t_size = np.empty(1)
req_t_time = np.empty(1)
req_t_size = np.empty(1)
resp_t_time = np.empty(1)
resp_t_size = np.empty(1)

# Variáveis que armazenam os parametros das distribuições de probabilidade
# time
dist_time = ""
arg_time = []
loc_time = 0
scale_time = 0
# # size
dist_size = ""
arg_size = []
loc_size = 0
scale_size = 0

# time request
req_dist_time = ""
req_arg_time = []
req_loc_time = 0
req_scale_time = 0
# size request
req_dist_size = ""
req_arg_size = []
req_loc_size = 0
req_scale_size = 0

# time response
resp_dist_time = ""
resp_arg_time = []
resp_loc_time = 0
resp_scale_time = 0
# size response
resp_dist_size = ""
resp_arg_size = []
resp_loc_size = 0
resp_scale_size = 0

# Variável de auxilio de parada da função tcdf
first_tcdf_time = 0
first_tcdf_size = 0
first_req_tcdf_time = 0
first_req_tcdf_size = 0
first_resp_tcdf_time = 0
first_resp_tcdf_size = 0

# Variável de auxilio de parada da função read_trace
first_trace_size = 0
first_trace_time = 0
first_req_trace_size = 0
first_req_trace_time = 0
first_resp_trace_size = 0
first_resp_trace_time = 0
aux_global_time = 0
resp_aux_global_time = 0
req_aux_global_time = 0

printo = []
# Função de leitura dos arquivos xml
def read_xml(parameter):
    global size_xml
    global stop_xml
    ifile = open('scratch/results-http-docker.pdml','r')

    print(ifile)

    columns = ["length", "time"]
    df = pd.DataFrame(columns = columns)

    data0 = []
    data1 = []
   
    for line in ifile.readlines(): 
        if ("httpSample" in line and "</httpSample>" not in line):
            data0.append(line)
        if ("httpSample" in line and "</httpSample>" not in line):
            data1.append(line)

    ifile.close()
    # Save parameters in DataFrames and Export to .txt
    df = pd.DataFrame(list(zip(data0, data1)), columns=['length', 'time'])


    df['length'] = df['length'].str.split('by="').str[-1]
    df['time'] = df['time'].str.split('ts="').str[-1]

    df['length'] = df['length'].str.split('"').str[0]
    df['time'] = df['time'].str.split('"').str[0]

    df["length"] = pd.to_numeric(df["length"],errors='coerce')
    df["time"] = pd.to_numeric(df["time"],errors='coerce')
    
    print("DF: ", df)



    size_xml = len(df["time"])

    stop_xml = df["time"]

    print("STOP: ", len(stop_xml))
    stop_xml = stop_xml[len(stop_xml)-1]

    if parameter == "Size":
        # Chamando variáveis globais
        global t_size
        global first_trace_size

        # Abrindo arquivos .txt
        t_size = np.array(df['length'])
        # print("Trace Size: ", t_size)

        # Plot histograma de t_size:
        plt.hist(t_size)
        plt.title("Histogram of trace ("+parameter+")")
        plt.show()

        # Com ajuda da lib Pandas podemos encontrar algumas estatísticas importantes.
        # y_size_df = pd.DataFrame(y_size, columns=['Size'])
        # y_size_df.describe()

        # Definindo que o parametro size pode ser lido apenas uma vez.
        first_trace_size = 1
    
    
    if parameter == "Time":
        # Chamando variáveis globais
        global t_time
        global first_trace_time

        # Abrindo arquivos .txt
        t_time = np.array(df['time'])

        # Obtendo os tempos entre pacotes do trace
        sub = []
        
        for i in range(0,len(t_time)-1):
            sub.append(t_time[i+1] - t_time[i])
        
        # Passando valores resultantes para a variável padrão t_time
        t_time = np.array(sub)
        # print("Trace Time: ", t_time)

        # Plot histograma t_time:
        plt.hist(t_time)
        plt.title("Histogram of trace ("+parameter+")")
        plt.show()

        # Com ajuda da lib Pandas pode-se encontrar algumas estatísticas importantes.
        # t_time_df = pd.DataFrame(t_time, columns=['Time'])
        # t_time_df.describe()

        # Definindo que o parametro time pode ser lido apenas uma vez.
        first_trace_time = 1    

# Função de leitura dos traces e atribuição dos respectivos dados aos vetores
def read_txt(parameter, traffic, app_protocol): 
    global plot_graph
    global t_net
    if parameter == "Time" and traffic == "send":
        # Chamando variáveis globais
        global t_time
        global first_trace_time
     
        # Abrindo arquivos .txt
        t_time = np.loadtxt("scratch/"+app_protocol+"_time.txt", usecols=0)
        t_time.sort()
        # Obtendo os tempos entre pacotes do trace
        sub = []
        
        for i in range(0,len(t_time)-1):
            sub.append(t_time[i+1] - t_time[i])
        
        # Passando valores resultantes para a variável padrão t_time
        # sub.remove("0\n")
        # sub.sort()
        
        t_time = np.array(sub)
        t_time = t_time.astype(float)
        
        # aux = t_time.where(0)
        # t_time.remove(aux)

        t_time = np.delete(t_time, np.where(t_time == 0))
        t_time.sort()

        # np.delete(t_time, 0)
        # t_time = sorted(t_time, reverse=True)
        # print("Trace Time: ", t_time[0:20])
        
        # t_time = np.around(t_time, 2)
        # print("t_time:", len(t_time))


        # Plot histograma t_time:
        # plt.hist(t_time)
        if plot_graph == True:
            fig, ax = plt.subplots(1, 1)
            ax = sns.distplot(t_time)
            plt.title("Histogram of trace "+traffic+" ("+parameter+")")
            if plot == "show":
                plt.show()
            if plot == "save":
                fig.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_hist_"+traffic+"_"+parameter, fmt="png",dpi=1000)
                plt.close()
        

        # Com ajuda da lib Pandas pode-se encontrar algumas estatísticas importantes.
        # t_time_df = pd.DataFrame(t_time, columns=['Time'])
        # t_time_df.describe()

        # Definindo que o parametro time pode ser lido apenas uma vez.
        first_trace_time = 1     

    if parameter == "Size" and traffic == "send":
        # Chamando variáveis globais
        global t_size
        global first_trace_size

        # Abrindo arquivos .txt
        t_size = np.loadtxt("scratch/"+app_protocol+"_size.txt", usecols=0)
        t_size = t_size.astype(float)
        t_size = np.array(t_size)
        
        # Plot histograma t_size:
        # plt.hist(t_size)
        if plot_graph == True:
            fig, ax = plt.subplots(1, 1)
            ax = sns.distplot(t_size)
            plt.title("Histogram of trace "+traffic+" ("+parameter+")")
            if plot == "show":
                plt.show()
            if plot == "save":
                fig.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_hist_"+traffic+"_"+parameter, fmt="png",dpi=1000)
                plt.close()
        # Definindo que o parametro time pode ser lido apenas uma vez.
        first_trace_size = 1 
        
    if parameter == "Time" and traffic == "request":
        # Chamando variáveis globais
        global req_t_time
        global first_req_trace_time

        # Abrindo arquivos .txt
        req_t_time = np.loadtxt("scratch/"+app_protocol+"_req_time.txt", usecols=0)
        req_t_time.sort()
        # Obtendo os tempos entre pacotes do trace
        sub = []
        
        for i in range(0, len(req_t_time)-1):
            sub.append(req_t_time[i+1] - req_t_time[i])
        
        # Passando valores resultantes para a variável padrão req_t_time
        req_t_time = np.array(sub)
        req_t_time = req_t_time.astype(float)
        
        req_t_time = np.delete(req_t_time, np.where(req_t_time == 0))
        req_t_time.sort()
        # print("Trace Time Request: ", req_t_time)
        # print("req_t_time:", req_t_time)
        # Plot histograma req_t_time:
        # plt.hist(req_t_time)
        if plot_graph == True:
            fig, ax = plt.subplots(1, 1)
            ax = sns.distplot(req_t_time)
            plt.title("Histogram of trace "+traffic+" ("+parameter+")")
            if plot == "show":
                plt.show()
            if plot == "save":
                fig.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_hist_"+traffic+"_"+parameter, fmt="png",dpi=1000)
                plt.close()
        # Com ajuda da lib Pandas pode-se encontrar algumas estatísticas importantes.
        # req_t_time_df = pd.DataFrame(req_t_time, columns=['Time'])
        # req_t_time_df.describe()

        # Definindo que o parametro time pode ser lido apenas uma vez.
        first_req_trace_time = 1     

    if parameter == "Size" and traffic == "request":
        # Chamando variáveis globais
        global req_t_size
        global first_req_trace_size

        # Abrindo arquivos .txt
        req_t_size = np.loadtxt("scratch/"+app_protocol+"_req_size.txt", usecols=0)
        req_t_size = req_t_size.astype(float)
        req_t_size = np.array(req_t_size)
        if np.mean(req_t_size) == req_t_size[0]: 
            req_t_size[0] = req_t_size[0]-1
        # Plot histograma req_t_size:
        # plt.hist(req_t_size)
        if plot_graph == True:
            fig, ax = plt.subplots(1, 1)
            ax = sns.distplot(req_t_size)
            plt.title("Histogram of trace "+traffic+" ("+parameter+")")
            if plot == "show":
                plt.show()
            if plot == "save":
                fig.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_hist_"+traffic+"_"+parameter, fmt="png",dpi=1000)
                plt.close()
        

        # Com ajuda da lib Pandas pode-se encontrar algumas estatísticas importantes.
        # req_t_size_df = pd.DataFrame(req_t_size, columns=['size'])
        # req_t_size_df.describe()

        # Definindo que o parametro size pode ser lido apenas uma vez.
        first_req_trace_size = 1     

    if parameter == "Time" and traffic == "response":
        # Chamando variáveis globais
        global resp_t_time
        global first_resp_trace_time

        # Abrindo arquivos .txt
        resp_t_time = np.loadtxt("scratch/"+app_protocol+"_resp_time.txt", usecols=0)
        resp_t_time.sort()
        # Obtendo os tempos entre pacotes do trace
        sub = []
        
        for i in range(0, len(resp_t_time)-1):
            sub.append(resp_t_time[i+1] - resp_t_time[i])
        
        # Passando valores resultantes para a variável padrão resp_t_time
        resp_t_time = np.array(sub)
        resp_t_time = resp_t_time.astype(float)
        
        resp_t_time = np.delete(resp_t_time, np.where(resp_t_time == 0))
        resp_t_time.sort()
        # print("Trace Time Response: ", resp_t_time)
        # print("req_t_time:", req_t_time)
        # Plot histograma resp_t_time:
        # plt.hist(resp_t_time)
        if plot_graph == True:
            fig, ax = plt.subplots(1, 1)
            ax = sns.distplot(resp_t_time)
            plt.title("Histogram of trace "+traffic+" ("+parameter+")")
            if plot == "show":
                plt.show()
            if plot == "save":
                fig.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_hist_"+traffic+"_"+parameter, fmt="png",dpi=1000)
                plt.close()

        # Com ajuda da lib Pandas pode-se encontrar algumas estatísticas importantes.
        # resp_t_time_df = pd.DataFrame(resp_t_time, columns=['Time'])
        # resp_t_time_df.describe()

        # Definindo que o parametro time pode ser lido apenas uma vez.
        first_resp_trace_time = 1     
  
    if parameter == "Size" and traffic == "response":
        # Chamando variáveis globais
        global resp_t_size
        global first_resp_trace_size

        # Abrindo arquivos .txt
        resp_t_size = np.loadtxt("scratch/"+app_protocol+"_resp_size.txt", usecols=0)
        resp_t_size = resp_t_size.astype(float)
        resp_t_size = np.array(resp_t_size)

        # if np.mean(resp_t_size) == resp_t_size[1]: 
        #     resp_t_size[0] = resp_t_size[0]-1
        # Plot histograma resp_t_size:
        # plt.hist(resp_t_size)
        if plot_graph == True:
            fig, ax = plt.subplots(1, 1)
            ax = sns.distplot(resp_t_size)
            plt.title("Histogram of trace "+traffic+" ("+parameter+")")
            if plot == "show":
                plt.show()
            if plot == "save":
                fig.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_hist_"+traffic+"_"+parameter, fmt="png",dpi=1000)
                plt.close()

        # Com ajuda da lib Pandas pode-se encontrar algumas estatísticas importantes.
        # resp_t_size_df = pd.DataFrame(resp_t_size, columns=['size'])
        # resp_t_size_df.describe()

        # Definindo que o parametro size pode ser lido apenas uma vez.
        first_resp_trace_size = 1     


cont = 0
first_req = True 
first_resp = True 
first_send = True 
y_req = 0
y_resp = 0
y_send = 0
# Função de geração de variáveis aleatórias por meio da ECDF
def ecdf(y, parameter, traffic):
    global cont
    global first_send
    global first_req
    global first_resp
    global y_req 
    global y_resp
    global y_send

    # Organizando o vetor com os dados do trace
    y.sort()
    
    if traffic == 'request':
        if first_req == True:
            # y = np.around(y, 4)
            # print("Original REQ: ", y)
            y_req = list(dict.fromkeys(y))
            y = y_req
            first_req = False
            # print("Modify REQ: ", y_req)
        else:
            y = y_req
            # print("Modify REQ: ",len(y_req))

    if traffic == 'response':
        if first_resp == True:
            # y = np.around(y, 4)
            # print("Original RESP: ", y)
            y_resp = list(dict.fromkeys(y))
            y = y_resp
            # print("Modify RESP: ",y_req)
            first_resp = False
        else:
            y = y_resp
            # print("Modify RESP: ",len(y_req))
    
    if traffic == 'send':
        if first_send == True:
            # y = np.around(y, 2)
            # print("Original SEND: ", y)
            y_send = list(dict.fromkeys(y))
            y = y_send
            # print("Modify SEND: ", y_req)
            first_send = False
        else:
            y = y_send
            # print("Modify SEND: ",len(y_req))

    # Criando listas para os dados utilizados
    Fx = []
    Fx_ = []
    # Realizando ajustes para os vetores que selecionaram os valores gerados
    for i in range(0, len(y)):
        Fx.append(i/(len(y)+1))
        if i != 0:
            Fx_.append(i/(len(y)+1))
    # Adicionando 1 no vetor Fx_
    Fx_.append(1)    

    # print ("Fx: ", len(Fx))
    # print ("Fx_: ", len(Fx_))
    
    
    # Gerando um valor aleatório entre 0 e 1 uniforme
    rand = np.random.uniform(0,1)
    # print("Rand: ", rand)
    
    # Pecorrer todos os valores do vetor com dados do trace
    # para determinar o valor a ser gerado de acordo com o resultado da distribuição uniforme
    r_N = 0
    for i in range(0, len(y)):
        # Condição que define em qual classe o valor é encontrado
        if rand > Fx[i] and rand < Fx_[i]:
            # Determinando o valor resultante 
            r_N = y[i-1]
            
    
    # print ("Fx: ", Fx)
    # print ("Fx_: ", Fx_)
    # print(r_N)
    # print ("Y: ", len(y))
    
    if len(y) == 1245:
        cont+=1
        print(cont)
    # Condição para retorno do valor de acordo com o parametro de rede.
    if parameter == "Size":
        # print ("ECDF SIZE: ", r_N)
        return(int(abs(r_N)))

    if parameter == "Time":
        # print ("ECDF TIME: ", r_N)
        r_N = np.around(r_N, 2)
        w = open("../../../Results/Prints/ecdf_"+traffic+"_Time.txt", "a")

        w.write("\n" + str(r_N) + "\n")
        w.close()
        # np.savetxt('scratch/'+'ecdf_'+app_protocol+'_time_req_ns3.txt', time_req_ns3, delimiter=',', fmt='%f')
        return(abs(r_N))

def ksvalid(size, Dobs):
    # Definir intervalo de confiança
    # IC = 99.90 -> alpha = 0.10
    # IC = 99.95 -> alpha = 0.05
    # IC = 99.975 -> alpha = 0.025
    # IC = 99.99 -> alpha = 0.01
    # IC = 99.995 -> alpha = 0.005
    # IC = 99.999 -> alpha = 0.001
    global IC
    
    D_critico = 0
    rejects = ""
    IC = str(IC)+"%"
    # print("IC: ", IC)
    if (size<=35):
        ks_df = pd.read_csv("../../../FWGNet/kstest.txt", sep=";")
        # ks_df = ks_df[ks_df['Size'].str.contains(""+size+"")]
        # print(ks_df)
        D_critico = ks_df[""+IC+""].iloc[size-1]
        
    else:
        # Condição para definir o D_critico de acordo com o tamanho dos dados
        if IC == "99.80%":
            D_critico = 1.07/np.sqrt(size)
        if IC == "99.85%":
            D_critico = 1.14/np.sqrt(size)
        if IC == "90.0%":
            D_critico = 1.224/np.sqrt(size)
        if IC == "95.0%":
            D_critico = 1.358/np.sqrt(size)
        if IC == "99.0%":
            D_critico = 1.628/np.sqrt(size)
    
    D_critico = np.around(D_critico, 2)
    # 0.20"	;	       "0.15"	;	 "0.10"	;	 "0.05"	;	     "0.01"
    # "1.07/sqrt(n)";"1.14/sqrt(n)";"1.22/sqrt(n)";"1.36/sqrt(n)";"1.63/sqrt(n)"

    # Condição para aceitar a hipótese nula do teste KS
    if Dobs > D_critico:
        rejects = "Reject the Null Hypothesis"
    else:
        rejects = "Fails to Reject the Null Hypothesis"
    
    # IC = IC[:-1]
    IC = IC.replace('%', '')
    # print("IC: ", IC)
    return rejects, float(IC), D_critico
    
# Função para definir a distribuição de probabilidade compatível com os 
# valores do trace utilizada para gerar valores aleatórios por TCDF
def tcdf(y, parameter, traffic):
    global t_net
    # Indexar o vetor y pelo vetor x
    x = np.arange(len(y))
    # Definindo o tamanho da massa de dados
    size = len(y)
    # Definindo a quantidade de bins (classes) dos dados
    nbins = int(np.sqrt(size))
    # if np.mean(y) == y[0]:
    #     nbins = 3
    
    # Normalização dos dados
    sc=StandardScaler()
    yy = y.reshape (-1,1)
    sc.fit(yy)
    y_std = sc.transform(yy)
    y_std = y_std.flatten()

    del yy
    # O python pode relatar avisos enquanto executa as distribuições

    # Mais distribuições podem ser encontradas no site da lib "scipy"
    # Veja https://docs.scipy.org/doc/scipy/reference/stats.html para mais detalhes
    dist_names = ['erlang',
                'expon',
                'gamma',
                'lognorm',
                'norm',
                'pareto',
                'triang',
                'uniform',
                'dweibull',
                'weibull_min',
                'weibull_max']
    # Obter os métodos de inferência KS test e Chi-squared
    # Configurar listas vazias para receber os resultados
    chi_square = []
    ks_values = []
    #--------------------------------------------------------#
    
    # Chi-square

    # Configurar os intervalos de classe (nbins) para o teste qui-quadrado
    # Os dados observados serão distribuídos uniformemente em todos os inervalos de classes
    percentile_bins = np.linspace(0,99,nbins)
    percentile_cutoffs = np.percentile(y, percentile_bins)
    observed_frequency, bins = (np.histogram(y, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Repetir para as distribuições candidatas
    for distribution in dist_names:
        # Configurando a distribuição e obtendo os parâmetros ajustados da distribuição
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(x)
        
        #
        # KS TEST
        #
        # Criando percentil
        percentile = np.linspace(0,99,len(y))
        percentile_cut = np.percentile(y, percentile) 
        
        # Criando CDF da teórica
        Ft = dist.cdf(percentile_cut, *param[:-2], loc=param[-2], scale=param[-1])

        # Criando CDF Inversa 
        Ft_ = dist.ppf(percentile_cut, *param[:-2], loc=param[-2], scale=param[-1])

        # Adicionando dados do trace
        t_Fe = y


        # Criando listas para armazenar as ECDFs
        Fe = []
        Fe_ = []

        # Criando ECDFs
        for i in range(1, len(y)+1):
            # ecdf i-1/n
            Fe.append((i-1)/len(y))
            # ecdf i/n
            Fe_.append(i/len(y))

        # Transformando listas em np.arrays()
        Fe = np.array(Fe)
        Fe_ = np.array(Fe_)
        Ft = np.array(Ft)
        Ft_ = np.array(Ft_)


        # Ordenando dados
        t_Fe.sort()
        Ft.sort()
        Ft_.sort()
        Fe.sort()
        Fe_.sort()
        
        # Inicio cálculo de rejeição
        #
        # Ft(t)-FE-(i),FE+(i)-Ft(t)
        Ft_Fe_ = np.subtract(Ft, Fe_)
        Fe_Ft = np.subtract(Fe, Ft)
        
        # Max(Ft(t)-FE-(i),FE+(i)-Ft(t))
        Dobs_max = np.maximum(Ft_Fe_, Fe_Ft)
        
        # Dobs= Max(Max (Ft(t)-FE-(i),FE+(i)-Ft(t)))
        Dobs = np.max(Dobs_max)
        #
        # Fim cálculo de rejeição
        
        ks_statistic, p_value = stats.ks_2samp(Ft,t_Fe, mode='exact', alternative='less')
            
        rejects, IC, D_critico = ksvalid(len(t_Fe), ks_statistic)
        # rejects, IC, D_critico = ksvalid(size, Dobs)

        
        # Imprimindo resultados do KS Test
        print(" ")
        print("KS TEST:")
        print("Confidence degree: ", IC,"%")
        print("D observed: ", Dobs)
        print("D observed(two samples): ", ks_statistic)
        print("D critical: ", D_critico)
        print(rejects, " to  Real Trace (Manual-ks_statistic/D_critico)")

        a = 1-(IC/100)
        
        a = np.around(a,4)

        if p_value < a:
            print("Reject - p-value: ", p_value, " is less wich alpha: ", a," (2samp)")
        else:
            print("Fails to Reject - p-value: ", p_value, " is greater wich alpha: ", a," (2samp)")
        
        print(" ")
  
        # Obtém a estatística do teste KS e arredonda para 5 casas decimais
        Dobs = np.around(Dobs,  5)
        # ks_values.append(Dobs)
        ks_values.append(ks_statistic)    

        #
        # CHI-SQUARE
        #

        # Obter contagens esperadas nos percentis
        # Isso se baseia em uma 'função de distribuição acumulada' (cdf)
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], scale=param[-1])
        
        # Definindo a frequência esperada
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # Calculando o qui-quadrado
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square.append(ss)

        # Set x² with IC
        IC = IC/100
        x2 = chi2.ppf(IC, nbins-1)
        
        # Imprimindo resultados do teste Chi-square
        print(" ")
        print("Chi-square test: ")
        print("Confidence degree: ", IC,"%")
        print("CS: ", ss)
        print("X²: ", x2)
        # Condição para aceitar a hipótese nula do teste Chi-square
        if x2 > ss:
            print("Fails to Reject the Null Hipothesis of ", distribution)
        else:
            print("Rejects the Null Hipothesis of ", distribution)
        print(" ")

    # Agrupar os resultados e classificar por qualidade de ajuste de acordo com o teste KS (melhor na parte superior)
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['ks_value'] = ks_values
    results['chi_square'] = chi_square
    results.sort_values(['ks_value'], inplace=True, ascending=True)

    # Apresentar os resultados em uma tabela
    print ('\nDistributions sorted by KS Test of ',traffic,'(',parameter,'):')
    print ('----------------------------------------')
    print (results)
    print (traffic," ",parameter," ",y[0:3])
    # Divida os dados observados em N posições para plotagem (isso pode ser alterado)
    bin_cutoffs = np.linspace(np.percentile(y,0), np.percentile(y,99), nbins)

    # Crie o gráfico
    # if plot_graph == True:
    h = plt.hist(y, bins = bin_cutoffs, color='0.75')
    
    if plot_graph == False:
        plt.clf()
        plt.close()
    # Receba as principais distribuições da fase anterior 
    # e seleciona a quantidade de distribuições.
    number_distributions_to_plot = 1
    dist_names = results['Distribution'].iloc[0:number_distributions_to_plot]

    # Crie uma lista vazia para armazenar parâmetros de distribuição ajustada
    parameters = []
    
    # Faça um loop pelas distribuições para obter o ajuste e os parâmetros da linha
    for dist_name in dist_names:
        # Chamando variáveis globais
        global arg_time
        global loc_time
        global scale_time
        global dist_time

        global arg_size
        global loc_size
        global scale_size
        global dist_size

        global req_arg_time
        global req_loc_time
        global req_scale_time
        global req_dist_time

        global req_arg_size
        global req_loc_size
        global req_scale_size
        global req_dist_size

        global resp_arg_time
        global resp_loc_time
        global resp_scale_time
        global resp_dist_time

        global resp_arg_size
        global resp_loc_size
        global resp_scale_size
        global resp_dist_size
        
        # Obtendo distribuições e seus parametros de acordo com o trace
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(y)
        parameters.append(param)

        arg = param[:-2]
        loc = param[-2]
        scale = param[-1]
        print(parameters)

        if parameter == "Time" and traffic == "send":
            dist_time = dist_name
            loc_time = loc
            scale_time = scale
            arg_time = arg

        if parameter == "Size" and traffic == "send":
            dist_size = dist_name
            loc_size = loc
            scale_size = scale
            arg_size = arg

        if parameter == "Time" and traffic == "request":
            req_dist_time = dist_name
            req_loc_time = loc
            req_scale_time = scale
            req_arg_time = arg

        if parameter == "Size" and traffic == "request":
            req_dist_size = dist_name
            req_loc_size = loc
            req_scale_size = scale
            req_arg_size = arg

        if parameter == "Time" and traffic == "response":
            resp_dist_time = dist_name
            resp_loc_time = loc
            resp_scale_time = scale
            resp_arg_time = arg

        if parameter == "Size" and traffic == "response":
            resp_dist_size = dist_name
            resp_loc_size = loc
            resp_scale_size = scale
            resp_arg_size = arg


        # Obter linha para cada distribuição (e dimensionar para corresponder aos dados observados)
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
        scale_pdf = np.trapz (h[0], h[1][:-1]) / np.trapz (pdf_fitted, x)
        pdf_fitted *= scale_pdf
        if plot_graph == True:
            # Adicione a linha ao gráfico
            plt.plot(pdf_fitted, label=dist_name)

            # Defina o eixo gráfico x para conter 99% dos dados
            # Isso pode ser removido, mas, às vezes, dados fora de padrão tornam o gráfico menos claro
            plt.xlim(0,np.percentile(y,99))
            plt.title("Histogram of trace "+traffic+"(" + parameter + ") + theorical distribuition " + dist_name)
    # Adicionar legenda
    plt.legend()
    if plot_graph == True:
        if plot == "show":
            plt.show()
        if plot == "save":
            plt.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_hist_tcdf_"+traffic+"_"+dist_name+"_"+parameter, fmt="png",dpi=1000)
            plt.close()
    # Armazenar parâmetros de distribuição em um quadro de dados (isso também pode ser salvo)
    dist_parameters = pd.DataFrame()
    dist_parameters['Distribution'] = (
            results['Distribution'].iloc[0:number_distributions_to_plot])
    dist_parameters['Distribution parameters'] = parameters

    # Printar os parâmetros
    print ('\nDistribution parameters:')
    print ('------------------------')

    for row in dist_parameters.iterrows():
        print ('\nDistribution:', row[0])
        print ('Parameters:', row[1] )

    
    # Plotando gráficos de inferência
    data = y_std.copy()
    # data = y

    data.sort()
    # Loop through selected distributions (as previously selected)
    
    for distribution in dist_names:
        # Set up distribution
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y)

        #
        # KS TEST
        #
        # Criando percentil
        percentile = np.linspace(0,100,len(y))
        percentile_cut = np.percentile(y, percentile)
        
        # Criando CDF da teórica
        Ft = dist.cdf(percentile_cut, *param[:-2], loc=param[-2], scale=param[-1])
        
        
        # Criando CDF Inversa 
        Ft_ = dist.ppf(percentile_cut, *param[:-2], loc=param[-2], scale=param[-1])
        
        # Adicionando dados do trace
        t_Fe = y

        # Ordenando dados
        t_Fe.sort()
        Ft.sort()
        Ft_.sort()

        # Criando listas para armazenar as ECDFs
        Fe = []
        Fe_ = []

        # Criando ECDFs
        for i in range(1, len(y)+1):
            # ecdf i-1/n
            Fe.append((i-1)/len(y))
            # ecdf i/n
            Fe_.append(i/len(y))

        # Transformando listas em np.arrays()
        Fe = np.array(Fe)
        Fe_ = np.array(Fe_)
        Ft = np.array(Ft)
        Ft_ = np.array(Ft_)
        
        # Inicio cálculo de rejeição
        #
        # Ft(t)-FE-(i),FE+(i)-Ft(t)
        Ft_Fe_ = np.subtract(Ft, Fe_)
        Fe_Ft = np.subtract(Fe, Ft)
        
        # Max(Ft(t)-FE-(i),FE+(i)-Ft(t))
        Dobs_max = np.maximum(Ft_Fe_, Fe_Ft)
        
        # Dobs= Max(Max (Ft(t)-FE-(i),FE+(i)-Ft(t)))
        Dobs = np.max(Dobs_max)
        #
        # Fim cálculo de rejeição

        ks_statistic, p_value = stats.ks_2samp(Ft,t_Fe, mode='exact', alternative='less')
      
        rejects, IC, D_critico = ksvalid(len(t_Fe), ks_statistic)
        # rejects, IC, D_critico = ksvalid(size, Dobs)

        
        # Imprimindo resultados do KS Test
        print(" ")
        print("KS TEST:")
        print("Confidence degree: ", IC,"%")
        print("D observed: ", Dobs)
        print("D observed(two samples): ", ks_statistic)
        print("D critical: ", D_critico)
        print(rejects, " to  Real Trace (Manual-ks_statistic/D_critico)")

        a = 1-(IC/100)
        
        a = np.around(a,4)

        if p_value < a:
            print("Reject - p-value: ", p_value, " is less wich alpha: ", a," (2samp)")
        else:
            print("Fails to Reject - p-value: ", p_value, " is greater wich alpha: ", a," (2samp)")
        
        print(" ")

        # Plotando resultados do teste KS
        if plot_graph == True:
            plt.plot(t_Fe, Ft, 'o', label='Teorical Distribution')
            plt.plot(t_Fe, Fe, 'o', label='Empirical Distribution')
            
            
            # plt.plot(t_Fe, Fe, 'o', label='Real Trace')
            # plt.plot(Ft, Fe, 'o', label='Syntatic Trace')
            # Definindo titulo
            plt.title("KS Test of Real Trace of "+traffic+" with " + distribution + " Distribution (" + parameter + ")")
            plt.legend()
            if plot == "show":
                plt.show()
            if plot == "save":
                plt.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_kstest_tcdf_"+traffic+"_"+distribution+"_"+parameter, fmt="png",dpi=1000)
                plt.close()
        global first_tcdf_time
        global first_tcdf_size

        if parameter == "Size" and traffic == "send":
            first_tcdf_size = 1
        if parameter == "Time" and traffic == "send":
            first_tcdf_time = 1

        global first_req_tcdf_time
        global first_req_tcdf_size
        global first_resp_tcdf_time
        global first_resp_tcdf_size

        if parameter == "Size" and traffic == "request":
            first_req_tcdf_size = 1
        if parameter == "Time" and traffic == "request":
            first_req_tcdf_time = 1

        if parameter == "Size" and traffic == "response":
            first_resp_tcdf_size = 1
        if parameter == "Time" and traffic == "response":
            first_resp_tcdf_time = 1

# Função de geração de variáveis aleatórias por meio da TCDF

def tcdf_generate(dist, loc, scale, arg, parameter):
    # Setar distribuição escolhida.
    dist_name = getattr(scipy.stats, dist)

    # Gerar número aleatório de acordo com a distribuição escolhida e seus parametros.
    r_N = dist_name.rvs(loc=loc, scale=scale, *arg)

    # Condição para retorno do valor de acordo com o parametro de rede.
    if parameter == "Size":
        # print("SIZE R_N:", r_N)
        return(int(abs(r_N)))

    if parameter == "Time":
        # print("TIME R_N:", r_N)
        return(float(abs(r_N)))

# Função de geração de variáveis aleatórias de acordo com distribuições 
# de probabilidade e parametros definidos
def wgwnet_PD(parameter, traffic):
    # Mais distribuições podem ser encontradas no site da lib "scipy"
    # Veja https://docs.scipy.org/doc/scipy/reference/stats.html para mais detalhes
    # global request
    # global size_request
    # global time_request
    if traffic == "request" and parameter == "Size":
        if const_Size == "False":
        # Selecionando distribuição de probabilidade para o parametro Size
            dist_name = 'uniform'
            # Definindo parametros da distribuição
            loc = 1024
            scale = 2048
            arg = []
            # Setando distribuição a escolhida e seus parametros 
            dist = getattr(scipy.stats, dist_name)

            # Gerando número aleatório de acordo com a distribuiução e os parametros definidos
            r_N = dist.rvs(loc=loc, scale=scale, *arg, size=1)
        else:
            r_N = 50
        # size_request = True
        return(int(r_N))

    if traffic == "request" and parameter == "Time":
       # Selecionando distribuição de probabilidade para o parametro Size
        dist_name = 'uniform'
        # Definindo parametros da distribuição
        loc = 0.5
        scale = 0.8
        arg = []
        # Setando distribuição a escolhida e seus parametros 
        dist = getattr(scipy.stats, dist_name)
        # Gerando número aleatório de acordo com a distribuiução e os parametros definidos
        r_N = dist.rvs(loc=loc, scale=scale, *arg, size=1)
        # time_request = True
        return(float(r_N))

    if traffic == "response" and parameter == "Size":
        if const_Size == "False":
        # Selecionando distribuição de probabilidade para o parametro Size
            dist_name = 'uniform'
            # Definindo parametros da distribuição
            loc = 1024
            scale = 2048
            arg = []
            # Setando distribuição a escolhida e seus parametros 
            dist = getattr(scipy.stats, dist_name)

            # size_request = False
            # Gerando número aleatório de acordo com a distribuiução e os parametros definidos
            r_N = dist.rvs(loc=loc, scale=scale, *arg, size=1)
        else:
            r_N = 100
        return(int(r_N))
            
    if traffic == "response" and parameter == "Time":
        # Selecionando distribuição de probabilidade para o parametro Size
        dist_name = 'uniform'
        # Definindo parametros da distribuição
        loc = 0.5
        scale = 0.8
        arg = []
        # Setando distribuição a escolhida e seus parametros 
        dist = getattr(scipy.stats, dist_name)

        # time_request = False
        # Gerando número aleatório de acordo com a distribuiução e os parametros definidos
        r_N = dist.rvs(loc=loc, scale=scale, *arg, size=1)
        return(float(r_N))


    if traffic == "send" and parameter == "Size":
        if const_Size == "False":
            # Selecionando distribuição de probabilidade para o parametro Size
            dist_name = 'uniform'
            # Definindo parametros da distribuição
            loc = 1024
            scale = 2048
            arg = []
            # Setando distribuição a escolhida e seus parametros 
            dist = getattr(scipy.stats, dist_name)

            # size_request = False
            # Gerando número aleatório de acordo com a distribuiução e os parametros definidos
            r_N = dist.rvs(loc=loc, scale=scale, *arg, size=1)
        else:
            r_N = 1400
        return(int(r_N))

    if traffic == "send" and parameter == "Time":
        # Selecionando distribuição de probabilidade para o parametro Size
        dist_name = 'uniform'
        # Definindo parametros da distribuição
        loc = 0.5
        scale = 0.8
        arg = []
        # Setando distribuição a escolhida e seus parametros 
        dist = getattr(scipy.stats, dist_name)

        # Gerando número aleatório de acordo com a distribuiução e os parametros definidos
        r_N = dist.rvs(loc=loc, scale=scale, *arg, size=1)
        return(float(r_N))
    
def constSize(traffic, app_protocol):
    size = 0
    global mt_const 
    global t_size
    global req_t_size
    global resp_t_size
    
    if app_protocol == "ftp":
        if traffic == "request":
            if mt_const == "Trace":
                size = np.mean(req_t_size)
            if mt_const == "Value":
                size = 500
            return int(size)
        if traffic == "response":
            if mt_const == "Trace":
                size = np.mean(resp_t_size)
            if mt_const == "Value":
                size = 500
            return int(size)
        if traffic == "send":
            if mt_const == "Trace":
                size = np.mean(t_size)
            if mt_const == "Value":
                size = 500
            return int(size)

    if app_protocol == "hls":
        if traffic == "request":
            if mt_const == "Trace":
                size = np.mean(req_t_size)
            if mt_const == "Value":
                size = 500
            return int(size)
        if traffic == "response":
            if mt_const == "Trace":
                size = np.mean(resp_t_size)
            if mt_const == "Value":
                size = 500
            return int(size)
        if traffic == "send":
            if mt_const == "Trace":
                size = np.mean(t_size)
            if mt_const == "Value":
                size = 500
            return int(size)

    if app_protocol == "http":
        if traffic == "request":
            if mt_const == "Trace":
                size = np.mean(req_t_size)
            if mt_const == "Value":
                size = 117
            return int(size)

        if traffic == "response":
            if mt_const == "Trace":
                size = np.mean(resp_t_size)
                print("response Size: ", size)
            if mt_const == "Value":
                size = 465
            return int(size)

    if app_protocol == "udp":
        if mt_const == "Trace":
            size = np.mean(t_size)
        if mt_const == "Value":
            size = 500
        return int(size)

    if app_protocol == "tcp":
        if mt_const == "Trace":
            size = np.mean(t_size)
        if mt_const == "Value":
            size = 536
        return int(size)

  
###### Requisição TCP tem por padrão 66 ou 54 por padrão. o resto do length é exatamente o tamanho dos arquivos enviados ######
# Classe de criação das requisições no NS3
# Classe de criação da aplicação do NS3
class ReqMyApp(ns3.Application):
    # Criando variáveis auxiliares
    tid = ns.core.TypeId("ReqMyApp")
    tid.SetParent(ns3.Application.GetTypeId())
    m_socket = m_packetSize = m_nRequestPackets = m_dataRate = m_packetsRequestSent = 0
    m_peer = m_sendEvent = None
    m_running = False
    
    count_Setup = count_Start = count_Stop = count_SendRequestPacket = count_ScheduleRequestTx = count_GetSendRequestPacket = count_GetTypeId = 0
    # Inicializador da simulação
    def __init__(self):
        super(ReqMyApp, self).__init__()
    # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):

    # Função de configuração da aplicação
    def Setup(self, socket, address, nRequestPackets):
        self.count_Setup = self.count_Setup + 1
        self.m_socket = socket
        self.m_peer = address
        # self.m_packetSize = packetSize
        self.m_nRequestPackets = nRequestPackets
        # self.m_dataRate = dataRate

    # Função de inicialização da aplicação
    def StartApplication(self):
        self.count_Start = self.count_Start + 1
        if self.m_nRequestPackets > 0 and self.m_nRequestPackets > self.m_packetsRequestSent:
            self.m_running = True
            self.m_packetsRequestSent = 0
            self.m_socket.Bind()
            self.m_socket.Connect(self.m_peer)
            self.RequestSendPacket()
        else:
            self.StopApplication()
    
    # Função de parada da aplicação
    def StopApplication(self):
        self.count_Stop = self.count_Stop + 1
        self.m_running = False
        if self.m_sendEvent != None and self.m_sendEvent.IsRunning() == True:
            ns.core.Simulator.Cancel(self.m_sendEvent)
        if self.m_socket:
            self.m_socket.Close()

    # Função de envio de pacotes
    def RequestSendPacket(self):
        
        # Contabiliza a quantidade de pacotes enviados
        self.count_SendRequestPacket = self.count_SendRequestPacket + 1
        
        # Chamando variáveis globais
        
        # Método de Geração de RN
        global mt_RG
           
        # Vetor com dados do parametro de tamanho dos pacotes obtidos do trace
        global req_t_size
        global parameter

        global req_arg_size
        global req_scale_size
        global req_loc_size
        global req_dist_size

        global first_req_tcdf_size
        global first_req_trace_size


        global reader
        global traffic

        global app_protocol
        global const_Size
        global mt_const

        traffic = "request" 
        parameter = "Size"

        # Defininfo se o pacote é constante
        if const_Size == "True":
            if mt_const == "Trace" and first_req_trace_size == 0:
                read_txt(parameter, traffic, app_protocol)
            req_aux_packet = constSize(traffic, app_protocol)
            req_packet = ns.network.Packet(req_aux_packet)
        else:
            # Condição de escolha do método de geração de variáveis aleatórias 
            # diretamente por uma distribuição de probabiidade
            if mt_RG == "PD":
                # Chamando a função wgwnet_PD() e retornando valor gerado para uma variável auxiliar
                req_aux_packet = wgwnet_PD(parameter, traffic)
                
                # Transformando a variávei auxiliar em um metadado de pacote
                req_packet = ns.network.Packet(req_aux_packet)

            
            # Condição de escolha do método de geração de variáveis aleatórias 
            # baseado nos dados do trace
            if first_req_tcdf_size == 0 and (mt_RG == "ecdf" or mt_RG == "tcdf"):

                if first_req_trace_size == 0:
                    # Definindo o método de leitura do arquivo trace
                    if reader == "txt":
                        read_txt(parameter, traffic, app_protocol)
                    if reader == "xml":
                        read_xml(parameter)
                
                # Condição de escolha do método por distribuições teórica equivalentes aos dados do trace
                if mt_RG == "tcdf":
                    # Condição de chamada única da função tcdf()
                    # Chamando a função tcdf para definir a distribuição de probabilidade compatível ao trace e 
                    # seus respectivos parametros para geração de números aleatórios
                    if first_req_tcdf_size == 0:
                        tcdf(req_t_size, parameter, traffic)

                    # Chamando a função tcdf_generate e retornando valor gerado para uma variável auxiliar
                    req_aux_packet = tcdf_generate(req_dist_size, req_loc_size, req_scale_size, req_arg_size, parameter)
                    # Transformando a variávei auxiliar em um metadado de pacote
                    req_packet = ns.network.Packet(req_aux_packet)
                    
        

                # Condição de escolha do método pela distribuição empírica dos dados do trace
                if mt_RG == "ecdf":
                    # Chamando a função ecdf e retornando valor gerado para uma variável auxiliar
                    req_aux_packet = ecdf(req_t_size, parameter, traffic)
                    # Transformando a variávei auxiliar em um metadado de pacote
                    req_packet = ns.network.Packet(req_aux_packet)
                
        # Imprimindo o tempo de envio do pacote e a quantidade de pacotes enviados
        print ("SendRequestPacket(): ", str(ns.core.Simulator.Now().GetSeconds()), "s,\t send ", str(self.m_packetsRequestSent), " Size ", req_packet.GetSize(), "#")
        
        # Configurando o socket da rede para enviar o pacote
        self.m_socket.Send(req_packet, 0)
        
        # Incrementando a quantidade de pacotes enviados
        # if traffic == "request":
        self.m_packetsRequestSent = self.m_packetsRequestSent + 1
        
        # Condição de parada da aplicação pela quantidade máxima de pacotes
        if self.m_packetsRequestSent < self.m_nRequestPackets:
            self.ScheduleRequestTx()
        else:
            self.StopApplication()
    
    # Função que prepara os eventos de envio de pacotes
    def ScheduleRequestTx(self):
        
        # Contabiliza a quantidade eventos que ocorrem na simulação
        self.count_ScheduleRequestTx = self.count_ScheduleRequestTx + 1

        # Condição que define se a aplicação ainda terá eventos
        if self.m_running:
            # Chamando variáveis globais
            # Auxiliar de tempo
            global req_aux_global_time
            # Método de Geração de RN
            global mt_RG
            
            # Vetor com dados do parametro de tamanho dos pacotes obtidos do trace
            global req_t_time

            global req_arg_time
            global req_scale_time
            global req_loc_time
            global req_dist_time

            global first_req_tcdf_time
            global first_req_trace_time

            global reader
            global traffic
            global app_protocol
            
            parameter = "Time"
            traffic = "request" 
            # Condição de escolha do método de geração de variáveis aleatórias 
            # diretamente por uma distribuição de probabiidade
            if mt_RG == "PD":
                # Chamando a função wgwnet_PD() e retornando valor gerado para uma variável auxiliar
                req_aux_global_time = wgwnet_PD(parameter, traffic)
                
            # Condição de escolha do método de geração de variáveis aleatórias 
            # baseado nos dados do trace
            if first_trace_time == 0 and mt_RG == "tcdf" or mt_RG == "ecdf":

                # Definindo o método de leitura do arquivo trace            
                if first_req_trace_time == 0:
                    # Definindo o método de leitura do arquivo trace
                    if reader == "txt":
                        read_txt(parameter, traffic, app_protocol)
                    if reader == "xml":
                        read_xml(parameter)

                # Condição de escolha do método por distribuições teórica equivalentes aos dados do trace
                if mt_RG == "tcdf":
                    # Condição de chamada única da função tcdf()
                    if traffic == "request" and first_req_tcdf_time == 0:
                        tcdf(req_t_time, parameter, traffic)

                    # Chamando a função tcdf_generate e retornando valor gerado para uma variável auxiliar
                    req_aux_global_time = tcdf_generate(req_dist_time, req_loc_time, req_scale_time, req_arg_time, parameter)
                
                # Condição de escolha do método pela distribuição empírica dos dados do trace
                if mt_RG == "ecdf":
                    # Chamando a função ecdf e retornando valor gerado para uma variável auxiliar
                    req_aux_global_time = ecdf(req_t_time, parameter, traffic)
                    print("Time: ", req_aux_global_time)
            # Transformando a variávei auxiliar em um metadado de tempo 
            req_tNext = ns.core.Seconds(req_aux_global_time)
            # req_tNext = ns.core.Seconds(10)
            
            # Criando evento de envio de pacote
            global nRequests
            nRequests = nRequests + 1
            # self.m_sendEvent = ns.core.Simulator.Schedule(req_tNext, ReqMyApp.RequestSendPacket, self)
            global printo
            printo.append(req_tNext)
            w = open("../../../Results/Prints/printo_ecdf_Time.txt", "a")

            w.write("\n"+ str(req_aux_global_time)+"\n")
            w.close()
            self.m_sendEvent = ns.core.Simulator.Schedule(req_tNext, ReqMyApp.RequestSendPacket, self)

    def GetSendRequestPacket(self):
        self.count_GetSendRequestPacket = self.count_GetSendRequestPacket + 1
        return self.m_packetsRequestSent

    def GetTypeId(self):
        self.count_GetTypeId = self.count_GetTypeId + 1
        return self.tid

class RespMyApp(ns3.Application):
    # Criando variáveis auxiliares 
    tid = ns.core.TypeId("RespMyApp")
    tid.SetParent(ns3.Application.GetTypeId())
    m_socket = m_packetSize = m_nResponsePackets = m_dataRate = m_packetsResponseSent = 0
    m_peer = m_sendEvent = None
    m_running = False
    
    count_Setup = count_Start = count_Stop = count_SendResponsePacket = count_ScheduleResponseTx = count_GetSendResponsePacket = count_GetTypeId = 0
    # Inicializador da simulação
    def __init__(self):
        super(RespMyApp, self).__init__()
    # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):

    # Função de configuração da aplicação
    def Setup(self, socket, address, nResponsePackets):
        self.count_Setup = self.count_Setup + 1
        self.m_socket = socket
        self.m_peer = address
        # self.m_packetSize = packetSize
        self.m_nResponsePackets = nResponsePackets
        # self.m_dataRate = dataRate

    # Função de inicialização da aplicação
    def StartApplication(self):
        self.count_Start = self.count_Start + 1
        if self.m_nResponsePackets > 0 and self.m_nResponsePackets > self.m_packetsResponseSent:
            self.m_running = True
            self.m_packetsResponseSent = 0
            self.m_socket.Bind()
            self.m_socket.Connect(self.m_peer)
            self.ResponseSendPacket()
        else:
            self.StopApplication()
    
    # Função de parada da aplicação
    def StopApplication(self):
        self.count_Stop = self.count_Stop + 1
        self.m_running = False
        if self.m_sendEvent != None and self.m_sendEvent.IsRunning() == True:
            ns.core.Simulator.Cancel(self.m_sendEvent)
        if self.m_socket:
            self.m_socket.Close()

    # Função de envio de pacotes
    def ResponseSendPacket(self):

        global nRequests
        print("Send Response: ", nRequests)
        # if nRequests > 0:
        # Contabiliza a quantidade de pacotes enviados
        self.count_SendResponsePacket = self.count_SendResponsePacket + 1
        
        # Chamando variáveis globais
        # Método de Geração de RN
        global mt_RG
        
        # Vetor com dados do parametro de tamanho dos pacotes obtidos do trace
        global resp_arg_size
        global resp_scale_size
        global resp_loc_size
        global resp_dist_size

        global first_resp_tcdf_size
        global first_resp_trace_size
        
        global reader
        global traffic
        global app_protocol
        global const_Size
        global mt_const

        traffic = "response"
        parameter = "Size"
        # Defininfo se o pacote é constante
        if const_Size == "True":
            if mt_const == "Trace" and first_resp_trace_size == 0:
                read_txt(parameter, traffic, app_protocol)
            resp_aux_packet = constSize(traffic, app_protocol)
            resp_packet = ns.network.Packet(resp_aux_packet)
        else:
            # Condição de escolha do método de geração de variáveis aleatórias 
            # diretamente por uma distribuição de probabiidade
            if mt_RG == "PD":
                # Chamando a função wgwnet_PD() e retornando valor gerado para uma variável auxiliar
                resp_aux_packet = wgwnet_PD(parameter, traffic)
                
                # Transformando a variávei auxiliar em um metadado de pacote
                resp_packet = ns.network.Packet(resp_aux_packet)

            # Condição de escolha do método de geração de variáveis aleatórias 
            # baseado nos dados do trace
            if first_resp_tcdf_size == 0 and (mt_RG == "ecdf" or mt_RG == "tcdf"):
                
                if first_resp_trace_size == 0:
                    # Definindo o método de leitura do arquivo trace
                    if reader == "txt":
                        read_txt(parameter, traffic, app_protocol)
                    if reader == "xml":
                        read_xml(parameter)
                
                # Condição de escolha do método por distribuições teórica equivalentes aos dados do trace
                if mt_RG == "tcdf":
                    # Condição de chamada única da função tcdf()
                    if first_resp_tcdf_size == 0:
                        tcdf(resp_t_size, parameter, traffic)

                    # Chamando a função tcdf_generate e retornando valor gerado para uma variável auxiliar
                    resp_aux_packet = tcdf_generate(resp_dist_size, resp_loc_size, resp_scale_size, resp_arg_size, parameter)
                    # Transformando a variávei auxiliar em um metadado de pacote
                    resp_packet = ns.network.Packet(resp_aux_packet)
                    
        

                # Condição de escolha do método pela distribuição empírica dos dados do trace
                if mt_RG == "ecdf":
                    # Chamando a função ecdf e retornando valor gerado para uma variável auxiliar
                    resp_aux_packet = ecdf(resp_t_size, parameter, traffic)
                    # Transformando a variávei auxiliar em um metadado de pacote
                    resp_packet = ns.network.Packet(resp_aux_packet)
                 
        # Imprimindo o tempo de envio do pacote e a quantidade de pacotes enviados
        print ("SendResponsePacket(): ", str(ns.core.Simulator.Now().GetSeconds()), "s,\t send ", str(self.m_packetsResponseSent), " Size ", resp_packet.GetSize(), "#")
        
        # Configurando o socket da rede para enviar o pacote
        self.m_socket.Send(resp_packet, 0)
        
        # Incrementando a quantidade de pacotes enviados
        
        self.m_packetsResponseSent = self.m_packetsResponseSent + 1
        
        # Condição de parada da aplicação pela quantidade máxima de pacotes
        if self.m_packetsResponseSent < self.m_nResponsePackets:
            self.ScheduleResponseTx()
        else:
            self.StopApplication()
    
    # Função que prepara os eventos de envio de pacotes
    def ScheduleResponseTx(self):
        
        # Contabiliza a quantidade eventos que ocorrem na simulação
        self.count_ScheduleResponseTx = self.count_ScheduleResponseTx + 1

        # Condição que define se a aplicação ainda terá eventos
        if self.m_running:
            # Chamando variáveis globais
            # Auxiliar de tempo
            global resp_aux_global_time
            # Método de Geração de RN
            global mt_RG
            
            # Vetor com dados do parametro de tamanho dos pacotes obtidos do trace
            global resp_t_time

            global resp_arg_time
            global resp_scale_time
            global resp_loc_time
            global resp_dist_time

            global first_resp_tcdf_time
            global first_resp_trace_time
            
            global reader
            global traffic
         
            global app_protocol
            
            parameter = "Time"
            traffic = "response"
 
             # Condição de escolha do método de geração de variáveis aleatórias 
            # diretamente por uma distribuição de probabiidade
            if mt_RG == "PD":
                # Chamando a função wgwnet_PD() e retornando valor gerado para uma variável auxiliar
                resp_aux_global_time = wgwnet_PD(parameter, traffic)
                
            # Condição de escolha do método de geração de variáveis aleatórias 
            # baseado nos dados do trace
            if first_resp_tcdf_time == 0 and (mt_RG == "ecdf" or mt_RG == "tcdf"):

                # Definindo o método de leitura do arquivo trace
                if first_resp_trace_time == 0:
                    # Definindo o método de leitura do arquivo trace
                    if reader == "txt":
                        read_txt(parameter, traffic, app_protocol)
                    if reader == "xml":
                        read_xml(parameter)
                
                # Condição de escolha do método por distribuições teórica equivalentes aos dados do trace
                if mt_RG == "tcdf":
                    # Condição de chamada única da função tcdf()
                    if traffic == "response" and first_resp_tcdf_time == 0:
                        tcdf(resp_t_time, parameter, traffic)
                    
                    # Chamando a função tcdf_generate e retornando valor gerado para uma variável auxiliar
                    resp_aux_global_time = tcdf_generate(resp_dist_time, resp_loc_time, resp_scale_time, resp_arg_time, parameter)
                
                # Condição de escolha do método pela distribuição empírica dos dados do trace
                if mt_RG == "ecdf":
                    # Chamando a função ecdf e retornando valor gerado para uma variável auxiliar
                    resp_aux_global_time = ecdf(resp_t_time, parameter, traffic)

            # Transformando a variávei auxiliar em um metadado de tempo 
            resp_tNext = ns.core.Seconds(resp_aux_global_time)
            # resp_tNext = ns.core.Seconds(10)
            
            
            # Criando evento de envio de pacote
            global nRequests
            nRequests = nRequests - 1
            print("Schedule Response: ", nRequests)
            # self.m_sendEvent = ns.core.Simulator.Schedule(resp_tNext, RespMyApp.ResponseSendPacket, self)
            self.m_sendEvent = ns.core.Simulator.Schedule(resp_tNext, RespMyApp.ResponseSendPacket, self)

    def GetSendResponsePacket(self):
        self.count_GetSendResponsePacket = self.count_GetSendResponsePacket + 1
        return self.m_packetsResponseSent

    def GetTypeId(self):
        self.count_GetTypeId = self.count_GetTypeId + 1
        return self.tid
time_simulate = []
dts = []
dns = []
aux_dts = 0
aux_dns = 0
# Classe de criação da aplicação do NS3
class MyApp(ns3.Application):
    # Criando variáveis auxiliares
    tid = ns.core.TypeId("MyApp")
    tid.SetParent(ns3.Application.GetTypeId())
    m_socket = m_packetSize = m_nPackets = m_dataRate = m_packetsSent = 0
    m_peer = m_sendEvent = None
    m_running = False
    
    count_Setup = count_Start = count_Stop = count_SendPacket = count_ScheduleTx = count_GetSendPacket = count_GetTypeId = 0
    # Inicializador da simulação
    def __init__(self):
        super(MyApp, self).__init__()
    # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):

    # Função de configuração da aplicação
    def Setup(self, socket, address, nPackets):
        self.count_Setup = self.count_Setup + 1
        self.m_socket = socket
        self.m_peer = address
        # self.m_packetSize = packetSize
        self.m_nPackets = nPackets
        # self.m_dataRate = dataRate

    # Função de inicialização da aplicação
    def StartApplication(self):
        self.count_Start = self.count_Start + 1
        if self.m_nPackets > 0 and self.m_nPackets > self.m_packetsSent:
            self.m_running = True
            self.m_packetsSent = 0
            self.m_socket.Bind()
            self.m_socket.Connect(self.m_peer)
            self.SendPacket()
        else:
            self.StopApplication()
    
    # Função de parada da aplicação
    def StopApplication(self):
        self.count_Stop = self.count_Stop + 1
        self.m_running = False
        if self.m_sendEvent != None and self.m_sendEvent.IsRunning() == True:
            ns.core.Simulator.Cancel(self.m_sendEvent)
        if self.m_socket:
            self.m_socket.Close()

    # Função de envio de pacotes
    def SendPacket(self):
        # 
        # Contabiliza a quantidade de pacotes enviados
        self.count_SendPacket = self.count_SendPacket + 1
        
        # Chamando variáveis globais
        
        # Método de Geração de RN
        global mt_RG

        # Vetor com dados do parametro de tamanho dos pacotes obtidos do trace
        global t_size
        global arg_size
        global scale_size
        global loc_size
        global dist_size

        global first_tcdf_size
        global first_trace_size
        
        global reader
        global traffic
      
        global app_protocol
        global const_Size
        global mt_const
     
        traffic = "send"
        parameter = "Size"

        # Defininfo se o pacote é constante
        if const_Size == "True":
            if mt_const == "Trace" and first_trace_size == 0:
                read_txt(parameter, traffic, app_protocol)
            aux_packet = constSize(traffic, app_protocol)
            packet = ns.network.Packet(aux_packet)
        else:
            # Condição de escolha do método de geração de variáveis aleatórias 
            # diretamente por uma distribuição de probabiidade
            if mt_RG == "PD":
                # Chamando a função wgwnet_PD() e retornando valor gerado para uma variável auxiliar
                aux_packet = wgwnet_PD(parameter, traffic)
                
                # Transformando a variávei auxiliar em um metadado de pacote
                packet = ns.network.Packet(aux_packet)

            
            # Condição de escolha do método de geração de variáveis aleatórias 
            # baseado nos dados do trace
            if first_trace_size == 0 and (mt_RG == "ecdf" or mt_RG == "tcdf"):
                
                if first_trace_size == 0:
                    # Definindo o método de leitura do arquivo trace
                    if reader == "txt":
                        read_txt(parameter, traffic, app_protocol)
                    if reader == "xml":
                        read_xml(parameter)
                

                
                # Condição de escolha do método por distribuições teórica equivalentes aos dados do trace
                if mt_RG == "tcdf":
                    # Condição de chamada única da função tcdf()
                    if first_tcdf_size == 0:
                        tcdf(t_size, parameter, traffic)

                    # Chamando a função tcdf_generate e retornando valor gerado para uma variável auxiliar
                    aux_packet = tcdf_generate(dist_size, loc_size, scale_size, arg_size, parameter)
                    # Transformando a variávei auxiliar em um metadado de pacote
                    packet = ns.network.Packet(aux_packet)
                    
        

                # Condição de escolha do método pela distribuição empírica dos dados do trace
                if mt_RG == "ecdf":
                    # Chamando a função ecdf e retornando valor gerado para uma variável auxiliar
                    aux_packet = ecdf(t_size, parameter, traffic)
                    # Transformando a variávei auxiliar em um metadado de pacote
                    packet = ns.network.Packet(aux_packet)

        

        # Imprimindo o tempo de envio do pacote e a quantidade de pacotes enviados
        print ("SendPacket(): ", str(ns.core.Simulator.Now().GetSeconds()), "s,\t send ", str(self.m_packetsSent), " Size ", packet.GetSize(), "#")

        # Configurando o socket da rede para enviar o pacote
        self.m_socket.Send(packet, 0)
        
        # Incrementando a quantidade de pacotes enviados
        self.m_packetsSent = self.m_packetsSent + 1
        
        # Condição de parada da aplicação pela quantidade máxima de pacotes
        if self.m_packetsSent < self.m_nPackets:
            self.ScheduleTx()
        else:
            self.StopApplication()
    
    # Função que prepara os eventos de envio de pacotes
    def ScheduleTx(self):
        # Contabiliza a quantidade eventos que ocorrem na simulação
        self.count_ScheduleTx = self.count_ScheduleTx + 1

        # Condição que define se a aplicação ainda terá eventos
        if self.m_running:
            # Chamando variáveis globais
            # Auxiliar de tempo
            global aux_global_time
            # Método de Geração de RN
            global mt_RG
            # Vetor com dados do parametro de tamanho dos pacotes obtidos do trace
            global t_time

            global arg_time
            global scale_time
            global loc_time
            global dist_time

            global first_tcdf_time
            global first_trace_time
            
            global reader
            global traffic
            global app_protocol
            
            parameter = "Time"
            traffic = "send"

            # Condição de escolha do método de geração de variáveis aleatórias 
            # diretamente por uma distribuição de probabiidade
            if mt_RG == "PD":
                # Chamando a função wgwnet_PD() e retornando valor gerado para uma variável auxiliar
                aux_global_time = wgwnet_PD(parameter, traffic)
                
            # Condição de escolha do método de geração de variáveis aleatórias 
            # baseado nos dados do trace
            if first_trace_time == 0 and (mt_RG == "ecdf" or mt_RG == "tcdf"):
                # Definindo o método de leitura do arquivo trace
                if reader == "txt":
                    read_txt(parameter, traffic, app_protocol)
                if reader == "xml":
                    read_xml(parameter)
           
            # Condição de escolha do método por distribuições teórica equivalentes aos dados do trace
            if mt_RG == "tcdf":
                # Condição de chamada única da função tcdf()
                if first_tcdf_time == 0:
                    tcdf(t_time, parameter, traffic)

                # Chamando a função tcdf_generate e retornando valor gerado para uma variável auxiliar                
                aux_global_time = tcdf_generate(dist_time, loc_time, scale_time, arg_time, parameter)


            # Condição de escolha do método pela distribuição empírica dos dados do trace
            if mt_RG == "ecdf":
                # Chamando a função ecdf e retornando valor gerado para uma variável auxiliar
                aux_global_time = ecdf(t_time, parameter, traffic)
                # aux_global_time = aux_global_time - 0.000001

                global time_simulate
                time_simulate.append(aux_global_time)
                # print("SUM: ",np.sum(time_simulate))
            # aux_global_time = 10
            # Transformando a variávei auxiliar em um metadado de tempo 
            tNext = ns.core.Seconds(aux_global_time)
            # tNext = ns.core.Seconds(1)
            #  tNext = ns.core.Seconds(0.002)
            
            
            
            # Criando evento de envio de pacote
            # self.m_sendEvent = ns.core.Simulator.Schedule(tNext, MyApp.SendPacket, self)
            self.m_sendEvent = ns.core.Simulator.Schedule(tNext, MyApp.SendPacket, self)
            
            global dts
            global dns
            global aux_dts
            global aux_dns
            aux_global_time = aux_global_time
            aux_dns = aux_dns + aux_global_time
            
            dns.append(aux_dns)
            dts.append(float(ns.core.Simulator.Now().GetSeconds()))
            # print ("SendPacket(time_simulate): ", str(np.sum(time_simulate)), "s,\t send ", str(self.m_packetsSent), "#")
            # print ("SendPacket(NOW): ", str(ns.core.Simulator.Now().GetSeconds()), "s,\t send ", str(self.m_packetsSent), "#")




    def GetSendPacket(self):
        self.count_GetSendPacket = self.count_GetSendPacket + 1
        return self.m_packetsSent

    def GetTypeId(self):
        self.count_GetTypeId = self.count_GetTypeId + 1
        return self.tid

# # Função de definição da janela de congestionamento
def CwndChange(app):
	# CwndChange(): 
	# n = app.GetSendPacket()
	# print ('CwndChange(): ' + str(ns.core.Simulator.Now().GetSeconds()) + 's, \t sum(send packets) = ' + str(n))
	ns.core.Simulator.Schedule(ns.core.Seconds(1), CwndChange, app)
# Função de definição da janela de congestionamento
def RequestCwndChange(app):
	# CwndChange(): 
	# n = app.GetSendRequestPacket()
	# print ('CwndChange(): ' + str(ns.core.Simulator.Now().GetSeconds()) + 's, \t sum(send packets) = ' + str(n))
	ns.core.Simulator.Schedule(ns.core.Seconds(1), RequestCwndChange, app)
# Função de definição da janela de congestionamento
def ResponseCwndChange(app):
	# CwndChange(): 
	# n = app.GetSendRequestPacket()
	# print ('CwndChange(): ' + str(ns.core.Simulator.Now().GetSeconds()) + 's, \t sum(send packets) = ' + str(n))
	ns.core.Simulator.Schedule(ns.core.Seconds(1), ResponseCwndChange, app)

# Função de impressão dos resultados da simulação do NS3
def print_stats(w, st, flow_id, proto, t, lost_packets, throughput, delay, jitter, rate_type):
    global app_protocol
    global mt_RG
    global t_net
    if flow_id == 1:
        w = open("../../../Results/Prints/valid-"+validation+"_"+t_net+"_"+app_protocol+"_stats_"+mt_RG+"_"+str(IC)+".txt", "w")
    else: 
        w = open("../../../Results/Prints/valid-"+validation+"_"+t_net+"_"+app_protocol+"_stats_"+mt_RG+"_"+str(IC)+".txt", "a")

    w.write("\n  FlowID: "+ str(flow_id)+"\n")
    w.write("  Protocol: " + str(proto)+"\n")
    w.write("  Flow: "+str(t.sourceAddress)+"("+str(t.sourcePort)+") --> "+str(t.destinationAddress)+" ("+str(t.destinationPort)+")\n") 
    w.write("  Duration: "+ str(st.timeLastRxPacket.GetSeconds()-st.timeFirstTxPacket.GetSeconds())+"\n")
    w.write("  Last Packet Time: "+ str(st.timeLastRxPacket.GetSeconds())+ " Seconds\n")
    w.write("  Tx Bytes: "+ str(st.txBytes)+"\n")
    w.write("  Rx Bytes: "+ str(st.rxBytes)+"\n")
    w.write("  Tx Packets: "+ str(st.txPackets)+"\n")
    w.write("  Rx Packets: "+ str(st.rxPackets)+"\n")
    # w.write("  Lost Packets: "+ str(st.lostPackets)+"\n")
    if st.rxPackets > 0:
        lost_packets.append(st.lostPackets)
        w.write("  Lost Packets "+ str(lost_packets[flow_id-1]) +"\n")
        throughput.append((st.rxBytes * 8.0 / (st.timeLastRxPacket.GetSeconds()-st.timeFirstTxPacket.GetSeconds())/1024))
        rate_type = "Kbps"
        w.write("  Throughput "+ str(throughput[flow_id-1]) +" "+ rate_type +"\n")

        delay.append(st.delaySum.GetSeconds() / st.rxPackets)
        w.write("  Mean{Delay}: "+ str(delay[flow_id-1])+"\n")

        jitter.append(st.jitterSum.GetSeconds() / (st.rxPackets))
        w.write("  Mean{Jitter}: "+ str(jitter[flow_id-1])+"\n")

        w.write("  Mean{Hop Count}: "+ str(float(st.timesForwarded) / st.rxPackets + 1)+"\n")

    if st.rxPackets == 0:
        w.write("Delay Histogram\n")
        for i in range(0, st.delayHistogram.GetNBins()):
            w.write(" "+ str(i)+ "("+ str(st.delayHistogram.GetBinStart(i))+ "-"+ str(st.delayHistogram.GetBinEnd(i))+ "): "+ str(st.delayHistogram.GetBinCount(i))+"\n")
        w.write("Jitter Histogram\n")
        for i in range(0, st.jitterHistogram.GetNBins()):
            w.write(" "+ str(i)+ "("+ str(st.jitterHistogram.GetBinStart(i))+ "-"+ str(st.jitterHistogram.GetBinEnd(i))+ "): "+ str(st.jitterHistogram.GetBinCount(i))+"\n")
        w.write("PacketSize Histogram\n")
        for i in range(0, st.packetSizeHistogram.GetNBins()):
            w.write(" "+ str(i)+ "("+ str(st.packetSizeHistogram.GetBinStart(i))+ "-"+ str(st.packetSizeHistogram.GetBinEnd(i))+ "): "+ str(st.packetSizeHistogram.GetBinCount(i))+"\n")

    for reason, drops in enumerate(st.packetsDropped):
        w.write("  Packets dropped by reason "+ str(reason) +": "+ str(drops)+"\n")
    w.close()
    


    print("\n  FlowID: "+ str(flow_id))
    print("  Protocol: " + str(proto))
    print("  Flow: "+str(t.sourceAddress)+"("+str(t.sourcePort)+") --> "+str(t.destinationAddress)+" ("+str(t.destinationPort)+")") 
    print("  Duration: "+ str(st.timeLastRxPacket.GetSeconds()-st.timeFirstTxPacket.GetSeconds()))
    print("  Last Packet Time: "+ str(st.timeLastRxPacket.GetSeconds())+ " Seconds")
    print("  Tx Bytes: "+ str(st.txBytes))
    print("  Rx Bytes: "+ str(st.rxBytes))
    print("  Tx Packets: "+ str(st.txPackets))
    print("  Rx Packets: "+ str(st.rxPackets))
    # print("  Lost Packets: "+ str(st.lostPackets))
    if st.rxPackets > 0:
        lost_packets.append(st.lostPackets)
        print("  Lost Packets "+ str(lost_packets[flow_id-1]) )
        throughput.append((st.rxBytes * 8.0 / (st.timeLastRxPacket.GetSeconds()-st.timeFirstTxPacket.GetSeconds())/1024))
        rate_type = "Kbps"
        print("  Throughput "+ str(throughput[flow_id-1]) +" "+ rate_type )

        delay.append(st.delaySum.GetSeconds() / st.rxPackets)
        print("  Mean{Delay}: "+ str(delay[flow_id-1]))

        jitter.append(st.jitterSum.GetSeconds() / (st.rxPackets))
        print("  Mean{Jitter}: "+ str(jitter[flow_id-1]))

        print("  Mean{Hop Count}: "+ str(float(st.timesForwarded) / st.rxPackets + 1)+"\n")

    if st.rxPackets == 0:
        print("Delay Histogram\n")
        for i in range(0, st.delayHistogram.GetNBins()):
            print(" "+ str(i)+ "("+ str(st.delayHistogram.GetBinStart(i))+ "-"+ str(st.delayHistogram.GetBinEnd(i))+ "): "+ str(st.delayHistogram.GetBinCount(i)))
        print("Jitter Histogram\n")
        for i in range(0, st.jitterHistogram.GetNBins()):
            print(" "+ str(i)+ "("+ str(st.jitterHistogram.GetBinStart(i))+ "-"+ str(st.jitterHistogram.GetBinEnd(i))+ "): "+ str(st.jitterHistogram.GetBinCount(i)))
        print("PacketSize Histogram\n")
        for i in range(0, st.packetSizeHistogram.GetNBins()):
            print(" "+ str(i)+ "("+ str(st.packetSizeHistogram.GetBinStart(i))+ "-"+ str(st.packetSizeHistogram.GetBinEnd(i))+ "): "+ str(st.packetSizeHistogram.GetBinCount(i)))

    for reason, drops in enumerate(st.packetsDropped):
        print("  Packets dropped by reason "+ str(reason) +": "+ str(drops))


    # np.savetxt('scratch/'+app_protocol+'_req_time.txt', time_req_df['Time'], delimiter=',', fmt='%f')

# Função de comparação dos resultados obtidos com o NS3 com os dados dos traces
# Esta função é utilizada apenas quando o método de geração variáveis aleatórias selecionado é por "Trace"
def compare(app_protocol):
    # Chamando variáveis globais
    global t_time
    global t_size
    global req_t_size
    global req_t_time
    global resp_t_size
    global resp_t_time
    global plot
    global const_Size
    global t_net
    global IC
    # global time_ns3
    # global size_ns3

    if app_protocol == "udp" or app_protocol == "tcp":
        Applications = ["Send"]
        
        ns3_df = pd.read_csv("scratch/compare_"+app_protocol+".txt", sep=";",  names=["Time", "Size"])
        ns3_df = ns3_df[ns3_df.Size != 0]
        # print(ns3_df[0:10])
        # print(time_ns3_df.describe())
        time_ns3 = np.array(ns3_df['Time'])
        
        time_ns3.sort()
        # print("Pré: ",time_ns3)
        sub = []
        
        for i in range(0, len(time_ns3)-1):
            sub.append(time_ns3[i+1] - time_ns3[i])
            # print((time_ns3[i+1] - time_ns3[i]),"\t", time_ns3[i],"\t","\n")
        
        # Passando valores resultantes para a variável padrão req_t_time
        time_ns3 = np.array(sub)
        time_ns3 = time_ns3.astype(float)
        # time_ns3 = np.delete(time_ns3, np.where(time_ns3 == 0))
        time_ns3.sort()
        
        if const_Size == "False":
            size_ns3 = np.array(ns3_df['Size'])
            # size_ns3 = np.delete(size_ns3, np.where(size_ns3 == 0))

    if app_protocol == "http":
        Applications = ["Request","Response"]
        ############################# SIZE #############################
        # Abrindo arquivos .txt
        ns3_df = pd.read_csv("scratch/compare_"+app_protocol+".txt", sep=";",  names=["Time", "Port","Size"])
        
        ns3_df = ns3_df[ns3_df.Size != 0]
        # print(ns3_df[:10])

        grouped = ns3_df.groupby(ns3_df.Port)

        # req_df = grouped.get_group(8080)
        # resp_df = grouped.get_group(8081)
        resp_df = grouped.get_group(49153)
        req_df = grouped.get_group(49153)

        # print("Client: \n",client_df[:10])
        # print("Server: \n",server_df[:10])

        time_resp_ns3 = np.array(resp_df['Time'])
        time_req_ns3 = np.array(req_df['Time'])

        time_resp_ns3.sort()
        sub = []
        
        for i in range(0, len(time_resp_ns3)-1):
            sub.append(time_resp_ns3[i+1] - time_resp_ns3[i])
        
        # Passando valores resultantes para a variável padrão req_t_time
        time_resp_ns3 = np.array(sub)
        time_resp_ns3 = time_resp_ns3.astype(float)
        time_resp_ns3 = np.delete(time_resp_ns3, np.where(time_resp_ns3 == 0))
        time_resp_ns3.sort()

        time_req_ns3.sort()
        sub = []
        
        for i in range(0, len(time_req_ns3)-1):
            sub.append(time_req_ns3[i+1] - time_req_ns3[i])
        
        # Passando valores resultantes para a variável padrão req_t_time
        time_req_ns3 = np.array(sub)
        time_req_ns3 = time_req_ns3.astype(float)
        time_req_ns3 = np.delete(time_req_ns3, np.where(time_req_ns3 == 0))
        time_req_ns3.sort()
        
        

        if const_Size == "False":
            size_req_ns3 = np.array(req_df['Size'])
            size_req_ns3 = np.delete(size_req_ns3, np.where(size_req_ns3 == 0))

            size_resp_ns3 = np.array(req_df['Size'])
            size_resp_ns3 = np.delete(size_resp_ns3, np.where(size_resp_ns3 == 0))
        # time_req_ns3 = np.around(time_req_ns3, 3)
        np.savetxt('scratch/'+'compare_'+app_protocol+'_time_req_ns3.txt', time_req_ns3, delimiter=',', fmt='%f')
        del req_df
        del resp_df          
        
    if app_protocol == "ftp" or app_protocol == "hls":
        Applications = ["Request","Response", "Send"]
        ############################# REQUEST AND RESPONSE #############################
        # Abrindo arquivos .txt
        ns3_df = pd.read_csv("scratch/compare_"+app_protocol+".txt", sep=";",  names=["Time", "Port","Size"])
        # print(ns_df[:10])
        ns3_df = ns3_df[ns3_df.Size != 0]
    
        grouped = ns3_df.groupby(ns3_df.Port)

        req_df = grouped.get_group(8080)
        resp_df = grouped.get_group(8081)
        send_df = grouped.get_group(8082)

        # print("Client: \n",client_df[:10])
        # print("Server: \n",server_df[:10])

        time_req_ns3 = np.array(req_df['Time'])
        time_resp_ns3 = np.array(resp_df['Time'])
        time_ns3 = np.array(send_df['Time'])

        # time_ns3 = np.cumsum(time_ns3)
        # np.savetxt("scratch/compare_time.txt",time_ns3,fmt='%f',delimiter=',')

        time_resp_ns3.sort()
        sub = []
        
        for i in range(0, len(time_resp_ns3)-1):
            sub.append(time_resp_ns3[i+1] - time_resp_ns3[i])
        
        # Passando valores resultantes para a variável padrão req_t_time
        time_resp_ns3 = np.array(sub)
        time_resp_ns3 = time_resp_ns3.astype(float)
        time_resp_ns3 = np.delete(time_resp_ns3, np.where(time_resp_ns3 == 0))
        time_resp_ns3.sort()

        

        time_req_ns3.sort()
        sub = []
        
        for i in range(0, len(time_req_ns3)-1):
            sub.append(time_req_ns3[i+1] - time_req_ns3[i])
        
        # Passando valores resultantes para a variável padrão req_t_time
        time_req_ns3 = np.array(sub)
        time_req_ns3 = time_req_ns3.astype(float)
        time_req_ns3 = np.delete(time_req_ns3, np.where(time_req_ns3 == 0))
        time_req_ns3.sort()

        time_ns3.sort()
        # print("Pré: ",time_ns3)
        sub = []
        
        for i in range(0, len(time_ns3)-1):
            sub.append(time_ns3[i+1] - time_ns3[i])
            # print((time_ns3[i+1] - time_ns3[i]),"\t", time_ns3[i],"\t","\n")
        
        # Passando valores resultantes para a variável padrão req_t_time
        time_ns3 = np.array(sub)
        time_ns3 = time_ns3.astype(float)
        time_ns3 = np.delete(time_ns3, np.where(time_ns3 == 0))
        time_ns3.sort()
        
        if const_Size == "False":
            size_req_ns3 = np.array(req_df['Size'])
            size_req_ns3 = np.delete(size_req_ns3, np.where(size_req_ns3 == 0))

            size_resp_ns3 = np.array(req_df['Size'])
            size_resp_ns3 = np.delete(size_resp_ns3, np.where(size_resp_ns3 == 0))
            
            size_ns3 = np.array(send_df['Size'])
            size_ns3 = np.delete(size_ns3, np.where(size_ns3 == 0))
        # print("Pós: ",time_ns3)
        
        # fig, ax = plt.subplots(1, 1)
        # ax = sns.distplot(time_ns3)
        # plt.title("DF Simulate")
        # plt.show()

        del req_df
        del resp_df
        del send_df
         
   
    # Definindo o parametro da rede a ser comparado
    if const_Size == "False":
        Parameters = ["Size", "Time"]
    else:
        Parameters = ["Time"]
    
    Methods = ["kstest"]
    # Methods = ["qq_e_pp","kstest","graphical"]
    
    for meth in Methods:
        for app in Applications:
            for parameter in Parameters:
                
                if parameter == "Size" and app == "Request":
                    # Adicionando valores gerados pelo NS3
                    x = size_req_ns3
                    # x = data_size_ns3
                    # Adicionando valores do trace
                    y = req_t_size
                    # y = data_size
                if parameter == "Time" and app == "Request":
                    # Adicionando valores gerados pelo NS3
                    x = time_req_ns3
                    # x = data_size_ns3
                    # Adicionando valores do trace
                    y = req_t_time
                    # y = data_size
                if parameter == "Size" and app == "Response":
                    # Adicionando valores gerados pelo NS3
                    x = size_resp_ns3
                    # x = data_size_ns3
                    # Adicionando valores do trace
                    y = resp_t_size
                    # y = data_size
                if parameter == "Time" and app == "Response":
                    # Adicionando valores gerados pelo NS3
                    x = time_resp_ns3
                    # x = data_size_ns3
                    # Adicionando valores do trace
                    y = resp_t_time
                    # y = data_size
                if parameter == "Size" and app == "Send":
                    # Adicionando valores gerados pelo NS3
                    x = size_ns3
                    # x = data_size_ns3
                    # Adicionando valores do trace
                    y = t_size
                    # y = data_size
                if parameter == "Time" and app == "Send":
                    # Adicionando valores gerados pelo NS3
                    x = time_ns3
                    # print(len(x))
                    # x = list(dict.fromkeys(x))
                    # print("X around: ", len(x))
                    # Adicionando valores do trace
                    y = t_time
                    # y = np.around(y, 4)
                    # print("Original SEND: ", y)
                    # y = list(dict.fromkeys(y))
                    # print("Y arround: ",len(y))
                    # y = data_size
                
             
                # Métodos de comparação dos traces
                if meth == "qq_e_pp":
                    x_qq_pp = x
                    y_qq_pp = y       
                    normalize = False
                    if normalize == True:
                        sc_x = StandardScaler()
                        yy_x = x_qq_pp.reshape (-1,1)
                        sc_x.fit(yy_x)
                        y_std_x = sc_x.transform(yy_x)
                        y_std_x = y_std_x.flatten()
                        
                        data_x = y_std_x.copy()
                        
                        # S_bins = np.percentile(x,range(0,100))
                        
                        x_qq_pp = y_std_x
                        y_qq_pp = data_x
                    
                    
                    # Ordenando dados
                    x_qq_pp.sort()
                    y_qq_pp.sort()

                    # Tornando vetores do mesmo tamanho
                    if len(x_qq_pp) > len(y_qq_pp):
                        x_qq_pp = x_qq_pp[0:len(y_qq_pp)]
                    if len(x_qq_pp) < len(y_qq_pp):
                        y_qq_pp = y_qq_pp[0:len(x_qq_pp)]

                    # Criando variável com tamanho dos dados
                    # S_size = len(x_qq_pp)
                    # Criando variável com o número de bins (classes)
                    # S_bins = int(np.sqrt(S_size))
                    
                    # Criando figura
                    fig = plt.figure(figsize=(8,5)) 

                    # Adicionando subplot com método "qq plot"
                    ax1 = fig.add_subplot(121) # Grid of 2x2, this is suplot 1
                    
                    # Plotando dados comparados
                    ax1.plot(x_qq_pp,y_qq_pp,"o")
                    
                    # Definindo valor máximo e mínimo dos dados
                    min_value = np.floor(min(min(x_qq_pp),min(y_qq_pp)))
                    max_value = np.ceil(max(max(x_qq_pp),max(y_qq_pp)))

                    # Plotando linha qua segue do minimo ao máximo
                    ax1.plot([min_value,max_value],[min_value,max_value],'r--')

                    # Setando limite dos dados dentro do valor máximo e mínimo
                    ax1.set_xlim(min_value,max_value)

                    # Definindo os títulos dos eixos x e y 
                    ax1.set_xlabel('Real Trace quantiles')
                    ax1.set_ylabel('Simulated Trace quantiles')

                    # Definindo o título do gráfico
                    title = 'qq plot for real and simulated trace'+app+'('+parameter+')'
                    ax1.set_title(title)

                    # Adicionando subplot com método "pp plot"
                    ax2 = fig.add_subplot(122)
                    
                    # Calculate cumulative distributions
                    # Criando classes dos dados por percentis
                    
                    # S_bins = int(np.sqrt(len(x_qq_pp)))
                    S_bins = np.percentile(x_qq_pp,range(0,101))

                    # Obtendo conunts e o número de classes de um histograma dos dados
                    y_counts, S_bins = np.histogram(y_qq_pp, S_bins)
                    x_counts, S_bins = np.histogram(x_qq_pp, S_bins)
                    # print("y_COUNTS: ",y_counts)
                    # print("x_Counts: ",x_counts)
                    # print("y_Counts: ",y_counts)
                    
                    # Gerando somatória acumulada dos dados
                    cum_y = np.cumsum(y_counts)
                    cum_x = np.cumsum(x_counts)
                    # print("CUMSUM_DATA: ", cum_y)

                    # Normalizando a somatória acumulada dos dados
                    cum_y = cum_y / max(cum_y)
                    cum_x = cum_x / max(cum_x)
                    #print("Cum_y: ",cum_y)
                    #print("Cum_x: ",cum_x)

                    # plot
                    # Plotando dados 
                    ax2.plot(cum_x,cum_y,"o")
                    
                    # Obtendo valores máximos e minimos
                    min_value = np.floor(min(min(cum_x),min(cum_y)))
                    max_value = np.ceil(max(max(cum_x),max(cum_y)))
                    
                    # Plotando linha entre valor máximo e mínimo dos dados
                    ax2.plot([min_value,max_value],[min_value,max_value],'r--')

                    # Definindo o limite dos dados entre os valores máximos e mínimos
                    ax2.set_xlim(min_value,max_value)

                    # Definindo titulos dos eixos x e y
                    ax2.set_xlabel('Real Trace cumulative distribution')
                    ax2.set_ylabel('Simulated Trace cumulative distribution')

                    # Definindo titulo do gráfico
                    title = 'pp plot for real and simulated trace '+app+'('+parameter+')'
                    ax2.set_title(title)

                    # Exibindo gráficos
                    plt.tight_layout(pad=4)
                    if plot == "show":
                        plt.show()  
                    if plot == "save":
                        fig.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_"+app+"_"+meth+"_plot_"+parameter, fmt="png",dpi=1000)
                        plt.close()

                if meth == "graphical":
                    x_gr = x
                    y_gr = y
                    ######################################################## x = empirico -> ns3
                    ######################################################## Gerar ECDF
                    ######################################################## y = função real -> trace real
                    ######################################################## Normalizar Função
                    Fe = [] 
                    
                    # Criando ECDFs
                    
                    for i in range(1, len(x_gr)+1):
                        # ecdf i/(n+1)
                        Fe.append(i/(len(x_gr)))
                   
                    
                    # Trandformando vetorem em np.arrays()
                    # x_gr = np.array(Fe)
                    x_gr_norm = []
                    for i in range(0, len(x_gr)):
                        x_gr_norm.append(x_gr[i]/np.sum(x_gr))
                    x_gr = np.cumsum(x_gr)
                    
                    
                    y_gr_norm = []
                    for i in range(0, len(y_gr)):
                        y_gr_norm.append(y_gr[i]/np.sum(y_gr))
                    y_gr = np.cumsum(y_gr)
                    
                        # print("Média: ",np.mean(y_gr))
                        # print("Atual: ",y_gr[i])
                        # print("Atual Norm: ",y_gr_norm[i])

                    y_gr = np.array(y_gr_norm)
                    x_gr = np.array(x_gr_norm)
                    # Ordenando os valores
                    x_gr.sort()
                    y_gr.sort()

                    # print("X: ", x_gr)
                    # print("Y: ", y_gr)
                    # Tornando os vetores do mesmo tamanho
                    if len(x_gr) > len(y_gr):
                        x_gr = x_gr[0:len(y_gr)]
                    if len(x_gr) < len(y_gr):
                        y_gr = y_gr[0:len(x_gr)]
                    
                    # print("X size: ", len(x))
                    # print("Y size: ", len(y))
                    # print("X: ", x)
                    # print("Y: ", y)

                    # Plotando dados x e y
                    plt.plot(x_gr,y_gr,"o")

                    # Definindo polinomial de x e y
                    z = np.polyfit(x_gr, y_gr, 1)

                    # Gerando polinomial de 1d com os dados de z e x 
                    y_hat = np.poly1d(z)(x_gr)

                    # Plotando linha tracejada 
                    plt.plot(x_gr, y_hat, "r--", lw=1)

                    # Imprimindo resultados da regressão linear dos dados comparados
                    text = f"$y={z[0]:0.6f}x{z[1]:+0.6f}$\n$R^2 = {r2_score(y_gr,y_hat):0.6f}$"
                    plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
                        fontsize=12, verticalalignment='top')
                    # Definindo titulo do gráfico
                    plt.title('Graphical Method Inference for Real and Simulated Trace '+app+' ('+parameter+')')
                    plt.xlabel('ECDF of Real Trace Data')
                    plt.ylabel('Normalize Data of Simulated Trace')
                    if plot == "show":
                        plt.show()  
                    if plot == "save":
                        plt.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_"+app+"_"+meth+"_plot_"+parameter, fmt="png",dpi=1000)
                        plt.close()
                if meth == "kstest":  

                    x_ks = x
                    y_ks = y
                    
                    # plt.plot(np.cumsum(x_ks))
                    # plt.title("X - NS3 Trace")
                    # plt.show()
                    global time_simulate
                    time_simulate.sort()
                    np.savetxt("scratch/simulated_trace_dns.txt",time_simulate, delimiter=',', fmt='%f')

                    global dts
                    global dns
                    dts.sort()
                    dns.sort()
                    np.savetxt("scratch/dns.txt",dns, delimiter=',', fmt='%f')
                    np.savetxt("scratch/dts.txt",dts, delimiter=',', fmt='%f')
                    
                                        
                    # fig, ax = plt.subplots(1, 1)
                    # ax = sns.distplot(t_time)
                    # plt.title("Y - Real Trace")
                    # plt.show()

                    # print("Tamanho do Trace - NS3", len(x_ks))
                    # print("Tamanho do Trace - Real", len(y_ks))
                    
                    #
                    # KS TEST
                    #
                    # # Adicionando valores do trace
                    Ft = y_ks
                    # # Adocionando valores obtidos do NS3
                    t_Fe = x_ks
                    

                    # fig, ax = plt.subplots(1, 1)
                    # ax = sns.distplot(t_Fe)
                    # plt.title("X - Simulated Trace")
                    # plt.show()
          
                    # ECDF(Ft)

                    
                    
                    # t_Fe = np.cumsum(t_Fe)
                   
                    

                    # print("y_ks: ", y_ks[0:10])
                    # print("x_ks: ", x_ks[0:10])
                    # ax = sns.distplot(t_Fe)
                    # plt.show()

                    # ax = sns.distplot(Ft)
                    # plt.show()

                    # Ordenando valores 
                    t_Fe = sorted(t_Fe, reverse=True)
                    Ft = sorted(Ft, reverse=True)
                    
                    
                    
                    # Definindo mesmo tamanho para os vetores
                    if len(Ft) > len(t_Fe):
                        Ft = Ft[0:len(t_Fe)]
                    if len(Ft) < len(t_Fe):
                        t_Fe = t_Fe[0:len(Ft)]
                    
                    t_Fe.sort()
                    Ft.sort()
                    
                    
                    np.savetxt("scratch/simulated_trace.txt",t_Fe, delimiter=',', fmt='%f')
                    np.savetxt("scratch/real_trace.txt",Ft, delimiter=',', fmt='%f')
                    
                    #
                    # print("Ft: ", Ft[0:10])
                    # print("Fe: ", Fe[0:10])
                    
                    # plt.plot(np.cumsum(Ft))
                    # plt.plot(np.cumsum(t_Fe))
                    # plt.show()

                    # Criando listas para a ecdf
                    Fe = []
                    Fe_ = []

                    size = len(t_Fe)
                    # Criando ECDFs
                    
                    # size = 100
                    for i in range(1,size+1):
                        # ecdf i-1/n
                        # print("I:",i)
                        # print("I-1:",i-1)
                        # print("I-1/n:",(i-1)/size)
                        Fe_.append((i-1)/size)
                        # ecdf i/n
                        Fe.append(i/size)
                  


                    # Ordenando dados
                    t_Fe.sort()
                    Ft.sort()
                    Fe.sort()
                    Fe_.sort()

                    

           

                    # Trandformando vetorem em np.arrays()
                    Fe = np.array(Fe)
                    Fe_ = np.array(Fe_)
                    Ft = np.array(Ft)
                    t_Fe = np.array(t_Fe)

                    Fe = np.around(Fe,  3)
                    Fe_ = np.around(Fe_,  3)
                    Ft = np.around(Ft,  5)
                    t_Fe = np.around(t_Fe,  5)


                    # print("Ft: ", Ft[0:10])
                    # print("t_Fe: ", t_Fe[0:10])
                    # print("Fe_: ", Fe_[0:10])
                    # print("Fe: ", Fe[0:10])
                    # print("Sub: ", np.subtract(Ft[0:15],t_Fe[0:15]))

                    # Inicio cálculo de rejeição
                    #
                    # Ft(t)-FE-(i),FE+(i)-Ft(t)
                    Ft_Fe_ = np.subtract(Ft, Fe_)
                    Fe_Ft = np.subtract(Fe, Ft)
                    
                    # Max(Ft(t)-FE-(i),FE+(i)-Ft(t))
                    Dobs_max = np.maximum(Ft_Fe_, Fe_Ft)
                    
                    # Dobs= Max(Max (Ft(t)-FE-(i),FE+(i)-Ft(t)))
                    Dobs = np.max(Dobs_max)
                    #
                    # Fim cálculo de rejeição
                    # alternative{‘two-sided’, ‘less’, ‘greater’}
                    # dist_name = getattr(scipy.stats, 'uniform')
                    # Ft = dist_name.rvs(loc=0, scale=10, size=len(t_Fe))
                    # Ft.sort()
                    print("Ft: ",len(Ft))
                    print("t_Fe: ",len(t_Fe))
                    
                    t_Fe = np.around(t_Fe, 2)
                    Ft = np.around(Ft, 2)

                    np.savetxt("scratch/ALT_simulated_trace.txt", t_Fe, delimiter=',', fmt='%f')
                    np.savetxt("scratch/ALT_real_trace.txt", Ft, delimiter=',', fmt='%f')
                    
                    
                    
                    
                    # Modos disponíveis ['asymp','exact', 'auto']
                    mode = ['exact']
                    # Alternativas disponíveis ['two-sided', 'less', 'greater']
                    alternative = ['less']
                    for md in mode:
                        for alt in alternative:
                            # if size < 10000:
                            ks_statistic, p_value = stats.ks_2samp(Ft,t_Fe, mode=''+md+'', alternative=''+alt+'')
                            ks_statistic = np.around(ks_statistic, 2)
                            p_value = np.around(p_value, 2)
                            # else:
                            #     ks_statistic, p_value = stats.ks_2samp(Ft,t_Fe, mode='asymp', alternative='two-sided')
                                

                            rejects, IC, D_critico = ksvalid(size, ks_statistic)
                            # rejects, IC, D_critico = ksvalid(size, Dobs)

                            
                            # Imprimindo resultados do KS Test
                            print(" ")
                            print("KS TEST(",app,"):")
                            print("Confidence degree: ", IC,"%")
                            print("D observed: ", Dobs)
                            print("D observed(two samples): ", ks_statistic)
                            print("D critical: ", D_critico)
                            print(rejects, " to  Real Trace (Manual-ks_statistic/D_critico)")

                            a = 1-(IC/100)
                            
                            a = np.around(a,5)

                            if p_value < a:
                                print("Reject - p-value: ", p_value, " is less wich alpha: ", a," (2samp)")
                            else:
                                print("Fails to Reject - p-value: ", p_value, " is greater wich alpha: ", a," (2samp)")
                            

                            w = open("../../../Results/Prints/valid-"+validation+"_"+t_net+"_"+app_protocol+"_stats_"+mt_RG+"_"+str(IC)+".txt", "a")
                            w.write("\nKS TEST("+str(app)+")_"+alt+"_"+md+":"+"\n")
                            w.write("Confidence degree: "+ str(IC)+"%\n")
                            w.write("D observed: "+ str(Dobs)+"\n")
                            w.write("D observed(two samples): "+ str(ks_statistic)+"\n")
                            w.write("D critical: "+str(D_critico)+"\n")
                            w.write(str(rejects)+ " to  Real Trace (Manual-ks_statistic/D_critico)\n")
                            
                            if p_value < a:
                                w.write("Reject - p-value: "+ str(p_value)+ " is less wich alpha: "+ str(a)+" (2samp)\n")
                            else:
                                w.write("Fails to Reject - p-value: "+ str(p_value)+ " is greater wich alpha: "+ str(a)+" (2samp)\n")
                            w.close()
                            # Gerar número aleatório de acordo com a distribuição escolhida e seus parametros.
                            
                            # Plotando resultados do teste KS
                    
                            plt.plot(Ft, Fe, '-', label='Simulated trace data')
                            plt.plot(t_Fe, Fe, '-', label='Real trace data')
                            
                            
                            # plt.plot(t_Fe, Ft, 'o', label='Teorical Distribution')
                            # plt.plot(t_Fe, Fe, 'o', label='Empirical Distribution')
                            # Definindo titulo
                            # plt.title("KS test of Real and Simulated Trace of "+app+" ("+parameter+")")
                            plt.legend(loc='lower right',fontsize=12)
                            if plot == "show":
                                plt.show()  
                            if plot == "save":
                                plt.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_"+app+"_"+meth+"_plot_"+parameter+"_("+alt+"-"+md+")", fmt="png",dpi=1000)
                                plt.close()
                            
                            # Definindo diferença entre traces
                            diff = Ft - t_Fe
                            fig, ax = plt.subplots(1, 1)
                            ax = sns.distplot(diff)
                            # plt.title("Histogram of differential "+traffic+" ("+parameter+")")
                            
                            # plt.hist(diff)
                            if plot == "show":
                                plt.show()  
                            if plot == "save":
                                plt.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_"+app+"_"+meth+"_hist_"+parameter, fmt="png",dpi=1000)
                                plt.close()
                    


def kstest():
    y = [142.773, 146.217, 147.676, 147.740, 149.016, 149.105, 150.476, 151.284, 151.461, 151.763, 151.932, 154.519, 154.632, 154.789, 155.008, 155.325, 155.402, 155.506, 155.545, 155.561, 155.581, 155.584, 155.701, 156.115, 156.340, 156.851, 156.879, 157.044, 157.404, 157.435, 157.573, 157.599, 157.688, 157.717, 157.858, 158.033, 158.154, 158.387, 158.475, 159.068, 159.215, 159.234, 159.366, 159.499, 159.576, 159.601, 159.767, 159.824, 159.978, 160.036, 160.289, 160.289, 160.327, 160.430, 160.496, 160.519, 160.719, 160.745, 160.942, 161.341, 161.438, 161.683, 161.767, 161.865, 162.064, 162.289, 162.302, 162.711, 162.752, 162.855, 162.866, 162.884, 162.918, 162.947, 163.136, 164.080, 164.138, 164.479, 164.524, 164.566, 164.850, 164.965, 165.000, 165.292, 165.397, 165.408, 165.538, 165.997, 166.311, 166.327, 166.367, 166.671, 167.214, 167.690, 168.178, 170.181, 170.633, 171.434, 173.424, 179.891]
    # Set up distribution
    size = len(y)
    distribution = 'norm'
    dist = getattr(scipy.stats, distribution)
    param = dist.fit(y)

  

    #
    # KS TEST
    #
    # Criando percentil
    percentile = np.linspace(0,100,len(y))
    percentile_cut = np.percentile(y, percentile)
    
    # Criando CDF da teórica
    Ft = dist.cdf(percentile_cut, *param[:-2], loc=param[-2], scale=param[-1])
    
    
    # Criando CDF Inversa 
    Ft_ = dist.ppf(percentile_cut, *param[:-2], loc=param[-2], scale=param[-1])
    
    # Adicionando dados do trace
    t_Fe = y

    # Ordenando dados
    t_Fe.sort()
    Ft.sort()
    Ft_.sort()

    # Criando listas para armazenar as ECDFs
    Fe = []
    Fe_ = []

    # Criando ECDFs
    for i in range(1, len(y)+1):
        # ecdf i-1/n
        Fe.append((i-1)/len(y))
        # ecdf i/n
        Fe_.append(i/len(y))

    # Transformando listas em np.arrays()
    Fe = np.array(Fe)
    Fe_ = np.array(Fe_)
    Ft = np.array(Ft)
    Ft_ = np.array(Ft_)
    
    # Inicio cálculo de rejeição
    #
    # Ft(t)-FE-(i),FE+(i)-Ft(t)
    Ft_Fe_ = np.subtract(Ft, Fe_)
    Fe_Ft = np.subtract(Fe, Ft)
    
    # Max(Ft(t)-FE-(i),FE+(i)-Ft(t))
    Dobs_max = np.maximum(Ft_Fe_, Fe_Ft)
    
    # Dobs= Max(Max (Ft(t)-FE-(i),FE+(i)-Ft(t)))
    Dobs = np.max(Dobs_max)
    #
    # Fim cálculo de rejeição

    # Definir intervalo de confiança
    # IC = 99.90 -> alpha = 0.10
    # IC = 99.95 -> alpha = 0.05
    # IC = 99.975 -> alpha = 0.025
    # IC = 99.99 -> alpha = 0.01
    # IC = 99.995 -> alpha = 0.005
    # IC = 99.999 -> alpha = 0.001
    IC = 99.95        
    # Condição para definir o D_critico de acordo com o tamanho dos dados
    if size > 35:
        if IC == 99.90:
            D_critico = 1.22/np.sqrt(len(y))
        
        if IC == 99.95:
            D_critico = 1.36/np.sqrt(len(y))
        
        if IC == 99.975:
            D_critico = 1.48/np.sqrt(len(y))
        
        if IC == 99.99:
            D_critico = 1.63/np.sqrt(len(y))
        
        if IC == 99.995:
            D_critico = 1.73/np.sqrt(len(y))
        if IC == 99.999:
            D_critico = 1.95/np.sqrt(len(y))

        # Condição para aceitar a hipótese nula do teste KS
        if Dobs > D_critico:
            rejects = "Reject the Null Hypothesis"
        else:
            rejects = "Fails to Reject the Null Hypothesis"

    # Imprimindo resultados do KS Test
    print("KS TEST:")
    print("Confidence degree: ", IC,"%")
    print(rejects, " of ", distribution)
    print("D observed: ", Dobs)
    print("D critical: ", D_critico)
    print(" ")

    # Plotando resultados do teste KS
    plt.plot(t_Fe, Ft, 'o', label='Teorical Distribution')
    plt.plot(t_Fe, Fe, 'o', label='Empirical Distribution')
    
    
    # plt.plot(t_Fe, Fe, 'o', label='Real Trace')
    # plt.plot(Ft, Fe, 'o', label='Syntatic Trace')
    # Definindo titulo
    plt.title("KS Test of Real Trace with " + distribution + " Distribution (" + parameter + ")")
    plt.legend()
    plt.show() 

def hls_read(app_protocol):
    global const_Size
    # Abrindo arquivos .txt    
    
    ns3_df = pd.read_csv("scratch/"+app_protocol+"_trace.txt", sep=";", names=["req_resp", "push", "Time", "Size"])
    
    # print(ns3_df[:20])

    
    time_req_df = ns3_df[ns3_df['req_resp'].str.contains("GET")]
    time_req_df["Time"] = time_req_df["Time"].apply(pd.to_numeric)
    
    time_resp_df = ns3_df[ns3_df['req_resp'].str.contains("Partial")]
    time_resp_df["Time"] = time_resp_df["Time"].apply(pd.to_numeric)

    time_send_df = ns3_df[ns3_df['req_resp'].str.contains("TCP")]
    time_send_df["Time"] = time_send_df["Time"].apply(pd.to_numeric)
    
    # ts_test = np.array(time_send_df["Time"])
    # ts_test.sort()
    
    np.savetxt('scratch/'+app_protocol+'_resp_time.txt', time_resp_df["Time"], delimiter=',', fmt='%f')
    np.savetxt('scratch/'+app_protocol+'_req_time.txt', time_req_df["Time"], delimiter=',', fmt='%f')
    np.savetxt('scratch/'+app_protocol+'_time.txt', time_send_df["Time"], delimiter=',', fmt='%f')

    if const_Size == "False":
        ns3_df = ns3_df[ns3_df.Size > 0]
        size_send_df = ns3_df[ns3_df['req_resp'].str.contains("TCP")]
        size_send_df["Size"] = size_send_df["Size"].apply(pd.to_numeric)

        size_req_df = ns3_df[ns3_df['req_resp'].str.contains("GET")]
        size_req_df["Size"] = size_req_df["Size"].apply(pd.to_numeric)

        size_resp_df = ns3_df[ns3_df['req_resp'].str.contains("Partial")]
        size_resp_df["Size"] = size_resp_df["Size"].apply(pd.to_numeric)

        np.savetxt('scratch/'+app_protocol+'_resp_size.txt', size_resp_df['Size'], delimiter=',', fmt='%f')
        np.savetxt('scratch/'+app_protocol+'_req_size.txt', size_req_df['Size'], delimiter=',', fmt='%f')
        np.savetxt('scratch/'+app_protocol+'_size.txt', size_send_df['Size'], delimiter=',', fmt='%f')
    
    timeStopSimulation = time_send_df["Time"].iloc[-1]
    
    nRequestPackets = len(time_req_df["Time"])
    nResponsePackets = len(time_resp_df["Time"])
    nPackets = len(time_send_df["Time"])
    
    return timeStopSimulation, nRequestPackets, nResponsePackets, nPackets

# Função de definição da aplicação HTTP
def http_read(app_protocol):
    global const_Size

    ns3_df = pd.read_csv("scratch/"+app_protocol+"_trace.txt", sep=";", names=["req_resp","Time","Size"])
    # print(ns3_df)
    ns3_df = ns3_df[ns3_df.Size > 0]
    time_req_df = ns3_df[ns3_df['req_resp'].str.contains("GET")]
    # time_req_df["Time"] = time_req_df["Time"].apply(pd.to_numeric)
    # print(time_req_df)
    time_resp_df = ns3_df[ns3_df['req_resp'].str.contains("OK")]
    # print(time_resp_df)
    # time_resp_df["Time"] = time_resp_df["Time"].apply(pd.to_numeric)
    
    np.savetxt('scratch/'+app_protocol+'_resp_time.txt', time_resp_df["Time"], delimiter=',', fmt='%f')
    np.savetxt('scratch/'+app_protocol+'_req_time.txt', time_req_df["Time"], delimiter=',', fmt='%f')
   
    if const_Size == "False":
    # Abrindo arquivos .txt
        size_req_df = ns3_df[ns3_df['req_resp'].str.contains("GET")]
        # size_req_df["Size"] = size_req_df["Size"].apply(pd.to_numeric)

        size_resp_df = ns3_df[ns3_df['req_resp'].str.contains("HTTP/1.1")]
        # size_resp_df["Size"] = size_resp_df["Size"].apply(pd.to_numeric)
        size_req_df.round(5)
        size_resp_df.round(5)
        np.savetxt('scratch/'+app_protocol+'_req_size.txt', size_req_df["Size"], delimiter=',', fmt='%f')
        np.savetxt('scratch/'+app_protocol+'_resp_size.txt', size_resp_df["Size"], delimiter=',', fmt='%f')
    
    timeStopSimulation =  time_resp_df["Time"].iloc[-1]
    print(timeStopSimulation)
    nRequestPackets = len(time_req_df["Time"])
    nResponsePackets = len(time_resp_df["Time"])
    
    return timeStopSimulation, nRequestPackets, nResponsePackets
        
def ftp_read(app_protocol):
    global const_Size
    global validation

    # Abrindo arquivos .txt    
    ns3_df = pd.read_csv("scratch/"+app_protocol+"_trace.txt", sep=";", names=["req_resp","Hand", "Time", "Size"])
    
    ns3_df = ns3_df[ns3_df.Size != 0]
    ns3_df.dropna(subset = ["Size"], inplace=True)
    
    # print(ns3_df[0:15])


    grouped_req_resp = ns3_df.groupby(ns3_df.req_resp)
    
    req_df = grouped_req_resp.get_group(1)
    # req_df["req_resp"] = req_df.drop(req_df["req_resp"].index[-1])
    # print(req_df)
    resp_df = grouped_req_resp.get_group(0)
    # print(resp_df)
    # resp_df["req_resp"] = resp_df.drop(resp_df["req_resp"].index[-1])
    
    np.savetxt('scratch/'+app_protocol+'_resp_time.txt', resp_df['Time'], delimiter=',', fmt='%f')
    np.savetxt('scratch/'+app_protocol+'_req_time.txt', req_df['Time'], delimiter=',', fmt='%f')

    if const_Size == "False":
        np.savetxt('scratch/'+app_protocol+'_resp_size.txt', resp_df["Size"], delimiter=',', fmt='%f')
        np.savetxt('scratch/'+app_protocol+'_req_size.txt', req_df["Size"], delimiter=',', fmt='%f')
    # ns3_df['Hand'] = ns3_df['Hand'].apply(pd.to_numeric)

    send_df = ns3_df[ns3_df[ns3_df['Hand'] == "22,22,22"].index[0]+1:]
    # print(send_df[:150])

    np.savetxt('scratch/'+app_protocol+'_time.txt', send_df['Time'], delimiter=',', fmt='%f')

    if const_Size == "False":
        np.savetxt('scratch/'+app_protocol+'_time.txt', send_df['Size'], delimiter=',', fmt='%f')

    
    timeStopSimulation = send_df["Time"].iloc[-1]
    timeStopReq_Resp = float((resp_df['Time'].iloc[-1]-1)-req_df['Time'].iloc[0])
    nRequestPackets = len(req_df["Time"])
    nResponsePackets = len(resp_df["Time"])
    nPackets = len(send_df["Time"])
    # print(timeStopReq_Resp)
    # print(timeStopSimulation)
    # print(nRequestPackets)
    # print(nResponsePackets)
    # print(nPackets)
    return timeStopSimulation, timeStopReq_Resp ,nRequestPackets, nResponsePackets, nPackets
    
def udp_read(app_protocol):
    global const_Size
    global validation
    ns3_df = pd.read_csv("scratch/"+app_protocol+"_trace.txt", sep=";", names=["Time", "Size"])
    # print(ns3_df[:10])
    ns3_df = ns3_df[ns3_df.Size > 1000]
    ns3_df.dropna(subset = ["Size"], inplace=True)
    # ns3_df = ns3_df[ns3_df[ns3_df['Size'] > 1000].index[0]+1:]
    
    # print("DF: ",len(ns3_df["Time"]))
    # send_df = ns3_df[ns3_df[ns3_df['dns'] == "NaN"].index[0]+1:]

    np.savetxt('scratch/'+app_protocol+'_time.txt', ns3_df['Time'], delimiter=',', fmt='%f')

    if const_Size == "False":
        np.savetxt('scratch/'+app_protocol+'_time.txt', ns3_df['Size'], delimiter=',', fmt='%f')


    timeStopSimulation = ns3_df["Time"].iloc[-1]
    nPackets = len(ns3_df["Time"])


    return timeStopSimulation, nPackets

def tcp_read(app_protocol):
    global const_Size
    global validation
    ns3_df = pd.read_csv("scratch/"+app_protocol+"_trace.txt", sep=";", names=["Time", "Size"])
    
    ns3_df = ns3_df[ns3_df.Size != 0]
    ns3_df.dropna(subset = ["Size"], inplace=True)

    # send_df = ns3_df[ns3_df[ns3_df['dns'] == "NaN"].index[0]+1:]

    np.savetxt('scratch/'+app_protocol+'_time.txt', ns3_df['Time'], delimiter=',', fmt='%f')

    if const_Size == "False":
        np.savetxt('scratch/'+app_protocol+'_time.txt', ns3_df['Size'], delimiter=',', fmt='%f')



    timeStopSimulation = ns3_df["Time"].iloc[-1]
    nPackets = len(ns3_df["Time"])


    return timeStopSimulation, nPackets

# Add some text for labels, title and custom x-axis tick labels, etc.
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    rotation=90,
                    ha='center', va='bottom')

# Função principal do código
def main(argv):
    # kstest()
    global reader
    global mt_RG
    global IC
    global traffic
    global app_protocol
    global const_Size
    global mt_const
    global validation
    global plot_graph
    global t_net
    # if (mt_RG == "Trace"):
    # Obtendo informações por linha de comando
    cmd = ns.core.CommandLine ()
    # cmd.timeStopRequest = 0
    cmd.nRequestPackets = 0
    cmd.nResponsePackets = 0
    cmd.nPackets = 0
    cmd.run = 0
    cmd.timeStopSimulation = 0
    cmd.app_protocol = "0"
    cmd.const_Size = "0"
    cmd.mt_const = "0"
    cmd.IC = "0"
    cmd.validation = "0"
    cmd.mt_RG = "0"
    # cmd.AddValue ("timeStopRequest", "Tempo final de requisições do cliente")
    cmd.AddValue("nRequestPackets", "Número de pacotes solicitados pelo cliente")
    cmd.AddValue("nResponsePackets", "Número de pacotes enviados pelo servidor")
    cmd.AddValue("nPackets", "Número de pacotes enviados pelo servidor")
    cmd.AddValue("timeStopSimulation", "Tempo final da simulação")
    cmd.AddValue("app_protocol", "Protocolo da aplicação")
    cmd.AddValue("const_Size", "Tipo do tamanho dos pacotes")
    cmd.AddValue("mt_const", "Tipo do valor constante")
    cmd.AddValue("validation", "Definição de comparação")
    cmd.AddValue("run", "Definição quantas vezes a simulação será executada")
    cmd.AddValue("mt_RG", "Tipo de gerador de carga utilizado")
    cmd.AddValue("IC", "Intervalo de Confiança do KSTest")
    
    cmd.Parse (sys.argv)
    # Definindo a quantidade de pacotes
    nRequestPackets = int (cmd.nRequestPackets)
    nResponsePackets = int (cmd.nResponsePackets)
    nPackets = int (cmd.nPackets)
    run = int (cmd.run)
    # Definindo o tempo de parada da simulação
    timeStopSimulation = float (cmd.timeStopSimulation)
    # timeStopRequest = float (cmd.timeStopRequest)
    # Definindo o protocolo da aplicação
    app_protocol = cmd.app_protocol
    const_Size = cmd.const_Size
    mt_const = cmd.mt_const
    validation = cmd.validation
    mt_RG = cmd.mt_RG
    IC = cmd.IC

    if validation == "True":
        plot_graph = True

    ##### Variáveis aleatórias do NS3
    # sz = ns.core.UniformRandomVariable()
    # sz.SetAttribute("Min", ns.core.DoubleValue(50))
    # sz.SetAttribute("Max", ns.core.DoubleValue(90))
    # print("Random Number Uniform: ",  sz.GetValue())
    
    # #  Deterministic Random Variable
    # deter = ns.core.DeterministicRandomVariable ()
    # # array = [1, 1, 2, 2, 4, 4]
    
    # # deter.SetValueArray(array, count)
    # array = [1.0, 1.0, 2.0, 2.0, 4.0, 4.0]

    # count = 6
    
    # # value = deter.GetValue()
    # # print("Random Number Deterministic: ", value)
    
    # det_array = []
    # 
    # for i in range(count):
    #     o = 0
    #     for o in range(len(array)):
    #         det_array.append(array[o])
    
    # print("Random Number Deterministic: ", det_array)
    # # Sequential Random Variable
    # seq = ns.core.SequentialRandomVariable ()
    # seq.SetAttribute("Min", ns.core.DoubleValue(1))
    # seq.SetAttribute("Max", ns.core.DoubleValue(2))
    # seq.SetAttribute("Increment", ns.core.PointerValue(sz))
    # seq.SetAttribute("Consecutive", ns.core.IntegerValue(5))
    # print("Random Number Sequential: ",  seq.GetValue())
    
    # # Zeta Random Variable
    # zeta = ns.core.ZetaRandomVariable ()
    # zeta.SetAttribute("Alpha", ns.core.DoubleValue(0.5))
    # print("Random Number Zeta: ",  zeta.GetValue())

    # # # Zipf Random Variable
    # zipf = ns.core.ZipfRandomVariable ()
    # zipf.SetAttribute("N", ns.core.IntegerValue(1))
    # zipf.SetAttribute("Alpha", ns.core.DoubleValue(2))
    # print("Random Number zipf: ",  zipf.GetValue())
  
    # if const_Size == "True" and app_protocol == "tcp":
    #     ns.core.Config.SetDefault ("ns3::TcpSocket::SegmentSize", ns.core.UintegerValue (SegmentSize))


    
    # Habilita todas as notificações no NS3
    # ns.core.LogComponentEnableAll(ns.core.LOG_INFO)

    n_users = 5 # 255 is max
    # Criando container de nós 
    nodes = ns.network.NodeContainer()
    # Criando nós
    nodes.Create(n_users)
    
    if n_users <= 2:
        t_net = "p2p"
        # Definindo comunicação P2P
        p2p = ns.point_to_point.PointToPointHelper()
        # Setando taxa de dados 
        p2p.SetDeviceAttribute("DataRate", ns.core.StringValue("1Mbps"))
        # Setando atraso da comunicação
        p2p.SetChannelAttribute("Delay", ns.core.StringValue("2ms"))
        # Instalando configurações nos nós
        devices = p2p.Install(nodes)
    else:
        t_net = "csma"
        csma = ns.csma.CsmaHelper ()
        csma.SetChannelAttribute ("DataRate", ns.core.StringValue("5Mbps"))
        csma.SetChannelAttribute ("Delay", ns.core.StringValue("2ms"))
        # csma.SetDeviceAttribute ("Mtu", ns.core.UintegerValue(1400))
        devices = csma.Install (nodes)
	

    # Criando Intert Stack 
    stack = ns.internet.InternetStackHelper()
    stack.Install(nodes)
    # Definindo IP dos nós
    address = ns.internet.Ipv4AddressHelper()
    address.SetBase(ns.network.Ipv4Address("192.168.1.0"), ns.network.Ipv4Mask("255.255.255.0"))
    interfaces = address.Assign(devices)
    
    ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

    # Definindo taxa de erro
    # em = ns3.RateErrorModel()
    # em.SetRate(1e-5)
    # # Definindo taxa de erro por uma distribuição uniform
    # em.SetRandomVariable(ns.core.UniformRandomVariable())
    # # Instalando taxa de erro no nó 1
    # devices.Get(1).SetAttribute("ReceiveErrorModel", ns3.PointerValue(em))
    
    if (app_protocol == "http"):
        if validation == "True":
            timeStopSimulation, nRequestPackets, nResponsePackets = http_read(app_protocol)
            # Application request
            sinkPort = 8080
            # Serve Application
            packetSinkHelper = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), sinkPort))
            sinkApps = packetSinkHelper.Install(nodes.Get(0))
            
            sinkApps.Start(ns.core.Seconds(0.0))
            sinkApps.Stop(ns.core.Seconds(timeStopSimulation))
            # Client Application
            sinkAddress = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(0), sinkPort))
            ns3TcpSocket = ns.network.Socket.CreateSocket(nodes.Get(1), ns.internet.TcpSocketFactory.GetTypeId())
            app = ReqMyApp()
            # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
            # app.Setup(ns3TcpSocket, sinkAddress, packetSize, nRequestPackets, ns3.DataRate(dataRate))
            app.Setup(ns3TcpSocket, sinkAddress, nRequestPackets)
            nodes.Get(1).AddApplication(app)

            app.SetStartTime(ns.core.Seconds(0.0))
            app.SetStopTime(ns.core.Seconds(timeStopSimulation))

            ns.core.Simulator.Schedule(ns.core.Seconds(1), RequestCwndChange, app)

            # Application Response
            sinkPort1 = 8081
            # Serve Application
            packetSinkHelper1 = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), sinkPort1))
            sinkApps1 = packetSinkHelper1.Install(nodes.Get(1))
            
            sinkApps1.Start(ns.core.Seconds(0.001))
            sinkApps1.Stop(ns.core.Seconds(timeStopSimulation))
            # Client Application
            sinkAddress1 = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(1), sinkPort1))
            ns3TcpSocket1 = ns.network.Socket.CreateSocket(nodes.Get(0), ns.internet.TcpSocketFactory.GetTypeId())
            app1 = RespMyApp()
            # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
            # app.Setup(ns3TcpSocket, sinkAddress, packetSize, nRequestPackets, ns3.DataRate(dataRate))
            app1.Setup(ns3TcpSocket1, sinkAddress1, nResponsePackets)
            nodes.Get(0).AddApplication(app1)

            app1.SetStartTime(ns.core.Seconds(0.001))
            app1.SetStopTime(ns.core.Seconds(timeStopSimulation))

            ns.core.Simulator.Schedule(ns.core.Seconds(1), ResponseCwndChange, app1)
        else:
            http_read(app_protocol)

            ################# Application Request #################
            req_sinkPort = 8080
            ################# Request Server Application #################
            # req_sinkApps = np.empty(n_users-1)
            # req_sinkApps = [0 for x in range(n_users-1)] 
            # req_packetSinkHelper = np.empty(n_users-1)
            # req_packetSinkHelper = [0 for x in range(n_users-1)] 

            ################# Setting sinkApps #################
            # for i in range(0,len(req_sinkApps),1):
            req_packetSinkHelper = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), req_sinkPort))
            req_sinkApps = req_packetSinkHelper.Install(nodes.Get(0))
            req_sinkApps.Start(ns.core.Seconds(0.0))
            req_sinkApps.Stop(ns.core.Seconds(timeStopSimulation))
            ################# Request Client Application #################
            req_app = np.empty(n_users-1)
            req_app = [0 for x in range(n_users-1)]

            for i in range(0,len(req_app),1):
                req_sinkAddress = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(0), req_sinkPort))
                req_ns3TcpSocket = ns.network.Socket.CreateSocket(nodes.Get(i), ns.internet.TcpSocketFactory.GetTypeId())
                req_app[i] = ReqMyApp()
                req_app[i].Setup(req_ns3TcpSocket, req_sinkAddress, nRequestPackets)
                nodes.Get(i+1).AddApplication(req_app[i])
                req_app[i].SetStartTime(ns.core.Seconds(0.0))
                req_app[i].SetStopTime(ns.core.Seconds(timeStopSimulation))
                ns.core.Simulator.Schedule(ns.core.Seconds(1), RequestCwndChange, req_app[i])
            
            ################### Application Response #####################
            resp_sinkPort = 8081
            ################# Response Server Application #################
            resp_sinkApps = np.empty(n_users-1)
            resp_sinkApps = [0 for x in range(n_users-1)] 
            resp_packetSinkHelper = np.empty(n_users-1)
            resp_packetSinkHelper = [0 for x in range(n_users-1)] 
            
            ################# Setting sinkApps #################
            for i in range(0,len(resp_sinkApps),1):    
                resp_packetSinkHelper[i] = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), resp_sinkPort))
                resp_sinkApps[i] = resp_packetSinkHelper[i].Install(nodes.Get(i))
                resp_sinkApps[i].Start(ns.core.Seconds(0.01))
                resp_sinkApps[i].Stop(ns.core.Seconds(timeStopSimulation))
            
            ################# Response Client Application #################
            resp_app = np.empty(n_users-1)
            resp_app = [0 for x in range(n_users-1)] 
            for i in range(0,len(resp_app),1):
                resp_sinkAddress = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(i), resp_sinkPort))
                resp_ns3TcpSocket = ns.network.Socket.CreateSocket(nodes.Get(0), ns.internet.TcpSocketFactory.GetTypeId())
                resp_app[i] = RespMyApp()
                resp_app[i].Setup(resp_ns3TcpSocket, resp_sinkAddress, nResponsePackets)
                nodes.Get(i+1).AddApplication(resp_app[i])
                resp_app[i].SetStartTime(ns.core.Seconds(0.01))
                resp_app[i].SetStopTime(ns.core.Seconds(timeStopSimulation))
                ns.core.Simulator.Schedule(ns.core.Seconds(1), ResponseCwndChange, resp_app[i])

    if (app_protocol == "ftp"):
        if validation == "True":
            
            timeStopSimulation, timeStopReq_Resp ,nRequestPackets, nResponsePackets, nPackets = ftp_read(app_protocol)
            
            # Application request
            sinkPort0 = 8080
            # Serve Application
            packetSinkHelper0 = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), sinkPort0))
            sinkApps0 = packetSinkHelper0.Install(nodes.Get(0))
            
            sinkApps0.Start(ns.core.Seconds(0.0))
            sinkApps0.Stop(ns.core.Seconds(timeStopSimulation))
            # Client Application
            sinkAddress0 = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(0), sinkPort0))
            ns3TcpSocket0 = ns.network.Socket.CreateSocket(nodes.Get(1), ns.internet.TcpSocketFactory.GetTypeId())
            app0 = ReqMyApp()
            # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
            # app.Setup(ns3TcpSocket, sinkAddress, packetSize, nRequestPackets, ns3.DataRate(dataRate))
            app0.Setup(ns3TcpSocket0, sinkAddress0, nRequestPackets)
            nodes.Get(1).AddApplication(app0)

            app0.SetStartTime(ns.core.Seconds(0.0))
            app0.SetStopTime(ns.core.Seconds(timeStopSimulation))

            ns.core.Simulator.Schedule(ns.core.Seconds(1), RequestCwndChange, app0)

            # Application Response
            sinkPort1 = 8081
            # Serve Application
            packetSinkHelper1 = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), sinkPort1))
            sinkApps1 = packetSinkHelper1.Install(nodes.Get(1))
            
            sinkApps1.Start(ns.core.Seconds(0.01))
            sinkApps1.Stop(ns.core.Seconds(timeStopSimulation))
            # Client Application
            sinkAddress1 = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(1), sinkPort1))
            ns3TcpSocket1 = ns.network.Socket.CreateSocket(nodes.Get(0), ns.internet.TcpSocketFactory.GetTypeId())
            app1 = RespMyApp()
            # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
            # app.Setup(ns3TcpSocket, sinkAddress, packetSize, nRequestPackets, ns3.DataRate(dataRate))
            app1.Setup(ns3TcpSocket1, sinkAddress1, nResponsePackets)
            nodes.Get(0).AddApplication(app1)

            app1.SetStartTime(ns.core.Seconds(0.01))
            app1.SetStopTime(ns.core.Seconds(timeStopSimulation))

            ns.core.Simulator.Schedule(ns.core.Seconds(1), ResponseCwndChange, app1)

            
            # Application Send Files
            # 
            sinkPort = 8082
            # Serve Application
            packetSinkHelper = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), sinkPort))
            
            # packetSinkHelper.SetAttribute ("ns3::MaxBytes", ns.core.UintegerValue (0))
            
            
            sinkApps = packetSinkHelper.Install(nodes.Get(1))
            
            sinkApps.Start(ns.core.Seconds(0.02))
            sinkApps.Stop(ns.core.Seconds(timeStopSimulation))
            # Client Application
            sinkAddress = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(1), sinkPort))
            ns3TcpSocket = ns.network.Socket.CreateSocket(nodes.Get(0), ns.internet.TcpSocketFactory.GetTypeId())
            
            
            # ns.network.Packet.EnablePrinting ()
            # ns.network.Packet.EnableChecking ()
            # ns.core.Config.SetDefault ("ns3::TcpSocket::SegmentSize", ns.core.UintegerValue (SegmentSize))

            app = MyApp()
            # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
            # app.Setup(ns3TcpSocket, sinkAddress, packetSize, nRequestPackets, ns3.DataRate(dataRate))
            app.Setup(ns3TcpSocket, sinkAddress, nPackets)
            nodes.Get(0).AddApplication(app)

            app.SetStartTime(ns.core.Seconds(0.02))
            app.SetStopTime(ns.core.Seconds(timeStopSimulation))

            ns.core.Simulator.Schedule(ns.core.Seconds(1), CwndChange, app)
        else:
            
            timeStopRequest, timeStopResponse = ftp_read(app_protocol)
            
            ################# Application Request #################
            req_sinkPort = 8080
            ################# Request Server Application #################
            req_sinkApps = np.empty(n_users-1)
            req_sinkApps = [0 for x in range(n_users-1)] 
            req_packetSinkHelper = np.empty(n_users-1)
            req_packetSinkHelper = [0 for x in range(n_users-1)] 
            # app = np.empty(n_users-1)
            
            for i in range(0,len(req_sinkApps),1):
                req_packetSinkHelper = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), req_sinkPort))
                req_sinkApps = req_packetSinkHelper.Install(nodes.Get(i))
                req_sinkApps.Start(ns.core.Seconds(0.0))
                req_sinkApps.Stop(ns.core.Seconds(timeStopRequest))
            ################# Request Client Application #################
            req_app = np.empty(n_users-1)
            req_app = [0 for x in range(n_users-1)] 
            
            for i in range(0,len(req_app),1):
                req_sinkAddress = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(0), req_sinkPort))
                req_ns3TcpSocket = ns.network.Socket.CreateSocket(nodes.Get(i), ns.internet.TcpSocketFactory.GetTypeId())
                req_app[i] = ReqMyApp()
                # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
                # app.Setup(ns3TcpSocket, sinkAddress, packetSize, nRequestPackets, ns3.DataRate(dataRate))
                req_app[i].Setup(req_ns3TcpSocket, req_sinkAddress, nRequestPackets)
                nodes.Get(i+1).AddApplication(req_app[i])

                req_app[i].SetStartTime(ns.core.Seconds(0.0))
                req_app[i].SetStopTime(ns.core.Seconds(timeStopRequest))

                ns.core.Simulator.Schedule(ns.core.Seconds(1), RequestCwndChange, req_app[i])
            
            ################### Application Response #####################
            resp_sinkPort = 8081
            ################# Response Server Application #################
            resp_sinkApps = np.empty(n_users-1)
            resp_sinkApps = [0 for x in range(n_users-1)] 
            resp_packetSinkHelper = np.empty(n_users-1)
            resp_packetSinkHelper = [0 for x in range(n_users-1)] 
            # app = np.empty(n_users-1)
            
            for i in range(0,len(resp_sinkApps),1):
                resp_packetSinkHelper[i] = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), resp_sinkPort))
                resp_sinkApps[i] = resp_packetSinkHelper[i].Install(nodes.Get(i))
                resp_sinkApps[i].Start(ns.core.Seconds(timeStopRequest))
                resp_sinkApps[i].Stop(ns.core.Seconds(timeStopResponse))

            ################# Response Client Application #################
            resp_app = np.empty(n_users-1)
            resp_app = [0 for x in range(n_users-1)] 
            
            for i in range(0,len(resp_app),1):
                resp_sinkAddress = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(i), resp_sinkPort))
                resp_ns3TcpSocket = ns.network.Socket.CreateSocket(nodes.Get(0), ns.internet.TcpSocketFactory.GetTypeId())
                resp_app[i] = RespMyApp()
                # def Setup(self, socket, address, packetSize, nResponsePackets, dataRate):
                # app.Setup(ns3TcpSocket, sinkAddress, packetSize, nResponsePackets, ns3.DataRate(dataRate))
                resp_app[i].Setup(resp_ns3TcpSocket, resp_sinkAddress, nResponsePackets)
                nodes.Get(i+1).AddApplication(resp_app[i])

                resp_app[i].SetStartTime(ns.core.Seconds(timeStopRequest))
                resp_app[i].SetStopTime(ns.core.Seconds(timeStopResponse))

                ns.core.Simulator.Schedule(ns.core.Seconds(1), ResponseCwndChange, resp_app[i])

            ################# Application Send Files #################
            ################# Serve Application #################
            sinkPort = 8082
            sinkApps = np.empty(n_users-1)
            sinkApps = [0 for x in range(n_users-1)] 
            packetSinkHelper = np.empty(n_users-1)
            packetSinkHelper = [0 for x in range(n_users-1)] 
            # app = np.empty(n_users-1)
            
            for i in range(0,len(sinkApps),1):
                packetSinkHelper[i] = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), sinkPort))
                sinkApps[i] = packetSinkHelper[i].Install(nodes.Get(i))
                sinkApps[i].Start(ns.core.Seconds(timeStopRequest))
                sinkApps[i].Stop(ns.core.Seconds(timeStopSimulation))

            app = np.empty(n_users-1)
            app = [0 for x in range(n_users-1)] 
            
            for i in range(0,len(app),1):
                ################# Aplicação do cliente #################
                sinkAddress = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(i), sinkPort))
                ns3UdpSocket = ns.network.Socket.CreateSocket(nodes.Get(0), ns.internet.TcpSocketFactory.GetTypeId())
                # Definindo aplicação na classe Myapp
                app[i] = MyApp()
                
                # Chamando a função setup para configurar a aplicação
                # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
                app[i].Setup(ns3UdpSocket, sinkAddress, nPackets)
                # Configurando app no nó 0
                nodes.Get(i+1).AddApplication(app[i])
                # Inicio da aplicação
                app[i].SetStartTime(ns.core.Seconds(timeStopResponse))
                # Término da aplicação
                app[i].SetStopTime(ns.core.Seconds(timeStopSimulation))

    if (app_protocol == "tcp"):
        timeStopSimulation, nPackets = tcp_read(app_protocol)
        # Application UDP
        sinkPort = 8080
        # Aplicação do servidor
        packetSinkHelper = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), sinkPort))
        sinkApps = packetSinkHelper.Install(nodes.Get(1))
        sinkApps.Start(ns.core.Seconds(0.0))
        sinkApps.Stop(ns.core.Seconds(timeStopSimulation))
        # Aplicação do cliente
        sinkAddress = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(1), sinkPort))
        ns3UdpSocket = ns.network.Socket.CreateSocket(nodes.Get(0), ns.internet.TcpSocketFactory.GetTypeId())
        # Definindo aplicação na classe Myapp
        app = MyApp()
        
        # Chamando a função setup para configurar a aplicação
        # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
        app.Setup(ns3UdpSocket, sinkAddress, nPackets)
        # Configurando app no nó 0
        nodes.Get(0).AddApplication(app)
        # Inicio da aplicação
        app.SetStartTime(ns.core.Seconds(0.0))
        # Término da aplicação
        app.SetStopTime(ns.core.Seconds(timeStopSimulation))

    if (app_protocol == "udp"):

        if validation == "True":
            timeStopSimulation, nPackets = udp_read(app_protocol)
            # Application UDP
            sinkPort = 8080
            # Aplicação do servidor
            packetSinkHelper = ns.applications.PacketSinkHelper("ns3::UdpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), sinkPort))
            sinkApps = packetSinkHelper.Install(nodes.Get(1))
            sinkApps.Start(ns.core.Seconds(0.0))
            sinkApps.Stop(ns.core.Seconds(timeStopSimulation))
            # Aplicação do cliente
            sinkAddress = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(1), sinkPort))
            ns3UdpSocket = ns.network.Socket.CreateSocket(nodes.Get(0), ns.internet.UdpSocketFactory.GetTypeId())
            # Definindo aplicação na classe Myapp
            app = MyApp()
            
            # Chamando a função setup para configurar a aplicação
            # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
            app.Setup(ns3UdpSocket, sinkAddress, nPackets)
            # Configurando app no nó 0
            nodes.Get(0).AddApplication(app)
            # Inicio da aplicação
            app.SetStartTime(ns.core.Seconds(0.0))
            # Término da aplicação
            app.SetStopTime(ns.core.Seconds(timeStopSimulation))
        else:
            # Application UDP
            sinkPort = 8080
            # Aplicação do servidor
            sinkApps = np.empty(n_users-1)
            sinkApps = [0 for x in range(n_users-1)] 
            packetSinkHelper = np.empty(n_users-1)
            packetSinkHelper = [0 for x in range(n_users-1)] 
            # app = np.empty(n_users-1)
            
            for i in range(0,len(sinkApps),1):
                packetSinkHelper[i] = ns.applications.PacketSinkHelper("ns3::UdpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), sinkPort))
                sinkApps[i] = packetSinkHelper[i].Install(nodes.Get(i))
                sinkApps[i].Start(ns.core.Seconds(0.0))
                sinkApps[i].Stop(ns.core.Seconds(timeStopSimulation))

            app = np.empty(n_users-1)
            app = [0 for x in range(n_users-1)] 
            
            for i in range(0,len(app),1):
                # Aplicação do cliente
                sinkAddress = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(i), sinkPort))
                ns3UdpSocket = ns.network.Socket.CreateSocket(nodes.Get(0), ns.internet.UdpSocketFactory.GetTypeId())
                # Definindo aplicação na classe Myapp
                app[i] = MyApp()
                
                # Chamando a função setup para configurar a aplicação
                # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
                app[i].Setup(ns3UdpSocket, sinkAddress, nPackets)
                # Configurando app no nó 0
                nodes.Get(i+1).AddApplication(app[i])
                # Inicio da aplicação
                app[i].SetStartTime(ns.core.Seconds(0.0))
                # Término da aplicação
                app[i].SetStopTime(ns.core.Seconds(timeStopSimulation))
            ns.core.Simulator.Schedule(ns.core.Seconds(3), IncRate, app, ns3.DataRate(dataRate))

    if (app_protocol == "hls"):
        if validation == "True":
            
            timeStopSimulation, nRequestPackets, nResponsePackets, nPackets = hls_read(app_protocol)
            
            # Application request
            sinkPort0 = 8080
            # Serve Application
            packetSinkHelper0 = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), sinkPort0))
            sinkApps0 = packetSinkHelper0.Install(nodes.Get(0))
            
            sinkApps0.Start(ns.core.Seconds(0.0))
            # sinkApps0.Stop(ns.core.Seconds(timeStopSimulation))
            # Client Application
            sinkAddress0 = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(0), sinkPort0))
            ns3TcpSocket0 = ns.network.Socket.CreateSocket(nodes.Get(1), ns.internet.TcpSocketFactory.GetTypeId())
            app0 = ReqMyApp()
            # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
            # app.Setup(ns3TcpSocket, sinkAddress, packetSize, nRequestPackets, ns3.DataRate(dataRate))
            app0.Setup(ns3TcpSocket0, sinkAddress0, nRequestPackets)
            nodes.Get(1).AddApplication(app0)

            app0.SetStartTime(ns.core.Seconds(0.0))
            app0.SetStopTime(ns.core.Seconds(timeStopSimulation))

            ns.core.Simulator.Schedule(ns.core.Seconds(1), RequestCwndChange, app0)

            # Application Response
            sinkPort1 = 8081
            # Serve Application
            packetSinkHelper1 = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), sinkPort1))
            sinkApps1 = packetSinkHelper1.Install(nodes.Get(1))
            
            sinkApps1.Start(ns.core.Seconds(0.01))
            # sinkApps1.Stop(ns.core.Seconds(timeStopSimulation))
            # Client Application
            sinkAddress1 = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(1), sinkPort1))
            ns3TcpSocket1 = ns.network.Socket.CreateSocket(nodes.Get(0), ns.internet.TcpSocketFactory.GetTypeId())
            app1 = RespMyApp()
            # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
            # app.Setup(ns3TcpSocket, sinkAddress, packetSize, nRequestPackets, ns3.DataRate(dataRate))
            app1.Setup(ns3TcpSocket1, sinkAddress1, nResponsePackets)
            nodes.Get(0).AddApplication(app1)

            app1.SetStartTime(ns.core.Seconds(0.01))
            app1.SetStopTime(ns.core.Seconds(timeStopSimulation))

            ns.core.Simulator.Schedule(ns.core.Seconds(1), ResponseCwndChange, app1)

            
            # Application Send Files
            # 
            sinkPort = 8082
            # Serve Application
            packetSinkHelper = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), sinkPort))
            
            # packetSinkHelper.SetAttribute ("ns3::MaxBytes", ns.core.UintegerValue (0))
            
            
            sinkApps = packetSinkHelper.Install(nodes.Get(1))
            
            sinkApps.Start(ns.core.Seconds(0.02))
            # sinkApps.Stop(ns.core.Seconds(timeStopSimulation))
            # Client Application
            sinkAddress = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(1), sinkPort))
            ns3TcpSocket = ns.network.Socket.CreateSocket(nodes.Get(0), ns.internet.TcpSocketFactory.GetTypeId())
            
            
            # ns.network.Packet.EnablePrinting ()
            # ns.network.Packet.EnableChecking ()
            # ns.core.Config.SetDefault ("ns3::TcpSocket::SegmentSize", ns.core.UintegerValue (SegmentSize))

            app = MyApp()
            # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
            # app.Setup(ns3TcpSocket, sinkAddress, packetSize, nRequestPackets, ns3.DataRate(dataRate))
            app.Setup(ns3TcpSocket, sinkAddress, nPackets)
            nodes.Get(0).AddApplication(app)

            app.SetStartTime(ns.core.Seconds(0.02))
            app.SetStopTime(ns.core.Seconds(timeStopSimulation))

            ns.core.Simulator.Schedule(ns.core.Seconds(1), CwndChange, app)
        else:
            
            timeStopRequest, timeStopResponse = ftp_read(app_protocol)
            
            ################# Application Request #################
            req_sinkPort = 8080
            ################# Request Server Application #################
            req_sinkApps = np.empty(n_users-1)
            req_sinkApps = [0 for x in range(n_users-1)] 
            req_packetSinkHelper = np.empty(n_users-1)
            req_packetSinkHelper = [0 for x in range(n_users-1)] 
            # app = np.empty(n_users-1)
            
            for i in range(0,len(req_sinkApps),1):
                req_packetSinkHelper = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), req_sinkPort))
                req_sinkApps = req_packetSinkHelper.Install(nodes.Get(i))
                req_sinkApps.Start(ns.core.Seconds(0.0))
                # req_sinkApps.Stop(ns.core.Seconds(timeStopRequest))
            ################# Request Client Application #################
            req_app = np.empty(n_users-1)
            req_app = [0 for x in range(n_users-1)] 
            
            for i in range(0,len(req_app),1):
                req_sinkAddress = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(0), req_sinkPort))
                req_ns3TcpSocket = ns.network.Socket.CreateSocket(nodes.Get(i), ns.internet.TcpSocketFactory.GetTypeId())
                req_app[i] = ReqMyApp()
                # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
                # app.Setup(ns3TcpSocket, sinkAddress, packetSize, nRequestPackets, ns3.DataRate(dataRate))
                req_app[i].Setup(req_ns3TcpSocket, req_sinkAddress, nRequestPackets)
                nodes.Get(i+1).AddApplication(req_app[i])

                req_app[i].SetStartTime(ns.core.Seconds(0.0))
                # req_app[i].SetStopTime(ns.core.Seconds(timeStopRequest))

                ns.core.Simulator.Schedule(ns.core.Seconds(1), RequestCwndChange, req_app[i])
            
            ################### Application Response #####################
            resp_sinkPort = 8081
            ################# Response Server Application #################
            resp_sinkApps = np.empty(n_users-1)
            resp_sinkApps = [0 for x in range(n_users-1)] 
            resp_packetSinkHelper = np.empty(n_users-1)
            resp_packetSinkHelper = [0 for x in range(n_users-1)] 
            # app = np.empty(n_users-1)
            
            for i in range(0,len(resp_sinkApps),1):
                resp_packetSinkHelper[i] = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), resp_sinkPort))
                resp_sinkApps[i] = resp_packetSinkHelper[i].Install(nodes.Get(i))
                resp_sinkApps[i].Start(ns.core.Seconds(timeStopRequest))
                resp_sinkApps[i].Stop(ns.core.Seconds(timeStopResponse))

            ################# Response Client Application #################
            resp_app = np.empty(n_users-1)
            resp_app = [0 for x in range(n_users-1)] 
            
            for i in range(0,len(resp_app),1):
                resp_sinkAddress = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(i), resp_sinkPort))
                resp_ns3TcpSocket = ns.network.Socket.CreateSocket(nodes.Get(0), ns.internet.TcpSocketFactory.GetTypeId())
                resp_app[i] = RespMyApp()
                # def Setup(self, socket, address, packetSize, nResponsePackets, dataRate):
                # app.Setup(ns3TcpSocket, sinkAddress, packetSize, nResponsePackets, ns3.DataRate(dataRate))
                resp_app[i].Setup(resp_ns3TcpSocket, resp_sinkAddress, nResponsePackets)
                nodes.Get(i+1).AddApplication(resp_app[i])

                resp_app[i].SetStartTime(ns.core.Seconds(timeStopRequest))
                # resp_app[i].SetStopTime(ns.core.Seconds(timeStopResponse))

                ns.core.Simulator.Schedule(ns.core.Seconds(1), ResponseCwndChange, resp_app[i])

            ################# Application Send Files #################
            ################# Serve Application #################
            sinkPort = 8082
            sinkApps = np.empty(n_users-1)
            sinkApps = [0 for x in range(n_users-1)] 
            packetSinkHelper = np.empty(n_users-1)
            packetSinkHelper = [0 for x in range(n_users-1)] 
            # app = np.empty(n_users-1)
            
            for i in range(0,len(sinkApps),1):
                packetSinkHelper[i] = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory", ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), sinkPort))
                sinkApps[i] = packetSinkHelper[i].Install(nodes.Get(i))
                sinkApps[i].Start(ns.core.Seconds(timeStopRequest))
                sinkApps[i].Stop(ns.core.Seconds(timeStopSimulation))

            app = np.empty(n_users-1)
            app = [0 for x in range(n_users-1)] 
            
            for i in range(0,len(app),1):
                ################# Aplicação do cliente #################
                sinkAddress = ns.network.Address(ns.network.InetSocketAddress(interfaces.GetAddress(i), sinkPort))
                ns3UdpSocket = ns.network.Socket.CreateSocket(nodes.Get(0), ns.internet.TcpSocketFactory.GetTypeId())
                # Definindo aplicação na classe Myapp
                app[i] = MyApp()
                
                # Chamando a função setup para configurar a aplicação
                # def Setup(self, socket, address, packetSize, nRequestPackets, dataRate):
                app[i].Setup(ns3UdpSocket, sinkAddress, nPackets)
                # Configurando app no nó 0
                nodes.Get(i+1).AddApplication(app[i])
                # Inicio da aplicação
                app[i].SetStartTime(ns.core.Seconds(timeStopResponse))
                # Término da aplicação
                app[i].SetStopTime(ns.core.Seconds(timeStopSimulation))



    # Inicializando Flowmonitor
    flowmon_helper = ns3.FlowMonitorHelper()
    # Instalando Flowmonitor em todos os nós
    monitor = flowmon_helper.InstallAll()
    monitor.SetAttribute("DelayBinWidth", ns.core.DoubleValue(1e-3))
    monitor.SetAttribute("JitterBinWidth", ns.core.DoubleValue(1e-3))
    monitor.SetAttribute("PacketSizeBinWidth", ns.core.DoubleValue(20))
    monitor.SerializeToXmlFile ("myapp-py.xml", True, True)

    # Gerador de .pcap da rede
    if n_users <= 2:
        p2p.EnablePcapAll ("../../../Results/Traces/"+app_protocol+"_eth-myapp-py.pcap", True)
    else:
        csma.EnablePcapAll ("../../../Results/Traces/"+app_protocol+"_eth-myapp-py.pcap", True)

    # Controle de inicio e fim da simulação
    ns.core.Simulator.Stop(ns.core.Seconds(timeStopSimulation))
    ns.core.Simulator.Run()
    ns.core.Simulator.Destroy()

    # Chamando Flowmonitor para obter informações do fluxo
    monitor.GetAllProbes()
    monitor.CheckForLostPackets()
    classifier = flowmon_helper.GetClassifier()
 
    # Imprimir informações dos fluxos da rede
    flow_id = 0
    flow_stats = 0
    
    lost_packets = []
    throughput = []
    delay = []
    jitter = []
    rate_type = ""

    for flow_id, flow_stats in monitor.GetFlowStats():
        t = classifier.FindFlow(flow_id)
        proto = {6: 'TCP', 17: 'UDP'} [t.protocol]
        print_stats(sys.stdout, flow_stats, flow_id, proto, t, lost_packets, throughput, delay, jitter, rate_type)
    
    m_lost_packets = np.mean(lost_packets)
    m_throughput = np.mean(throughput)
    m_jitter = np.mean(jitter)
    m_delay = np.mean(delay)

    # print("Mean throughput", m_throughput)
    # print("Mean jitter", m_jitter)
    # print("Mean delay", m_delay)

    if run <= 1:
        w = open("../../../Results/Prints/valid-"+validation+"_"+t_net+"_"+app_protocol+"_qos_"+mt_RG+"_"+str(IC)+".txt", "w")
        w.write('"n_Run";"Lost_Packets";"Throughput";"Delay";"Jitter"\n')
        
    else:
        w = open("../../../Results/Prints/valid-"+validation+"_"+t_net+"_"+app_protocol+"_qos_"+mt_RG+"_"+str(IC)+".txt", "a")
    
    w.write('"'+str(run) + '";"' + str(m_lost_packets) + '";"' + str(m_throughput) + '";"' + str(m_delay) + '";"' + str(m_jitter)+'"\n')
    w.close()
    
    if run == 1 and validation == "False":
        ecdf_df = pd.read_csv("../../../Results/Prints/valid-"+validation+"_"+t_net+"_"+app_protocol+"_qos_ecdf_"+str(IC)+".txt", sep=";")
        tcdf_df = pd.read_csv("../../../Results/Prints/valid-"+validation+"_"+t_net+"_"+app_protocol+"_qos_tcdf_"+str(IC)+".txt", sep=";")
        pd_df = pd.read_csv("../../../Results/Prints/valid-"+validation+"_"+t_net+"_"+app_protocol+"_qos_PD_"+str(IC)+".txt", sep=";")
        # print("PD: ")
        # print (pd_df)
        # print("ECDF: ")
        # print (ecdf_df)
        # print("TCDF: ")
        # print (tcdf_df)

        Lost_Packets = []
        Lost_Packets.append(tcdf_df['Lost_Packets'].mean())
        Lost_Packets.append(ecdf_df['Lost_Packets'].mean())
        Lost_Packets.append(pd_df['Lost_Packets'].mean())

        Throughput = []
        Throughput.append(tcdf_df['Throughput'].mean())
        Throughput.append(ecdf_df['Throughput'].mean())
        Throughput.append(pd_df['Throughput'].mean())

        Delay = []
        Delay.append(tcdf_df['Delay'].mean())
        Delay.append(ecdf_df['Delay'].mean())
        Delay.append(pd_df['Delay'].mean())

        Jitter = []
        Jitter.append(tcdf_df['Jitter'].mean())
        Jitter.append(ecdf_df['Jitter'].mean())
        Jitter.append(pd_df['Jitter'].mean())

        Lost_Packets = np.around(Lost_Packets,  2)
        Throughput = np.around(Throughput,  2)
        Delay = np.around(Delay,  2)
        Jitter = np.around(Jitter,  2)

        labels = ["PD","ecdf","tcdf"]

        # Inicio do plot do QoS
        x = np.arange(len(labels))
        width = 0.25  # the width of the bars
        fig, ax = plt.subplots()

        # rects1 = ax.bar(x - width/2, men_means, width, label='Men')
        rects0 = ax.bar(x + 0.00, Lost_Packets, width = width, label='Lost Packets', edgecolor='none', align='center',
                fill=True, facecolor="blue", zorder=1, alpha=0.5)
        rects1 = ax.bar(x + 0.25, Throughput, width = width, label='Throughput (KBps)', edgecolor='none', align='center',
                fill=True, facecolor="blue", zorder=1, alpha=0.5)
        rects2 = ax.bar(x + 0.50, Delay, width = width, label='Delay', edgecolor='none', align='center',
                fill=True, facecolor="green", zorder=1, alpha=0.5)
        rects3 = ax.bar(x + 0.75, Jitter, width = width, label='Jitter', edgecolor='none', align='center',
                fill=True, facecolor="red", zorder=1, alpha=0.5)
        ax.set_ylabel('QoS Parameters')
        ax.set_xlabel('Methods of Random Generate')
        ax.set_title('Compare methods random generate')
        ax.set_xticks([0.25, 1+0.25, 2+0.25])
        ax.set_xticklabels(labels)

        max_value = np.ceil(max(max(Delay),max(Lost_Packets),max(Throughput),max(Jitter)))
        ax.set_ylim(0,max_value+0.5)
        ax.legend()

        autolabel(rects0, ax)
        autolabel(rects1, ax)
        autolabel(rects2, ax)
        autolabel(rects3, ax)

        fig.tight_layout()

        if plot == "show":
            plt.show()  
        if plot == "save":
            plt.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_qos", fmt="png", dpi=1000)
            plt.close()

    if (mt_RG == "tcdf" or mt_RG == "ecdf") and validation == "True":
        # os.system("cd ../../../FWGNet/")
        os.system("sudo  chmod 777 ../../../FWGNet/run-pos.sh")
        os.system("sudo bash ./../../../FWGNet/run-pos.sh "+app_protocol)
        compare(app_protocol)


if __name__ == '__main__':
    main(sys.argv)