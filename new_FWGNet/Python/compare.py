# coding: utf-8

import sys
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


def ksvalid(size, Dobs, IC):
    # Definir intervalo de confiança
    # IC = 99.90 -> alpha = 0.10
    # IC = 99.95 -> alpha = 0.05
    # IC = 99.975 -> alpha = 0.025
    # IC = 99.99 -> alpha = 0.01
    # IC = 99.995 -> alpha = 0.005
    # IC = 99.999 -> alpha = 0.001
    
    
    D_critico = 0
    rejects = ""
    
    str_IC = str(IC)+"%"
    
    if (size<=35):
        ks_df = pd.read_csv("/home/carl/New_Results/Files/kstest.txt", sep=";")
        # print(ks_df)
        D_critico = ks_df[""+str_IC+""].iloc[size-1]
        
    else:
        # Condição para definir o D_critico de acordo com o tamanho dos dados
        if str_IC == "99.80%":
            D_critico = 1.07/np.sqrt(size)
        if str_IC == "99.85%":
            D_critico = 1.14/np.sqrt(size)
        if str_IC == "90.0%":
            D_critico = 1.224/np.sqrt(size)
        if str_IC == "95.0%":
            D_critico = 1.358/np.sqrt(size)
        if str_IC == "99.0%":
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
    str_IC = str_IC.replace('%', '')
    # print("IC: ", IC)
    return rejects, D_critico


def plot_histogram(y, save_Graph, parameter, case_Study, proto):
    if save_Graph == True:
        fig, ax = plt.subplots(1, 1)
        ax = sns.distplot(y)
        plt.title("Histogram of flow "+proto+" ("+parameter+")")
        fig.savefig("/home/carl/New_Results/Graphics/"+case_Study+"_histogram_"+proto+"_"+parameter, fmt="png",dpi=1000)
        plt.close()


def compare(ns3_arr_protocols, case_Study, const_Size, save_Graph, IC):
    ks_first = True
    for proto in ns3_arr_protocols:
        # time_ns3_df = pd.read_csv('/home/carl/New_Results/Files/ns3_'+proto+'_time.txt', sep="",  names=["Time"])
        time_ns3 = np.loadtxt("/home/carl/New_Results/Files/ns3_"+proto+"_time.txt", usecols=0)
        time_ns3_df = pd.DataFrame(data=time_ns3,columns=["Time"])
        # print("NS3", proto)
        # print(time_ns3_df.describe())
        # time_ns3 = np.array(time_ns3_df['Time'])
        time_ns3 = time_ns3.astype(float)

        # size_ns3_df = pd.read_csv('/home/carl/New_Results/Files/ns3_'+proto+'_size.txt', sep="",  names=["Size"])
        size_ns3 = np.loadtxt("/home/carl/New_Results/Files/ns3_"+proto+"_size.txt", usecols=0)
        # print(ns3_df.describe())
        # size_ns3 = np.array(size_ns3_df['Size'])
        size_ns3 = size_ns3.astype(float)           



        time_trace = np.loadtxt("/home/carl/New_Results/Files/"+proto+"_time.txt", usecols=0)
        time_trace_df = pd.DataFrame(data=time_trace,columns=["Time"])
        # print("TRACE", proto)
        # print(time_trace_df.describe())
        # time_trace = np.array(time_trace['Time'])
        time_trace = time_trace.astype(float)

        size_trace = np.loadtxt("/home/carl/New_Results/Files/"+proto+"_size.txt", usecols=0)
        # print(trace.describe())
        # size_trace = np.array(size_trace['Size'])
        size_trace = size_trace.astype(float)   
    
        # Definindo o parametro da rede a ser comparado
        if const_Size == "False":
            Parameters = ["Size", "Time"]
        else:
            Parameters = ["Time"]
        
        Methods = ["kstest"]
        # Methods = ["qq_e_pp","kstest","graphical"]

        # ks_first = True
        
        for meth in Methods:
            for parameter in Parameters:
                
                if parameter == "Size":
                    # Adicionando valores gerados pelo NS3
                    x = size_ns3
                    # x = data_size_ns3
                    # Adicionando valores do trace
    
                    y = size_trace
                    # y = data_size
                if parameter == "Time":
                    # Adicionando valores gerados pelo NS3
                    x = time_ns3
                    # x = data_size_ns3
                    # Adicionando valores do trace
                    y = time_trace
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

                # if meth == "graphical":
                #     x_gr = x
                #     y_gr = y
                #     ######################################################## x = empirico -> ns3
                #     ######################################################## Gerar ECDF
                #     ######################################################## y = função real -> trace real
                #     ######################################################## Normalizar Função
                #     Fe = [] 
                    
                #     # Criando ECDFs
                    
                #     for i in range(1, len(x_gr)+1):
                #         # ecdf i/(n+1)
                #         Fe.append(i/(len(x_gr)))
                
                    
                #     # Trandformando vetorem em np.arrays()
                #     # x_gr = np.array(Fe)
                #     x_gr_norm = []
                #     for i in range(0, len(x_gr)):
                #         x_gr_norm.append(x_gr[i]/np.sum(x_gr))
                #     x_gr = np.cumsum(x_gr)
                    
                    
                #     y_gr_norm = []
                #     for i in range(0, len(y_gr)):
                #         y_gr_norm.append(y_gr[i]/np.sum(y_gr))
                #     y_gr = np.cumsum(y_gr)
                    
                #         # print("Média: ",np.mean(y_gr))
                #         # print("Atual: ",y_gr[i])
                #         # print("Atual Norm: ",y_gr_norm[i])

                #     y_gr = np.array(y_gr_norm)
                #     x_gr = np.array(x_gr_norm)
                #     # Ordenando os valores
                #     x_gr.sort()
                #     y_gr.sort()

                #     # print("X: ", x_gr)
                #     # print("Y: ", y_gr)
                #     # Tornando os vetores do mesmo tamanho
                #     if len(x_gr) > len(y_gr):
                #         x_gr = x_gr[0:len(y_gr)]
                #     if len(x_gr) < len(y_gr):
                #         y_gr = y_gr[0:len(x_gr)]
                    
                #     # print("X size: ", len(x))
                #     # print("Y size: ", len(y))
                #     # print("X: ", x)
                #     # print("Y: ", y)

                #     # Plotando dados x e y
                #     plt.plot(x_gr,y_gr,"o")

                #     # Definindo polinomial de x e y
                #     z = np.polyfit(x_gr, y_gr, 1)

                #     # Gerando polinomial de 1d com os dados de z e x 
                #     y_hat = np.poly1d(z)(x_gr)

                #     # Plotando linha tracejada 
                #     plt.plot(x_gr, y_hat, "r--", lw=1)

                #     # Imprimindo resultados da regressão linear dos dados comparados
                #     text = f"$y={z[0]:0.6f}x{z[1]:+0.6f}$\n$R^2 = {r2_score(y_gr,y_hat):0.6f}$"
                #     plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
                #         fontsize=12, verticalalignment='top')
                #     # Definindo titulo do gráfico
                #     plt.title('Graphical Method Inference for Real and Simulated Trace '+app+' ('+parameter+')')
                #     plt.xlabel('ECDF of Real Trace Data')
                #     plt.ylabel('Normalize Data of Simulated Trace')
                #     if plot == "show":
                #         plt.show()  
                #     if plot == "save":
                #         plt.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_"+app+"_"+meth+"_plot_"+parameter, fmt="png",dpi=1000)
                #         plt.close()
                if meth == "kstest":  

                    x_ks = x
                    y_ks = y
                                        
                                        
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
                    

                    #
                    # KS TEST
                    #
                    # Criando percentil
                    size = len(y)
                    # percentile = np.linspace(0,100,size)
                    # percentile_cut = np.percentile(y, percentile)
                    
                    # # Criando CDF da teórica
                    # Ft = dist.cdf(percentile_cut, *param[:-2], loc=param[-2], scale=param[-1])
                    
                    
                    # # Criando CDF Inversa 
                    # Ft_ = dist.ppf(percentile_cut, *param[:-2], loc=param[-2], scale=param[-1])
                    
                    # Adicionando dados do trace
                    # t_Fe = y

                    t_Fe = sorted(t_Fe, reverse=True)
                    Ft = sorted(Ft, reverse=True)
                    
                    # Definindo mesmo tamanho para os vetores
                    if len(Ft) > len(t_Fe):
                        Ft = Ft[0:len(t_Fe)]
                    if len(Ft) < len(t_Fe):
                        t_Fe = t_Fe[0:len(Ft)]
                
                    # Ordenando dados
                    t_Fe.sort()
                    Ft.sort()
                    # Ft_.sort()

                    # Criando listas para armazenar as ECDFs
                    Fe = []
                    Fe_ = []
                    
                    # ecdf = np.array([12.0, 15.2, 19.3])  
                    arr_ecdf = np.empty(size)
                    arr_ecdf = [id_+1 for id_ in range(size)]
                    arr_div = np.array(size) 
                    
                    # Criando ECDFs
                    # for i in range(0, size):
                        # ecdf i-1/n
                    Fe = (np.true_divide(arr_ecdf, size))
                
                    arr_ecdf = np.subtract(arr_ecdf, 1)

                    Fe_ = (np.true_divide(arr_ecdf, size))
                    
                    # Transformando listas em np.arrays()
                    Fe = np.array(Fe)
                    Fe_ = np.array(Fe_)
                    Ft = np.array(Ft)
                    # Ft_ = np.array(Ft_)
                    
                    # Inicio cálculo de rejeição
                    #
                    # Ft(t)-FE-(i),FE+(i)-Ft(t)
                    Ft_Fe_ = abs(np.subtract(Fe_, Ft))
                    Fe_Ft = (np.subtract(Fe, Ft))
                    
                    # Max(Ft(t)-FE-(i),FE+(i)-Ft(t))
                    Dobs_max = np.maximum(Ft_Fe_, Fe_Ft)
                    
                    # Dobs= Max(Max (Ft(t)-FE-(i),FE+(i)-Ft(t)))
                    Dobs = np.max(Dobs_max)
                    #
                    # Fim cálculo de rejeição

                    # ks_statistic, D_critico = stats.ks_2samp(Ft,t_Fe)
                
                    rejects, D_critico = ksvalid(size, Dobs, IC)
                    # rejects, IC, D_critico = ksvalid(size, Dobs)

                    if ks_first == True:
                        w = open("/home/carl/New_Results/Files/compare_results_"+parameter+".txt", "w")
                        w.write('"flow_Trace";"KS-Test";"Rejection"\n')
                        ks_first = False
                    else:
                        w = open("/home/carl/New_Results/Files/compare_results_"+parameter+".txt", "a")
                        w.write(''+str(proto) + ' ' + str(Dobs) + ' ' + str(rejects) + '\n')
                    w.close()

                    # Plotando resultados do teste KS
                    if save_Graph == True:
                        plt.plot(Ft, Fe_, 'o', label='CDF Trace Distribution')
                        plt.plot(t_Fe, Fe_, 'o', label='CDF NS3 Distribution')
                        
                        plt.xlabel('Time (s)')
                        plt.ylabel('CDF')
                        
                        # plt.plot(Ft, Fe, 'o', label='Teorical Distribution')
                        # plt.plot(t_Fe, Fe, 'o', label='Empirical Distribution')
                        
                        # Definindo titulo
                        plt.title("KS Test of Real and Simulated Trace of "+proto+ "(" + parameter + ")")
                        plt.legend()
                        
                        plt.savefig("/home/carl/New_Results/Graphics/"+case_Study+"_compare_ks_test_"+proto+"_("+parameter+")", fmt="png",dpi=1000)
                        plt.close()
                        
def read_filter(const_Size, type_Size, save_Graph, case_Study):
                                                      
 
    ns3_ip_df = pd.read_csv("/home/carl/New_Results/Files/ns3_ip.txt", sep=";", names=["protocols","ip_SRC"])
    # print(ns3_ip_df)

    ns3_df = pd.read_csv("/home/carl/New_Results/Filter_Traces/ns3_"+case_Study+"_trace.txt", sep=";", names=["ip_SRC","ip_DST","time","size","protocols","tcp_Size","udp_Size"])
    
    ns3_df = ns3_df[ns3_df.tcp_Size != 0]
    ns3_df = ns3_df[ns3_df.udp_Size != 0]

    ns3_df = ns3_df.fillna(0)
    
    ns3_df['protocols'] = ns3_df.protocols.str.replace('ppp:ip:', '')

    for index, row in ns3_df.iterrows():
        for index_ip, row_ip in ns3_ip_df.iterrows():
            if row['ip_SRC'] == row_ip['ip_SRC']:
                ns3_df['protocols'][index] = row_ip['protocols']
   

    
    ns3_arr_protocols = list(set(ns3_df["protocols"]))
    ns3_arr_protocols = np.array(ns3_arr_protocols)
    
  

    # Cria valores só pra um valor da coluna 
    for ns3_proto in ns3_arr_protocols:
        
        # trace_df[] = trace_df.loc[trace_df['protocols'] == str(eth:ethertype:ip:tcp)]
        ns3_data_df = ns3_df[ns3_df['protocols'].str.contains(ns3_proto)]
        # print(ns3_data_df)
        if len(ns3_data_df.index) > 2:
            
            ns3_ip_proto_df = ns3_df.drop(['time', 'size', 'tcp_Size', 'udp_Size'], axis=1)

            ns3_dst_proto_df = ns3_ip_proto_df.drop(['ip_SRC'], axis=1)
            ns3_src_proto_df = ns3_ip_proto_df.drop(['ip_DST'], axis=1)

            ns3_dst_proto_df = ns3_dst_proto_df.drop_duplicates()
            ns3_src_proto_df = ns3_src_proto_df.drop_duplicates()
            

            ######## Definindo Tempos ######

            ns3_t_Time = np.array(ns3_data_df["time"])
            ns3_t_Time.sort()
            
            ns3_sub = []
          
            for i in range(0,len(ns3_t_Time)-1):
                ns3_sub.append(ns3_t_Time[i+1] - ns3_t_Time[i])
            
            # Passando valores resultantes para a variável padrão t_time
            ns3_t_Time = np.array(ns3_sub)
            ns3_t_Time = ns3_t_Time.astype(float)

            

            ns3_t_Time = np.delete(ns3_t_Time, np.where(ns3_t_Time == 0))
            ns3_t_Time.sort()

            # Plot histograma t_time:
            plot_histogram(ns3_t_Time, save_Graph, "time", case_Study, ns3_proto)

            np.savetxt('/home/carl/New_Results/Files/ns3_'+ns3_proto+'_time.txt', ns3_t_Time, delimiter=',', fmt='%f')
            

            ############ Definindo Tamanhos #########

            if const_Size == False:
                # Plot histograma t_time:
                plot_histogram(ns3_data_df["size"], save_Graph, "size", case_Study, ns3_proto)
                np.savetxt('/home/carl/New_Results/Files/ns3_'+ns3_proto+'_size.txt', ns3_data_df["size"], delimiter=',', fmt='%f')
            else:
                if type_Size == "mean_Trace":
                    ns3_size = np.mean(ns3_data_df["size"])
                if type_Size == "const_Value":
                    ns3_size = 500
                
                ns3_arr_Size = np.empty(len(ns3_data_df["size"])-1)
                ns3_arr_Size = [ns3_size for x in range(len(ns3_data_df["size"]))]
                
                np.savetxt('/home/carl/New_Results/Files/ns3_'+ns3_proto+'_size.txt', ns3_arr_Size, delimiter=',', fmt='%f')
            
            

            ns3_df = ns3_df[ns3_df.protocols != ns3_proto]
        else:
            ns3_arr_protocols = np.delete(ns3_arr_protocols, np.where(ns3_arr_protocols == ns3_proto))
          





    trace_df = pd.read_csv("/home/carl/New_Results/Filter_Traces/"+case_Study+"_trace.txt", sep=";", names=["ip_SRC","ip_DST","time","size","protocols","tcp_Size","udp_Size"])
    
    trace_df = trace_df[trace_df.tcp_Size != 0]
    trace_df = trace_df[trace_df.udp_Size != 0]

    trace_df = trace_df.fillna(0)

    # print(trace_df)
    
    trace_df['protocols'] = trace_df.protocols.str.replace('ethertype:ip:', '')
    trace_df['protocols'] = trace_df.protocols.str.replace('ethertype:ipv6:', '')
    trace_df['protocols'] = trace_df.protocols.str.replace('wlan_radio:wlan:', '')
    
    
    arr_protocols = list(set(trace_df["protocols"]))
    arr_protocols = np.array(arr_protocols)


    # Cria valores só pra um valor da coluna 
    for proto in arr_protocols:
        
        # trace_df[] = trace_df.loc[trace_df['protocols'] == str(eth:ethertype:ip:tcp)]
        data_df = trace_df[trace_df['protocols'].str.contains(proto)]
        # print(data_df)
        if len(data_df.index) > 2:
            
            ip_proto_df = trace_df.drop(['time', 'size', 'tcp_Size', 'udp_Size'], axis=1)

            dst_proto_df = ip_proto_df.drop(['ip_SRC'], axis=1)
            src_proto_df = ip_proto_df.drop(['ip_DST'], axis=1)

            dst_proto_df = dst_proto_df.drop_duplicates()
            src_proto_df = src_proto_df.drop_duplicates()
            

            ######## Definindo Tempos ######

            t_Time = np.array(data_df["time"])
            t_Time.sort()
            # print(t_Time)
            sub = []
          
            for i in range(0,len(t_Time)-1):
                sub.append(t_Time[i+1] - t_Time[i])
            
            # Passando valores resultantes para a variável padrão t_time
            t_Time = np.array(sub)
            t_Time = t_Time.astype(float)
            # print(t_Time)
            

            t_Time = np.delete(t_Time, np.where(t_Time == 0))
            t_Time.sort()

            # Plot histograma t_time:
            plot_histogram(t_Time, save_Graph, "time", case_Study, proto)

            np.savetxt('/home/carl/New_Results/Files/'+proto+'_time.txt', t_Time, delimiter=',', fmt='%f')
            

            ############ Definindo Tamanhos #########

            if const_Size == False:
                # Plot histograma t_time:
                plot_histogram(data_df["size"], save_Graph, "size", case_Study, proto)
                np.savetxt('/home/carl/New_Results/Files/'+proto+'_size.txt', data_df["size"], delimiter=',', fmt='%f')
            else:
                if type_Size == "mean_Trace":
                    size = np.mean(data_df["size"])
                if type_Size == "const_Value":
                    size = 500
                
                arr_Size = np.empty(len(data_df["size"])-1)
                arr_Size = [size for x in range(len(data_df["size"]))]
                
                np.savetxt('/home/carl/New_Results/Files/'+proto+'_size.txt', arr_Size, delimiter=',', fmt='%f')
            
            

            trace_df = trace_df[trace_df.protocols != proto]
        else:
            arr_protocols = np.delete(arr_protocols, np.where(arr_protocols == proto))
          

    return arr_protocols, ns3_arr_protocols

# Função principal do código
def main(argv):

    # Determina se os tamanhos de pacotes serão constantes caso True ou se seguirão o padrão do trace caso False
    const_Size = sys.argv[1]
    # Tipo de tamanho de pacote: "const_Value"(Valor específico) | "mean_Trace"(Usa a média do tamanho dos pacotes do trace)
    type_Size = sys.argv[2]
    save_Graph = sys.argv[3]
    
    # parameters = ["size", "time"]
    case_Study = sys.argv[4]
    # "99.80%";"99.85%";"99.90%";"99.95%";"99.99%"
    IC = sys.argv[5]
    IC = float(IC)

    save_Graph = eval(save_Graph)
    const_Size = eval(const_Size)
    # Chamada da função de filtro do trace e criação de arquivos com os parametros da rede
    arr_protocols, ns3_arr_protocols = read_filter(const_Size, type_Size, save_Graph, case_Study)
    
    compare(ns3_arr_protocols, case_Study, const_Size, save_Graph, IC)

if __name__ == '__main__':
    main(sys.argv)