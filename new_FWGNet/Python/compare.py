# coding: utf-8


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

# Função de comparação dos resultados obtidos com o NS3 com os dados dos traces
# Esta função é utilizada apenas quando o método de geração variáveis aleatórias selecionado é por "Trace"
def compare(app_protocol, n_users):
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
        ns3_df = pd.read_csv("scratch/compare_"+app_protocol+".txt", sep=";",  names=["Time", "SRC", "DST","Size"])
        
        ns3_df = ns3_df[ns3_df.Size != 0]
        # print(ns3_df[:20])

        grouped_src = ns3_df.groupby(ns3_df.SRC)
        for i in range(1, n_users):
            req_df = grouped_src.get_group("192.168.1."+str(i))
            
            time_req_ns3[i] = req_df.loc[df["SRC"] == "192.168.1."+str(i), 'Time']

            time_req_ns3 = np.array(req_df['Time'])



        grouped_dst = ns3_df.groupby(ns3_df.DST)
        for i in range(1, n_users):
            resp_df = grouped_dst.get_group("192.168.1."+str(i))
            time_resp_ns3[i] = np.array(resp_df[i]['Time'])
        
        print(req_df)
        print(resp_df)
        
        # req_df = grouped.get_group(8080)
        # resp_df = grouped.get_group(8081)
        # resp_df = grouped.get_group(49153)
        # req_df = grouped.get_group(49153)

        # print("Client: \n",client_df[:10])
        # print("Server: \n",server_df[:10])

        # time_resp_ns3 = np.array(resp_df['Time'])
        # time_req_ns3 = np.array(req_df['Time'])

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
                    


# Função principal do código
def main(argv):
    compare()
if __name__ == '__main__':
    main(sys.argv)