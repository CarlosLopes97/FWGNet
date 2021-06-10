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


# Função que valida dados por meio do método de Goodness of Fitness KS-Test
def ksvalid(size, Dobs, IC):
    # Definir intervalo de confiança
    # IC = 99.90 -> alpha = 0.10
    # IC = 99.95 -> alpha = 0.05
    # IC = 99.975 -> alpha = 0.025
    # IC = 99.99 -> alpha = 0.01
    # IC = 99.995 -> alpha = 0.005
    # IC = 99.999 -> alpha = 0.001
    
    # Defininfo D_critico
    D_critico = 0
    # Definindo variável para informar a rejeição dos dados
    rejects = ""
    # Defininfo string compatível com o arquivo "kstest.txt"
    str_IC = str(IC)+"%"
    
    # Se o tamanho dos dados for menor ou igual a 35, deve-se utilizar o padrão existente no arquivo "kstest.txt"
    if (size<=35):
        # Chamando dados do arquivo
        ks_df = pd.read_csv("/home/carl/New_Results/Files/kstest.txt", sep=";")
        # Atribruindo Dcritico do arquivo à variável
        D_critico = ks_df[""+str_IC+""].iloc[size-1]
    # Se o tamanho so arquivo for maior que 35, utiliza-se uma equação para cada IC
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
    # Definindo apenas duas casa decimais para D_critico
    D_critico = np.around(D_critico, 2)

    # Condição para aceitar a hipótese nula do teste KS
    if Dobs > D_critico:
        rejects = "Reject the Null Hypothesis"
    else:
        rejects = "Fails to Reject the Null Hypothesis"
    
    # Apagando "%" de IC para evitar conflitos em múltimas leituras
    str_IC = str_IC.replace('%', '')
    # Retornando valor de D_critico e string que informa a rejeição do KS-Test
    return rejects, D_critico


# Função que realiza o plot dos histogramas
def plot_histogram(y, save_Graph, parameter, case_Study, proto):
    # Caso a variável save_Graph seja verdadeira, salvar histograma 
    if save_Graph == True:
        # Criando axis
        fig, ax = plt.subplots(1, 1)
        # Criando histograma pelo sklearn
        ax = sns.distplot(y)
        # Definindo título
        plt.title("Histogram of flow "+proto+" ("+parameter+")")
        # Definindo nome e local do arquivo
        fig.savefig("/home/carl/New_Results/Graphics/"+case_Study+"_histogram_"+proto+"_"+parameter, fmt="png",dpi=1000)
        # Fechando arquivo de plot
        plt.close()

# 
def compare(ns3_arr_protocols, case_Study, const_Size, save_Graph, IC):
    # Definindo a primeira entrada do teste
    ks_first = True
    # Percorre cada protocolo
    for proto in ns3_arr_protocols:
        
        # Adiciona à variável os tempos da simulação
        time_ns3 = np.loadtxt("/home/carl/New_Results/Files/ns3_"+proto+"_time.txt", usecols=0)
        # Adiciona os tempos a um dataframe
        time_ns3_df = pd.DataFrame(data=time_ns3,columns=["Time"])
        
        # Apresenta métricas básicas do dataframe
        # print(time_ns3_df.describe())

        # Convertendo valores para float
        time_ns3 = time_ns3.astype(float)

        # Adiciona à variável os tamanhos de pacotes da simulação
        size_ns3 = np.loadtxt("/home/carl/New_Results/Files/ns3_"+proto+"_size.txt", usecols=0)
        
        # Adiciona os tamanhos de pacotes a um dataframe
        size_ns3_df = pd.DataFrame(data=size_ns3,columns=["Size"])
        
        # Apresenta métricas básicas do dataframe
        # print(time_ns3_df.describe())

        # Convertendo valores para int
        size_ns3 = size_ns3.astype(int)           


        # Adiciona à variável os tempos do trace real
        time_trace = np.loadtxt("/home/carl/New_Results/Files/"+proto+"_time.txt", usecols=0)
       
        # Adiciona os tempos a um dataframe
        time_trace_df = pd.DataFrame(data=time_trace,columns=["Time"])
        
        # Apresenta métricas básicas do dataframe
        # print(time_trace_df.describe())
        
        # Convertendo valores para float
        time_trace = time_trace.astype(float)


        # Adiciona à variável os tamanhos de pacotes do trace real
        size_trace = np.loadtxt("/home/carl/New_Results/Files/"+proto+"_size.txt", usecols=0)
        
        # Adiciona os tamanhos de pacotes a um dataframe
        time_trace_df = pd.DataFrame(data=time_trace,columns=["Size"])

        # Apresenta métricas básicas do dataframe
        # print(time_trace_df.describe())
        
        # Convertendo valores para float
        size_trace = size_trace.astype(float)   
    
        
        # Definindo o parametro da rede a ser comparado
        if const_Size == "False":
            Parameters = ["Size", "Time"]
        else:
            Parameters = ["Time"]
        
        # Methods = ["qq_e_pp","kstest","graphical"]
        Methods = ["graphical", "kstest"]
        
        # Percorre todos os métodos selecionados
        for meth in Methods:
            # Percorre todos os parâmetros
            for parameter in Parameters:
                
                if parameter == "Size":
                    # Adicionando valores gerados pelo NS3
                    x = size_ns3
                    
                    # Adicionando valores do trace
                    y = size_trace
                    
                if parameter == "Time":
                    # Adicionando valores gerados pelo NS3
                    x = time_ns3
                    
                    # Adicionando valores do trace
                    y = time_trace
                    
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

                    fig.savefig("/home/carl/New_Results/Graphics/"+case_Study+"_compare_qq_e_pp_plot_"+proto+"_("+parameter+")", fmt="png",dpi=1000)
                    plt.close()


                if meth == "kstest":  

                    x_ks = x
                    y_ks = y
                                    
                    # # Adicionando valores do trace
                    Ft = y_ks
                    # # Adocionando valores obtidos do NS3
                    t_Fe = x_ks
                    # Definindo tamanho do array
                    size = len(Ft)
                    
                    # Organiza dados 
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
                
                    # Criando listas para armazenar as ECDFs
                    Fe = []
                    Fe_ = []
                    
                    # Criando ECDFs
                    arr_ecdf = np.empty(size)
                    arr_ecdf = [id_+1 for id_ in range(size)]
                    arr_div = np.array(size) 
                    
                    Fe = (np.true_divide(arr_ecdf, size))
                    arr_ecdf = np.subtract(arr_ecdf, 1)
                    Fe_ = (np.true_divide(arr_ecdf, size))
                    
                    # Transformando listas em np.arrays()
                    Fe = np.array(Fe)
                    Fe_ = np.array(Fe_)
                    Ft = np.array(Ft)
                   
                    
                    # Inicio cálculo de rejeição
                    #
                    Ft_Fe_ = abs(np.subtract(Fe_, Ft))
                    Fe_Ft = (np.subtract(Fe, Ft))
                    Dobs_max = np.maximum(Ft_Fe_, Fe_Ft)
                    Dobs = np.max(Dobs_max)
                    #
                    # Fim cálculo de rejeição
                    
                    # Chamando função que retorna validação do KS-test
                    rejects, D_critico = ksvalid(size, Dobs, IC)

                    # Salvando arquivo com resultados do goodness of fitness
                    # Caso seja a primeira analise 
                    if ks_first == True:
                        # Abrindo arquido e sobrescrevendo linha
                        w = open("/home/carl/New_Results/Files/compare_results_"+parameter+".txt", "w")
                        w.write('"flow_Trace";"KS-Test";"Rejection"\n')
                        ks_first = False
                    else:
                        # Concatenando linha no arquivo
                        w = open("/home/carl/New_Results/Files/compare_results_"+parameter+".txt", "a")
                        w.write(''+str(proto) + ' ' + str(Dobs) + ' ' + str(rejects) + '\n')
                    w.close()

                    # Plotando resultados do teste KS
                    if save_Graph == True:
                        plt.plot(Ft, Fe_, 'o', label='CDF Trace Distribution')
                        plt.plot(t_Fe, Fe_, 'o', label='CDF NS3 Distribution')
                        
                        if parameter == "Size":
                            plt.xlabel('Size (Bytes)')
                        
                        if parameter == "Time":
                            plt.xlabel('Time (s)')
                        
                        plt.ylabel('CDF')
                        
                        # plt.plot(Ft, Fe, 'o', label='Teorical Distribution')
                        # plt.plot(t_Fe, Fe, 'o', label='Empirical Distribution')
                        
                        
                        # Definindo titulo do grpafico
                        plt.title("KS Test of Real and Simulated Trace of "+proto+ "(" + parameter + ")")
                        # Definindo Legenda
                        plt.legend()
                        # Definindo nome e diretório do gráfico
                        plt.savefig("/home/carl/New_Results/Graphics/"+case_Study+"_compare_ks_test_"+proto+"_("+parameter+")", fmt="png",dpi=1000)
                        # Fechando gráfico
                        plt.close()
            

                if meth == "graphical":
                    x_gr = x
                    y_gr = y

                    Fe = [] 
                    size = len(x_gr)
                    
                    # Criando ECDFs
                    arr_ecdf = np.empty(size)
                    arr_ecdf = [id_+1 for id_ in range(size)]
                    arr_div = np.array(size) 
                    
                    Fe = (np.true_divide(arr_ecdf, size))
                  

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
                    # text = f"$y={z[0]:0.6f}x{z[1]:+0.6f}$\n$R^2 = {r2_score(y_gr,y_hat):0.6f}$"
                    z1 = np.around(z[0], 4)
                    z2 = np.around(z[1], 4)
                    z3 = np.around(r2_score(y_gr,y_hat), 4)
                    text = "y="+ str(z1)+"x"+str(z2)+"\nR^2 = "+str(z3)
                    plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
                        fontsize=12, verticalalignment='top')
                    # Definindo titulo do gráfico
                    plt.title('Graphical Method Inference for Real and Simulated Trace '+case_Study+' ('+parameter+')')
                    plt.xlabel('ECDF of Real Trace Data')
                    plt.ylabel('Normalize Data of Simulated Trace')

                    plt.savefig("/home/carl/New_Results/Graphics/"+case_Study+"_compare_graphical_"+proto+"_("+parameter+")", fmt="png",dpi=1000)
                    plt.close()
      
# Função de leitura de arquivos 
def read_filter(const_Size, type_Size, save_Graph, case_Study):
                                                      
    # Definindo Dataframe com dados de ip e protocolos do arquivo de trace filtrado 
    ns3_ip_df = pd.read_csv("/home/carl/New_Results/Files/ns3_ip.txt", sep=";", names=["protocols","ip_SRC"])

    # Definindo Dataframe com dados do arquivo de trace filtrado 
    ns3_df = pd.read_csv("/home/carl/New_Results/Filter_Traces/ns3_"+case_Study+"_trace.txt", sep=";", names=["ip_SRC","ip_DST","time","size","protocols","tcp_Size","udp_Size"])
    # Removendo linhas com tamanho igual a 0
    ns3_df = ns3_df[ns3_df.tcp_Size != 0]
    ns3_df = ns3_df[ns3_df.udp_Size != 0]
    # Atribuindo zeros para valores com NaN
    ns3_df = ns3_df.fillna(0)
    # Removendo prefixo dos protocolos
    ns3_df['protocols'] = ns3_df.protocols.str.replace('ppp:ip:', '')

    # Identificando ips iguais com protocolos diferentes e substiruindo o protocolo
    for index, row in ns3_df.iterrows():
        for index_ip, row_ip in ns3_ip_df.iterrows():
            if row['ip_SRC'] == row_ip['ip_SRC']:
                ns3_df['protocols'][index] = row_ip['protocols']
   

    # Criando lista com protocolos
    ns3_arr_protocols = list(set(ns3_df["protocols"]))
    # Convertendo lista para numpy array
    ns3_arr_protocols = np.array(ns3_arr_protocols)
    
    # Percorre cada protocolo
    for ns3_proto in ns3_arr_protocols:
       
        # Criando Dataframe com valores de um único protocolo
        ns3_data_df = ns3_df[ns3_df['protocols'].str.contains(ns3_proto)]
        # Se o tamanho do dataframe criado for maior que dois.
        if len(ns3_data_df.index) > 2:

            ######## Definindo Tempos ######
            # Definindo variável que contem os tempos do protocolo
            ns3_t_Time = np.array(ns3_data_df["time"])
            # Ordenando valores da variável
            ns3_t_Time.sort()
 
            # Criando variável auxiliar para subtração dos valores
            ns3_sub = []
            
            # Subtraindo valores para obtenção do TBP (Time Between Packets)
            for i in range(0,len(ns3_t_Time)-1):
                ns3_sub.append(ns3_t_Time[i+1] - ns3_t_Time[i])
            
            # Passando valores resultantes para a variável t_time
            ns3_t_Time = np.array(ns3_sub)
            # Convertendo todos os valores para o tipo float
            ns3_t_Time = ns3_t_Time.astype(float)
            # Removendo valores igual a zero
            ns3_t_Time = np.delete(ns3_t_Time, np.where(ns3_t_Time == 0))
            # Ordenando valores
            ns3_t_Time.sort()

            # Plot histograma t_time:
            plot_histogram(ns3_t_Time, save_Graph, "time", case_Study, ns3_proto)
            # Salvando t_Time em um arquivo
            np.savetxt('/home/carl/New_Results/Files/ns3_'+ns3_proto+'_time.txt', ns3_t_Time, delimiter=',', fmt='%f')
            

            ############ Definindo Tamanhos #########
            # Se o tamanho dos pacotes não é constante
            if const_Size == False:
                # Plot histograma t_time:
                plot_histogram(ns3_data_df["size"], save_Graph, "size", case_Study, ns3_proto)
                # Salvando tamanhos em arquivo
                np.savetxt('/home/carl/New_Results/Files/ns3_'+ns3_proto+'_size.txt', ns3_data_df["size"], delimiter=',', fmt='%f')
            else:
                # Definindo como será o tamanho dos pacotes
                # Definição de type_Size pela média
                if type_Size == "mean_Trace":
                    ns3_size = np.mean(ns3_data_df["size"])
                # Definição de type_Size por um valor constante
                if type_Size == "const_Value":
                    ns3_size = 500
                # Atribuindo tamanho constante para array com a mesma quantidade de linhas que o arquivo original
                ns3_arr_Size = np.empty(len(ns3_data_df["size"])-1)
                ns3_arr_Size = [ns3_size for x in range(len(ns3_data_df["size"]))]
                # Salvando tamanhos em arquivo
                np.savetxt('/home/carl/New_Results/Files/ns3_'+ns3_proto+'_size.txt', ns3_arr_Size, delimiter=',', fmt='%f')
            
            
            # Removendo proto utilizado a cada rodada do "for" 
            ns3_df = ns3_df[ns3_df.protocols != ns3_proto]
        else:
            # Removendo protocolo que possui menos de dois envios de pacotes
            ns3_arr_protocols = np.delete(ns3_arr_protocols, np.where(ns3_arr_protocols == ns3_proto))
          


    # Definindo Dataframe com dados do arquivo de trace filtrado 
    trace_df = pd.read_csv("/home/carl/New_Results/Filter_Traces/"+case_Study+"_trace.txt", sep=";", names=["ip_SRC","ip_DST","time","size","protocols","tcp_Size","udp_Size"])
    # Removendo linhas com tamanho igual a 0
    trace_df = trace_df[trace_df.tcp_Size != 0]
    trace_df = trace_df[trace_df.udp_Size != 0]
    # Atribuindo zeros para valores com NaN
    trace_df = trace_df.fillna(0)

    # Removendo prefixo dos protocolos
    trace_df['protocols'] = trace_df.protocols.str.replace('ppp:ip:', '')

    # Criando lista com protocolos
    trace_arr_protocols = list(set(trace_df["protocols"]))
    # Convertendo lista para numpy array
    trace_arr_protocols = np.array(trace_arr_protocols)
    
    # Percorre cada protocolo
    for trace_proto in trace_arr_protocols:
       
        # Criando Dataframe com valores de um único protocolo
        trace_data_df = trace_df[trace_df['protocols'].str.contains(trace_proto)]
        # Se o tamanho do dataframe criado for maior que dois.
        if len(trace_data_df.index) > 2:

            ######## Definindo Tempos ######
            # Definindo variável que contem os tempos do protocolo
            trace_t_Time = np.array(trace_data_df["time"])
            # Ordenando valores da variável
            trace_t_Time.sort()
 
            # Criando variável auxiliar para subtração dos valores
            trace_sub = []
            
            # Subtraindo valores para obtenção do TBP (Time Between Packets)
            for i in range(0,len(trace_t_Time)-1):
                trace_sub.append(trace_t_Time[i+1] - trace_t_Time[i])
            
            # Passando valores resultantes para a variável t_time
            trace_t_Time = np.array(trace_sub)
            # Convertendo todos os valores para o tipo float
            trace_t_Time = trace_t_Time.astype(float)
            # Removendo valores igual a zero
            trace_t_Time = np.delete(trace_t_Time, np.where(trace_t_Time == 0))
            # Ordenando valores
            trace_t_Time.sort()

            # Plot histograma t_time:
            plot_histogram(trace_t_Time, save_Graph, "time", case_Study, trace_proto)
            # Salvando t_Time em um arquivo
            np.savetxt('/home/carl/New_Results/Files/'+trace_proto+'_time.txt', trace_t_Time, delimiter=',', fmt='%f')
            

            ############ Definindo Tamanhos #########
            # Se o tamanho dos pacotes não é constante
            if const_Size == False:
                # Plot histograma t_time:
                plot_histogram(trace_data_df["size"], save_Graph, "size", case_Study, trace_proto)
                # Salvando tamanhos em arquivo
                np.savetxt('/home/carl/New_Results/Files/'+trace_proto+'_size.txt', trace_data_df["size"], delimiter=',', fmt='%f')
            else:
                # Definindo como será o tamanho dos pacotes
                # Definição de type_Size pela média
                if type_Size == "mean_Trace":
                    trace_size = np.mean(trace_data_df["size"])
                # Definição de type_Size por um valor constante
                if type_Size == "const_Value":
                    trace_size = 500
                # Atribuindo tamanho constante para array com a mesma quantidade de linhas que o arquivo original
                trace_arr_Size = np.empty(len(trace_data_df["size"])-1)
                trace_arr_Size = [trace_size for x in range(len(trace_data_df["size"]))]
                # Salvando tamanhos em arquivo
                np.savetxt('/home/carl/New_Results/Files/trace_'+trace_proto+'_size.txt', trace_arr_Size, delimiter=',', fmt='%f')
            
            
            # Removendo proto utilizado a cada rodada do "for" 
            trace_df = trace_df[trace_df.protocols != trace_proto]
        else:
            # Removendo protocolo que possui menos de dois envios de pacotes
            trace_arr_protocols = np.delete(trace_arr_protocols, np.where(trace_arr_protocols == trace_proto))
          



    return trace_arr_protocols, ns3_arr_protocols

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
    trace_arr_protocols, ns3_arr_protocols = read_filter(const_Size, type_Size, save_Graph, case_Study)
    # Chamando função de comparação dos dados
    compare(ns3_arr_protocols, case_Study, const_Size, save_Graph, IC)

if __name__ == '__main__':
    main(sys.argv)