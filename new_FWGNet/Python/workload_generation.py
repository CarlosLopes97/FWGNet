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
import matplotlib.pyplot as plt

# plt.style.use('science')
# plt.rcParams.update({
#     "font.family": "serif",   # specify font family here
#     "font.serif": ["Times"],  # specify font here
#     "font.size":11})   
# plt.style.reload_library()


import os
import io
import subprocess
from scipy.interpolate import interp1d
from scipy.stats.distributions import chi2
import random
from matplotlib.ticker import FixedLocator, FixedFormatter
import math
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

    # if plot == "show":
    #     plt.show()
    # if plot == "save":
        

# Função para definir a distribuição de probabilidade compatível com os 
# valores do trace utilizada para gerar valores aleatórios por TCDF
        
def tcdf(y, parameter, case_Study, save_Graph, IC, proto, tcdf_First):
    # y = [142.773, 146.217, 147.676, 147.740, 149.016, 149.105, 150.476, 151.284, 151.461, 151.763, 151.932, 154.519, 154.632, 154.789, 155.008, 155.325, 155.402, 155.506, 155.545, 155.561, 155.581, 155.584, 155.701, 156.115, 156.340, 156.851, 156.879, 157.044, 157.404, 157.435, 157.573, 157.599, 157.688, 157.717, 157.858, 158.033, 158.154, 158.387, 158.475, 159.068, 159.215, 159.234, 159.366, 159.499, 159.576, 159.601, 159.767, 159.824, 159.978, 160.036, 160.289, 160.289, 160.327, 160.430, 160.496, 160.519, 160.719, 160.745, 160.942, 161.341, 161.438, 161.683, 161.767, 161.865, 162.064, 162.289, 162.302, 162.711, 162.752, 162.855, 162.866, 162.884, 162.918, 162.947, 163.136, 164.080, 164.138, 164.479, 164.524, 164.566, 164.850, 164.965, 165.000, 165.292, 165.397, 165.408, 165.538, 165.997, 166.311, 166.327, 166.367, 166.671, 167.214, 167.690, 168.178, 170.181, 170.633, 171.434, 173.424, 179.891]
    # y = np.array(y)
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
    dist_names = ['expon',
                'norm',
                'gamma',
                'lognorm',
                'norm',
                'pareto',
                'triang',
                'uniform',
                'dweibull',
                'weibull_min',
                'weibull_max',
                'erlang'
                ]
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
        y.sort()
        t_Fe = y


        # Criando listas para armazenar as ECDFs
        Fe = []
        Fe_ = []
        
        # ecdf = np.array([12.0, 15.2, 19.3])  
        arr_ecdf = np.empty(size)
        arr_ecdf = [id+1 for id in range(size)]
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
        Ft_ = np.array(Ft_)
        
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
        
        # ks_statistic, D_critico = stats.ks_samp(t_Fe,Ft)
            
        rejects, D_critico = ksvalid(size, Dobs, IC)
        # rejects, D_critico = ksvalid(size, Dobs)

        
        # Imprimindo resultados do KS Test
        print(" ")
        print("KS TEST:")
        print("Confidence degree: ", IC,"%")
        print("D observed: ", Dobs)
        # print("D observed(two samples): ", ks_statistic)
        print("D critical: ", D_critico)
        print(rejects, " to  Real Trace (Manual-ks_statistic/D_critico)")

        a = 1-(IC/100)
        
        a = np.around(a,4)

        if D_critico < a:
            print("Reject - p-value: ", D_critico, " is less wich alpha: ", a," (2samp)")
        else:
            print("Fails to Reject - p-value: ", D_critico, " is greater wich alpha: ", a," (2samp)")
        
        print(" ")
  
        # Obtém a estatística do teste KS e arredonda para 5 casas decimais
        Dobs = np.around(Dobs,  5)
        ks_values.append(Dobs)
        # ks_values.append(ks_statistic)    

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
        

        # Set x² with IC
        # IC = IC/100
        x2 = chi2.ppf(a, nbins-1)
        chi_square.append(ss)
        
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
    print ('Distributions sorted by KS Test of ',proto,'(',parameter,'):')
    print ('----------------------------------------')
    print (results)
    print (proto, parameter)


    # Divida os dados observados em N posições para plotagem (isso pode ser alterado)
    bin_cutoffs = np.linspace(np.percentile(y,0), np.percentile(y,99), nbins)

    # Crie o gráfico
    # if save_Graph == True:
    h = plt.hist(y, bins = bin_cutoffs, color='0.75')
    
    # Receba as principais distribuições da fase anterior 
    # e seleciona a quantidade de distribuições.
    number_distributions_to_plot = 1
    dist_names = results['Distribution'].iloc[0:number_distributions_to_plot]

    # Crie uma lista vazia para armazenar parâmetros de distribuição ajustada
    parameters = []
    
    # Faça um loop pelas distribuições para obter o ajuste e os parâmetros da linha
    for dist_name in dist_names:
        # Chamando variáveis globais       
        # Obtendo distribuições e seus parametros de acordo com o trace
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(y)
        parameters.append(param)

        arg = param[:-2]
        loc = param[-2]
        scale = param[-1]
        # print(parameters)

        # Obter linha para cada distribuição (e dimensionar para corresponder aos dados observados)
        pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale)
        scale_pdf = np.trapz(h[0],h[1][:-1])/np.trapz(pdf_fitted, x)
        pdf_fitted *= scale_pdf
        if save_Graph == True:
            # Adicione a linha ao gráfico
            plt.plot(pdf_fitted, label=dist_name)

            # Defina o eixo gráfico x para conter 99% dos dados
            # Isso pode ser removido, mas, às vezes, dados fora de padrão tornam o gráfico menos claro
            plt.xlim(0,np.percentile(y,99))
            plt.title("Histogram of trace (" + parameter + ") + theorical distribuition " + dist_name)
    # Adicionar legenda
    plt.legend()
    if save_Graph == True:
            plt.savefig("/home/carl/New_Results/Graphics/"+case_Study+"_histogram_"+proto+"_tcdf_("+dist_name+")_"+parameter, fmt="png",dpi=1000)
            plt.close()
    # Armazenar parâmetros de distribuição em um quadro de dados (isso também pode ser salvo)
    dist_parameters = pd.DataFrame()
    dist_parameters['Distribution'] = (
            results['Distribution'].iloc[0:number_distributions_to_plot])
    dist_parameters['Distribution parameters'] = parameters

    # Printar os parâmetros
    print ('Distribution parameters:')
    print ('------------------------')

    for row in dist_parameters.iterrows():
        print ('Distribution:', row[0])
        print ('Parameters:', row[1])

    
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
        Ft_ = np.array(Ft_)
        
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

        if tcdf_First == True:
            w = open("/home/carl/New_Results/Files/tcdf_results_"+parameter+".txt", "w")
            w.write('"flow_Trace";"Distributions";"KS-Test";"Chi-Square";"Rejection"\n')
            tcdf_First = False
        else:
            w = open("/home/carl/New_Results/Files/tcdf_results_"+parameter+".txt", "a")
            w.write(''+str(proto) + ' ' + str(dist_names) + ' ' + str(ks_values) + ' ' + str(chi_square) + ' ' + str(rejects) + '\n')
        w.close()
        # Imprimindo resultados do KS Test
        # print(" ")
        # print("KS TEST:")
        # print("Confidence degree: ", IC,"%")
        # print("D observed: ", Dobs)
        # # print("D observed(two samples): ", ks_statistic)
        # print("D critical: ", D_critico)
        # print(rejects, " to  Real Trace (Manual-ks_statistic/D_critico)")

        # a = 1-(IC/100)
        
        # a = np.around(a,4)

        # if D_critico < a:
        #     print("Reject - p-value: ", D_critico, " is less wich alpha: ", a," (2samp)")
        # else:
        #     print("Fails to Reject - p-value: ", D_critico, " is greater wich alpha: ", a," (2samp)")
        
        # print(Fe_)

        # Plotando resultados do teste KS
        if save_Graph == True:
            plt.plot(t_Fe, Ft, 'o', label='Teorical Distribution')
            plt.plot(t_Fe, Fe_, 'o', label='Empirical Distribution')
            
            # plt.plot(Ft, Fe, 'o', label='Teorical Distribution')
            # plt.plot(t_Fe, Fe, 'o', label='Empirical Distribution')
            
            # Definindo titulo
            plt.title("KS Test of Real Trace of "+proto+" with " + distribution + " Distribution (" + parameter + ")")
            plt.legend()
            
            plt.savefig("/home/carl/New_Results/Graphics/"+case_Study+"_valid-ks_test_"+proto+"_tcdf_("+distribution+")_"+parameter, fmt="png",dpi=1000)
            plt.close()
     
        return dist_name, loc, scale, arg, tcdf_First

def tcdf_generate(dist, loc, scale, arg, parameter, size):
    # Setar distribuição escolhida.
    dist_name = getattr(scipy.stats, dist)

    # Gerar número aleatório de acordo com a distribuição escolhida e seus parametros.
    r_N = dist_name.rvs(loc=loc, scale=scale, *arg, size=size)

    # Condição para retorno do valor de acordo com o parametro de rede.
    if parameter == "size":
        # print("SIZE R_N:", r_N)
        return r_N.astype(int)

    if parameter == "time":
        # print("TIME R_N:", r_N)
        return r_N.astype(float)

# Função de geração de variáveis aleatórias de acordo com distribuições 
# de probabilidade e parametros definidos
def PD(parameter, const_Size, size_Trace):
    # Mais distribuições podem ser encontradas no site da lib "scipy"
    # Veja https://docs.scipy.org/doc/scipy/reference/stats.html para mais detalhes
    # global request
    # global size_request
    # global time_request
    r_N = []
    if parameter == "size":
        if const_Size == False:
        # Selecionando distribuição de probabilidade para o parametro Size
            dist_name = 'uniform'
            # Definindo parametros da distribuição
            loc = 1024
            scale = 2048
            arg = []
            # Setando distribuição a escolhida e seus parametros 
            dist = getattr(scipy.stats, dist_name)
            # Gerando número aleatório de acordo com a distribuiução e os parametros definidos
            r_N = dist.rvs(loc=loc, scale=scale, *arg, size=int(size_Trace))
        # size_request = True
       

    if parameter == "time":
       # Selecionando distribuição de probabilidade para o parametro Size
        dist_name = 'uniform'
        # Definindo parametros da distribuição
        loc = 0.5
        scale = 0.8
        arg = []
        # Setando distribuição a escolhida e seus parametros 
        dist = getattr(scipy.stats, dist_name)
        # Gerando número aleatório de acordo com a distribuiução e os parametros definidos
        r_N = dist.rvs(loc=loc, scale=scale, *arg, size=int(size_Trace))
        
    return(r_N)

def ecdf(y, parameter, proto):
    size = len(y)
    # Criando listas para os dados utilizados
    Fx = []
    Fx_ = []
    
    Fe = []
    Fe_ = []
    
    # ecdf = np.array([12.0, 15.2, 19.3])  
    arr_ecdf = np.empty(size)
    arr_ecdf = [x+1 for x in range(size)]
    arr_div = np.array(size) 
    
    # Criando ECDFs
    # for i in range(0, size):
    # ecdf i-1/n
    Fx = (np.true_divide(arr_ecdf, size))
    # print(len(Fx))
    arr_ecdf = np.subtract(arr_ecdf, 1)

    Fx_ = (np.true_divide(arr_ecdf, size))
    # print(len(Fx_))
    # Realizando ajustes para os vetores que selecionaram os valores gerados
    # for i in range(0, len(y)):
    #     Fx.append(i/(len(y)+1))
    #     print(Fx)
    #     if i != 0:
    #         Fx_.append(i/(len(y)+1))
    # Adicionando 1 no vetor Fx_
    Fx_[:-1]=1    

    # print ("Fx: ", len(Fx))
    # print ("Fx_: ", len(Fx_))
    r_N = []
    for id in range(0, len(y)):
        # Gerando um valor aleatório entre 0 e 1 uniforme
        rand = np.random.uniform(0,1)
        # print("Rand: ", rand)
        
        # Pecorrer todos os valores do vetor com dados do trace
        # para determinar o valor a ser gerado de acordo com o resultado da distribuição uniforme
        # r_N = 0
        for i in range(0, len(y)):
            # Condição que define em qual classe o valor é encontrado
            if rand > Fx[i] and rand < Fx_[i]:
                # Determinando o valor resultante 
                r_N.append(y[i-1]) 
            
    # w = open("/home/carl/New_Results/Filter_Traces/RV_ecdf_"+proto+"_"+parameter+".txt", "a")
    # w.write("\n" + str(r_N) + "\n")
    # w.close()
    return r_N


# Função de definição da aplicação HTTP
def read_filter(const_Size, type_Size, save_Graph, case_Study):
    # global const_Size
    first = True                                                    
    txt_df = pd.read_csv("/home/carl/New_Results/Filter_Traces/"+case_Study+"_trace.txt", sep=";", names=["ip_SRC","ip_DST","time","size","protocols","tcp_Size","udp_Size"])
    
    txt_df = txt_df[txt_df.tcp_Size != 0]
    txt_df = txt_df[txt_df.udp_Size != 0]
    # print(txt_df)
    # txt_df["ip_SRC"] = pd.to_numeric(txt_df["ip_SRC"],errors='coerce')
    txt_df.dropna(subset = ["ip_SRC"], inplace=True)
    # print(txt_df)
    
    txt_df['protocols'] = txt_df.protocols.str.replace('ethertype:ip:', '')
    txt_df['protocols'] = txt_df.protocols.str.replace('wlan_radio:wlan:', '')

    # txt_df['protocols'] = [x.split('ethertype:ip:')[0] for x in txt_df['protocols']]
    # txt_df['protocols'] = [x.split('wlan_radio:wlan:')[0] for x in txt_df['protocols']]

    arr_protocols = list(set(txt_df["protocols"]))
    # print(arr_protocols)
    # id_Proto = 0
    # for id_Proto in range(len(arr_protocols)):
    #     # arr_protocols[id_Proto] = arr_protocols[id_Proto] +'_'+str(id_Proto)

    #     print(arr_protocols[id_Proto])

        # aux_txt_df = txt_df['protocols'].str.contains(str(arr_protocols[id_Proto])).astype(str)+'_'+str(id_Proto)
        # print(arr_protocols[id_Proto])
        # txt_df['protocols'] = aux_txt_df
        # print(aux_txt_df)
        # del aux_txt_df
    
    # arr_protocols = list(set(txt_df["protocols"]))
    # print(arr_protocols)
    # Agrupar dados do framework por protocol
    # Criar os arquivos de acordo com os protocolos diferentes
    # print(txt_df["protocols"])




    # txt_df['ip_SRC'][3] = "172.1.1.5"
    # arr_IP_src = list(set(txt_df["ip_SRC"]))
    # arr_IP_dst = list(set(txt_df["ip_DST"]))
    # arr_IP = arr_IP_dst + arr_IP_src
    # arr_IP = list(dict.fromkeys(arr_IP))
    
    first_IP = True
    # Cria valores só pra um valor da coluna 
    for proto in arr_protocols:
        print(proto)
        # txt_df[] = txt_df.loc[txt_df['protocols'] == str(eth:ethertype:ip:tcp)]
        data_df = txt_df[txt_df['protocols'].str.contains(proto)]
        
        
        # remover = "ethertype:ip:"
        # remover = "wlan_radio:wlan:"
        
        # print(txt_df)
        # arr_IP = txt_df['ip_DST'][txt_df['ip_DST'].duplicated(keep=False)]
        # proto_IP = []
        # for ip in arr_IP:
        # proto_IP.append(data_df.loc[data_df['protocols'] == proto, arr_IP])
        # proto_IP.append(data_df.loc[data_df['protocols'] == proto, arr_IP])
        # txt_df = txt_df.drop_duplicates(subset=["ip_SRC", "ip_DST"])
        # txt_df = txt_df.drop_duplicates(subset="ip_SRC")
        # print(txt_df)
        # txt_df = txt_df.drop_duplicates(subset="ip_DST")
        # print(txt_df)
        # a = txt_df.set_index(["protocols", "ip_SRC"]).count(level="protocols")
        # b = txt_df.set_index(["protocols", "ip_DST"]).count(level="protocols")
        # print(a)
        # print(b)
        # id_src_IP = 0
        # count_ip_SRC = 0
        # for id_src_IP in arr_IP_src:
        #     if txt_df["protocols"][:-1] == arr_IP_src[id_arr_IP]:
        #         count_ip_SRC+=1
        # id_dst_IP = 0
        # count_ip_DST = 0
        # for id_dst_IP in arr_IP_dst:
        #     df[df['c'].isin(A)]
        #     if txt_df["protocols"].loc[:-1] == arr_IP_dst[id_dst_IP]:
        #         count_ip_DST+=1
        
        # arr_IP_src = list(set(a['ip_DST']))
        # arr_IP_dst = list(set(b['ip_SRC']))
        # arr_IP = arr_IP_dst + arr_IP_src
        # arr_IP = list(dict.fromkeys(arr_IP))
        # print(arr_IP)
        ip_proto_df = txt_df.drop(['time', 'size', 'tcp_Size', 'udp_Size'], axis=1)
    
        ip_proto_df['ip_DST'][3] = "172.1.1.5"
        
        # print(ip_proto_df)

        dst_proto_df = ip_proto_df.drop(['ip_SRC'], axis=1)
        src_proto_df = ip_proto_df.drop(['ip_DST'], axis=1)
        # print(dst_proto_df)
        # print(src_proto_df)

        dst_proto_df = dst_proto_df.drop_duplicates()
        src_proto_df = src_proto_df.drop_duplicates()
        
        count_ip_SRC = (src_proto_df.protocols == proto).sum()
        count_ip_DST = (dst_proto_df.protocols == proto).sum()

        if first_IP == True:
            w = open("/home/carl/New_Results/Files/proto_ips.txt", "w")
            # w.write('"Proto";"SRC";"DST"\n')
            w.write(''+str(proto) + ' ' + str(count_ip_SRC) + ' ' + str(count_ip_DST) + '\n')
            first_IP = False
        else:
            w = open("/home/carl/New_Results/Files/proto_ips.txt", "a")
            w.write(''+str(proto) + ' ' + str(count_ip_SRC) + ' ' + str(count_ip_DST) + '\n')
        w.close()
   
        t_Time = np.array(data_df["time"])
        t_Time.sort()
        # print(t_Time)
        sub = []
        if len(t_Time) > 1:
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

        np.savetxt('/home/carl/New_Results/Files/'+case_Study+'_flow_'+proto+'_time.txt', t_Time, delimiter=',', fmt='%f')

        if const_Size == False:
            # Plot histograma t_time:
            # print(data_df["size"])
            plot_histogram(data_df["size"], save_Graph, "size", case_Study, proto)
            if len(t_Time) > 1:
                np.savetxt('/home/carl/New_Results/Files/'+case_Study+'_flow_'+proto+'_size.txt', data_df["size"], delimiter=',', fmt='%f')
            
        else:

            if type_Size == "mean_Trace":
                size = np.mean(data_df["size"])
            if type_Size == "const_Value":
                size = 500
            
            
            arr_Size = np.empty(len(data_df["size"])-1)
            arr_Size = [size for x in range(len(data_df["size"]))]
            if len(t_Time) > 1:
                np.savetxt('/home/carl/New_Results/Files/'+case_Study+'_flow_'+proto+'_size.txt', arr_Size, delimiter=',', fmt='%f')
        
        
        
        if first == True:
            w = open("/home/carl/New_Results/Files/list_tr_size.txt", "w")
            w.write('"flow_Trace";"size_Trace"\n')
            first = False
    
        else:
            w = open("/home/carl/New_Results/Files/list_tr_size.txt", "a")
        if len(data_df) > 1:
            w.write(''+str(proto) + ' ' + str(len(data_df)) + '\n')
        w.close()
        # print(proto)
        
        
        txt_df = txt_df[txt_df.protocols != proto]


            # lista = list(proto, size)
            # param_df = pd.DataFrame({'flows': [proto], 'len': [size]})
            # param_df = pd.Dataframe(float(len(data_df["size"])), "")
            # param_df.to_csv('/home/carl/New_Results/Files/list_tr_size.txt')

        # np.savetxt('/home/carl/New_Results/Files/flow_'+proto+'_size.txt', param_df, delimiter=',', fmt='%f')
        # timeStopSimulation =  time_resp_df["time"].iloc[-1]
        # print(timeStopSimulation)
        # nRequestPackets = len(time_req_df["time"])
        # nResponsePackets = len(time_resp_df["time"])
    return arr_protocols

def kstest():
    y = [142.773, 146.217, 147.676, 147.740, 149.016, 149.105, 150.476, 151.284, 151.461, 151.763, 151.932, 154.519, 154.632, 154.789, 155.008, 155.325, 155.402, 155.506, 155.545, 155.561, 155.581, 155.584, 155.701, 156.115, 156.340, 156.851, 156.879, 157.044, 157.404, 157.435, 157.573, 157.599, 157.688, 157.717, 157.858, 158.033, 158.154, 158.387, 158.475, 159.068, 159.215, 159.234, 159.366, 159.499, 159.576, 159.601, 159.767, 159.824, 159.978, 160.036, 160.289, 160.289, 160.327, 160.430, 160.496, 160.519, 160.719, 160.745, 160.942, 161.341, 161.438, 161.683, 161.767, 161.865, 162.064, 162.289, 162.302, 162.711, 162.752, 162.855, 162.866, 162.884, 162.918, 162.947, 163.136, 164.080, 164.138, 164.479, 164.524, 164.566, 164.850, 164.965, 165.000, 165.292, 165.397, 165.408, 165.538, 165.997, 166.311, 166.327, 166.367, 166.671, 167.214, 167.690, 168.178, 170.181, 170.633, 171.434, 173.424, 179.891]
    y.sort()
    # Set up distribution
    size = len(y)
    distribution = 'norm'
    dist = getattr(scipy.stats, distribution)
    param = dist.fit(y)
    print(param)
  

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
    
    # ecdf = np.array([12.0, 15.2, 19.3])  
    arr_ecdf = np.empty(size)
    arr_ecdf = [x+1 for x in range(size)]
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
    Ft_ = np.array(Ft_)
    
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

    # print("T_FE: ", t_Fe)
    # print("FT: ", Ft)
    # print("FE: ", Fe)
    # Plotando resultados do teste KS
    plt.plot(t_Fe, Ft, 'o', label='Teorical Distribution')
    plt.plot(t_Fe, Fe, 'o', label='Empirical Distribution')
    
    
    # plt.plot(t_Fe, Fe, 'o', label='Real Trace')
    # plt.plot(Ft, Fe, 'o', label='Syntatic Trace')
    # Definindo titulo
    plt.title("KS Test of Real Trace with " + distribution + " Distribution")
    plt.legend()
    plt.show() 
    
def main(argv):
    # kstest()
    #
    # Filtro e criação de arquivos
    #
    # Determina se os tamanhos de pacotes serão constantes caso True ou se seguirão o padrão do trace caso False
    const_Size = True
    # Tipo de tamanho de pacote: "const_Value"(Valor específico) | "mean_Trace"(Usa a média do tamanho dos pacotes do trace)
    type_Size = "const_Value"
    save_Graph = True
    # parameters = ["size", "time"]
    case_Study = "http"
    # "99.80%";"99.85%";"99.90%";"99.95%";"99.99%"
    IC=95.0
    # Chamada da função de filtro do trace e criação de arquivos com os parametros da rede
    arr_protocols = read_filter(const_Size, type_Size, save_Graph, case_Study)

    #
    # Criação das variáveis aleatórias
    #
    # Determinação do método de geração de carga de trabalho
    # TCDF (Criação de teóricas com método de fitness com trace) 
    # ECDF (Criação a partir da distribuição empirica do Trace)
    # PD (Criação por meio de distribuições desejadas e seus parâmetros)
    mt_RG = "PD"
    # Definindo os parametros de tempo(Time) e tamanho(Size)
    parameters = ["size", "time"]
    tcdf_First = True

    for proto in arr_protocols:
        for parameter in parameters:
            aux_Packet = 0
            # txt_df = pd.read_csv("/home/carl/New_Results/Filter_Traces/http_trace.txt", sep=";", names=["ip_SRC","ip_DST","time","size","protocols"])
            filter_Trace = np.loadtxt("/home/carl/New_Results/Files/"+case_Study+"_flow_"+proto+"_"+parameter+".txt", usecols=0)
            filter_Trace = np.array(filter_Trace)
            # print(filter_Trace)
            # filter_Trace.sort()
            size_Trace = len(filter_Trace)

            # Condição de escolha do método de geração de variáveis aleatórias 
            # diretamente por uma distribuição de probabiidade
            if mt_RG == "PD":
                # Chamando a função PD() e retornando valor gerado para uma variável auxiliar
                aux_Packet = PD(parameter, const_Size, size_Trace)
                # Salvando arquivos de variáveis aleatórias
                np.savetxt('/home/carl/New_Results/Files/RV_'+mt_RG+'_'+proto+'_'+parameter+'.txt', aux_Packet, delimiter=',', fmt='%f')
            
            if parameter == "time" or const_Size == False:
                # Condição de escolha do método por distribuições teórica equivalentes aos dados do trace
                if mt_RG == "tcdf":
                    # Condição de chamada única da função tcdf()
                    # Chamando a função tcdf para definir a distribuição de probabilidade compatível ao trace e 
                    # seus respectivos parametros para geração de números aleatórios
                    dist_name, loc, scale, arg, tcdf_First = tcdf(filter_Trace, parameter, case_Study, save_Graph, IC, proto, tcdf_First)

                    # Chamando a função tcdf_generate e retornando valor gerado para uma variável auxiliar
                    aux_Packet = tcdf_generate(dist_name, loc, scale, arg, parameter, len(filter_Trace))
                    # Salvando arquivos de variáveis aleatórias
                    np.savetxt('/home/carl/New_Results/Files/RV_'+mt_RG+'_'+proto+'_'+parameter+'.txt', aux_Packet, delimiter=',', fmt='%f')
            

                # Condição de escolha do método pela distribuição empírica dos dados do trace
                if mt_RG == "ecdf":
                    # Chamando a função ecdf e retornando valor gerado para uma variável auxiliar
                    aux_Packet = ecdf(filter_Trace, parameter, proto)
                    # Salvando arquivos de variáveis aleatórias
                    np.savetxt('/home/carl/New_Results/Files/RV_'+mt_RG+'_'+proto+'_'+parameter+'.txt', aux_Packet, delimiter=',', fmt='%f')
            
                        
                        
            

if __name__ == '__main__':
    main(sys.argv)