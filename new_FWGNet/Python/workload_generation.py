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
# from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import os
import io
import subprocess
# import statsmodels.distributions.empirical_distribution as edf
from scipy.interpolate import interp1d
from scipy.stats.distributions import chi2
import random
from matplotlib.ticker import FixedLocator, FixedFormatter


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
    IC = str(IC)+"%"
    # print("IC: ", IC)
    if (size<=35):
        ks_df = pd.read_csv("/home/carl/New_Results/Files/kstest.txt", sep=";")
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



def plot_histogram(y, save_graph, parameter, case_study, proto):
    if save_graph == True:
        fig, ax = plt.subplots(1, 1)
        ax = sns.distplot(y)
        plt.title("Histogram of flow "+proto+" ("+parameter+")")
        fig.savefig("/home/carl/New_Results/Files/"+case_study+"_histogram_"+proto+"_hist_"+parameter, fmt="png",dpi=1000)
        plt.close()

    # if plot == "show":
    #     plt.show()
    # if plot == "save":
        

# Função para definir a distribuição de probabilidade compatível com os 
# valores do trace utilizada para gerar valores aleatórios por TCDF
        
def tcdf(y, parameter, case_study, save_graph, IC):
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
        
        ks_statistic, p_value = stats.ks_2samp(Ft,t_Fe)
            
        rejects, IC, D_critico = ksvalid(len(t_Fe), ks_statistic, IC)
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
    print ('\nDistributions sorted by KS Test of ',proto,'(',parameter,'):')
    print ('----------------------------------------')
    print (results)
    print (proto," ",parameter," ",y[0:3])
    # Divida os dados observados em N posições para plotagem (isso pode ser alterado)
    bin_cutoffs = np.linspace(np.percentile(y,0), np.percentile(y,99), nbins)

    # Crie o gráfico
    # if save_graph == True:
    h = plt.hist(y, bins = bin_cutoffs, color='0.75')
    
    if save_graph == False:
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
        # Obtendo distribuições e seus parametros de acordo com o trace
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(y)
        parameters.append(param)

        arg = param[:-2]
        loc = param[-2]
        scale = param[-1]
        print(parameters)

        # Obter linha para cada distribuição (e dimensionar para corresponder aos dados observados)
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
        scale_pdf = np.trapz (h[0], h[1][:-1]) / np.trapz (pdf_fitted, x)
        pdf_fitted *= scale_pdf
        if save_graph == True:
            # Adicione a linha ao gráfico
            plt.plot(pdf_fitted, label=dist_name)

            # Defina o eixo gráfico x para conter 99% dos dados
            # Isso pode ser removido, mas, às vezes, dados fora de padrão tornam o gráfico menos claro
            plt.xlim(0,np.percentile(y,99))
            plt.title("Histogram of trace (" + parameter + ") + theorical distribuition " + dist_name)
    # Adicionar legenda
    plt.legend()
    if save_graph == True:
        if plot == "show":
            plt.show()
        if plot == "save":
            plt.savefig("/home/carl/New_Results/Files/"+case_study+"_"+histogram+"_"+proto+"_tcdf_"+dist_name+"_"+parameter, fmt="png",dpi=1000)
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
      
        rejects, IC, D_critico = ksvalid(len(t_Fe), ks_statistic, IC)
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
        if save_graph == True:
            plt.plot(t_Fe, Ft, 'o', label='Teorical Distribution')
            plt.plot(t_Fe, Fe, 'o', label='Empirical Distribution')
            
            
            # plt.plot(t_Fe, Fe, 'o', label='Real Trace')
            # plt.plot(Ft, Fe, 'o', label='Syntatic Trace')
            # Definindo titulo
            plt.title("KS Test of Real Trace of "+proto+" with " + distribution + " Distribution (" + parameter + ")")
            plt.legend()
            if save_Plot == True:
            #     plt.show()
            # if plot == "save":
                plt.savefig("../../../Results/Figures/valid-"+validation+"_"+t_net+"_"+app_protocol+"_kstest_tcdf_"+proto+"_"+distribution+"_"+parameter, fmt="png",dpi=1000)
                plt.close()
     
        return dist_name, loc, scale, arg

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
def PD(parameter, const_Size):
    # Mais distribuições podem ser encontradas no site da lib "scipy"
    # Veja https://docs.scipy.org/doc/scipy/reference/stats.html para mais detalhes
    # global request
    # global size_request
    # global time_request
    r_N = []
    if parameter == "Size":
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
       

    if parameter == "Time":
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
def read_filter(const_Size, type_Size, save_graph, case_study):
    # global const_Size
    first = True                                                    
    txt_df = pd.read_csv("/home/carl/New_Results/Filter_Traces/"+case_study+"_trace.txt", sep=";", names=["ip_SRC","ip_DST","Time","Size","protocols"])
    
    # print(txt_df)
    # txt_df = txt_df[txt_df.Size > 0]

    # Agrupar dados do framework por protocol
    # Criar os arquivos de acordo com os protocolos diferentes
    arr_protocols = list(set(txt_df["protocols"]))
    
    # print(arr_protocols)
    # Cria valores só pra um valor da coluna 
    for proto in arr_protocols:

        # txt_df[] = txt_df.loc[txt_df['protocols'] == str(eth:ethertype:ip:tcp)]
        data_df = txt_df[txt_df['protocols'].str.contains(str(proto))]
        # print(proto)
        # print(data_df)
    
        t_Time = np.array(data_df["Time"])
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
            plot_histogram(t_Time, save_graph, "Time", case_study, proto)

        np.savetxt('/home/carl/New_Results/Files/'+case_study+'_flow_'+proto+'_time.txt', t_Time, delimiter=',', fmt='%f')

        if const_Size == False:
            # Plot histograma t_time:
            # print(data_df["Size"])
            plot_histogram(data_df["Size"], save_graph, "Size", case_study, proto)
            if len(t_Time) > 1:
                np.savetxt('/home/carl/New_Results/Files/'+case_study+'_flow_'+proto+'_size.txt', data_df["Size"], delimiter=',', fmt='%f')
            
        else:

            if type_Size == "mean_Trace":
                size = np.mean(data_df["Size"])
            if type_Size == "const_Value":
                size = 500
            
            
            arr_Size = np.empty(len(data_df["Size"])-1)
            arr_Size = [size for x in range(len(data_df["Size"]))]
            if len(t_Time) > 1:
                np.savetxt('/home/carl/New_Results/Files/'+case_study+'_flow_'+proto+'_size.txt', arr_Size, delimiter=',', fmt='%f')
        
        
        
        if first == True:
            w = open("/home/carl/New_Results/Files/list_tr_size.txt", "w")
            w.write('"flow_Trace";"size_Trace"\n')
            first = False
    
        else:
            w = open("/home/carl/New_Results/Files/list_tr_size.txt", "a")
        if len(data_df) > 1:
            w.write('"'+str(proto) + '";"' + str(len(data_df)) + '"\n')
        w.close()
        # print(proto)


            # lista = list(proto, size)
            # param_df = pd.DataFrame({'flows': [proto], 'len': [size]})
            # param_df = pd.Dataframe(float(len(data_df["Size"])), "")
            # param_df.to_csv('/home/carl/New_Results/Files/list_tr_size.txt')

        # np.savetxt('/home/carl/New_Results/Files/flow_'+proto+'_size.txt', param_df, delimiter=',', fmt='%f')
        # timeStopSimulation =  time_resp_df["Time"].iloc[-1]
        # print(timeStopSimulation)
        # nRequestPackets = len(time_req_df["Time"])
        # nResponsePackets = len(time_resp_df["Time"])
    return arr_protocols



def main(argv):
    #
    # Filtro e criação de arquivos
    #
    # Determina se os tamanhos de pacotes serão constantes caso True ou se seguirão o padrão do trace caso False
    const_Size = True
    # Tipo de tamanho de pacote: "const_Value"(Valor específico) | "mean_Trace"(Usa a média do tamanho dos pacotes do trace)
    type_Size = "const_Value"
    save_graph = True
    # parameters = ["Size", "Time"]
    case_study = "http"
    # "99.80%";"99.85%";"99.90%";"99.95%";"99.99%"
    IC="95.0"
    # Chamada da função de filtro do trace e criação de arquivos com os parametros da rede
    arr_protocols = read_filter(const_Size, type_Size, save_graph, case_study)

    #
    # Criação das variáveis aleatórias
    #
    # Determinação do método de geração de carga de trabalho
    # TCDF (Criação de teóricas com método de fitness com trace) 
    # ECDF (Criação a partir da distribuição empirica do Trace)
    # PD (Criação por meio de distribuições desejadas e seus parâmetros)
    mt_RG = "tcdf"
    # Definindo os parametros de tempo(Time) e tamanho(Size)
    parameters = ["size", "time"]

    for proto in arr_protocols:
        for parameter in parameters:
            # txt_df = pd.read_csv("/home/carl/New_Results/Filter_Traces/http_trace.txt", sep=";", names=["ip_SRC","ip_DST","Time","Size","protocols"])
            filter_Trace = np.loadtxt("/home/carl/New_Results/Files/"+case_study+"_flow_"+proto+"_"+parameter+".txt", usecols=0)
            print(str("flow_"+proto+"_"+parameter+".txt"))
            filter_Trace = np.array(filter_Trace)
            # print(filter_Trace)
            # filter_Trace.sort()
            size_Trace = len(filter_Trace)

            # Condição de escolha do método de geração de variáveis aleatórias 
            # diretamente por uma distribuição de probabiidade
            if mt_RG == "PD":
                # Chamando a função PD() e retornando valor gerado para uma variável auxiliar
                aux_Packet = PD(parameter, const_Size)
             
            # Condição de escolha do método por distribuições teórica equivalentes aos dados do trace
            if mt_RG == "tcdf":
                # Condição de chamada única da função tcdf()
                # Chamando a função tcdf para definir a distribuição de probabilidade compatível ao trace e 
                # seus respectivos parametros para geração de números aleatórios
                dist_name, loc, scale, arg = tcdf(filter_Trace, parameter, case_study, save_graph, IC)

                # Chamando a função tcdf_generate e retornando valor gerado para uma variável auxiliar
                aux_Packet = tcdf_generate(dist_name, loc, scale, arg, parameter)

            # Condição de escolha do método pela distribuição empírica dos dados do trace
            if mt_RG == "ecdf":
                # Chamando a função ecdf e retornando valor gerado para uma variável auxiliar
                aux_Packet = ecdf(filter_Trace, parameter, proto)
                    
                    
            # Salvando arquivos de variáveis aleatórias
            np.savetxt('/home/carl/New_Results/Files/RV_'+mt_RG+'_'+proto+'_'+parameter+'.txt', aux_Packet, delimiter=',', fmt='%f')
            

if __name__ == '__main__':
    main(sys.argv)