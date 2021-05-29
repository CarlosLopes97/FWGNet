# coding: utf-8

import sys
import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
# import seaborn as sns
# import statsmodels as sm
import scipy.stats as stats
# from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib as plt
import os
import io
import subprocess
# import statsmodels.distributions.empirical_distribution as edf
from scipy.interpolate import interp1d
from scipy.stats.distributions import chi2
import random
from matplotlib.ticker import FixedLocator, FixedFormatter


# Função de leitura dos traces e atribuição dos respectivos dados aos vetores
def read_txt(parameter, traffic, app_protocol): 
    global plot_graph
    global t_net
    if parameter == "Time" and traffic == "send":
        # Chamando variáveis globais
        global t_time
        global first_trace_time
     
        # Abrindo arquivos .txt
        t_time = np.loadtxt("/home/carl/New_Results/Files/"+app_protocol+"_time.txt", usecols=0)
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
        t_size = np.loadtxt("/home/carl/New_Results/Files/"+app_protocol+"_size.txt", usecols=0)
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
        req_t_time = np.loadtxt("/home/carl/New_Results/Files/"+app_protocol+"_req_time.txt", usecols=0)
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
        req_t_size = np.loadtxt("/home/carl/New_Results/Files/"+app_protocol+"_req_size.txt", usecols=0)
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
        resp_t_time = np.loadtxt("/home/carl/New_Results/Files/"+app_protocol+"_resp_time.txt", usecols=0)
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
        resp_t_size = np.loadtxt("/home/carl/New_Results/Files/"+app_protocol+"_resp_size.txt", usecols=0)
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
        # np.savetxt('/home/carl/New_Results/Files/'+'ecdf_'+app_protocol+'_time_req_ns3.txt', time_req_ns3, delimiter=',', fmt='%f')
        return(abs(r_N))


# Função de definição da aplicação HTTP
def read_filter(const_Size, type_Size):
    # global const_Size
                                                             
    txt_df = pd.read_csv("/home/carl/New_Results/Filter_Traces/http_trace.txt", sep=";", names=["ip_SRC","ip_DST","Time","Size","protocols"])
    # print(txt_df)
    # txt_df = txt_df[txt_df.Size > 0]

    # Agrupar dados do framework por protocol
    # Criar os arquivos de acordo com os protocolos diferentes
    arr_protocols = list(set( txt_df["protocols"]))
    
    # print(arr_protocols)
    # Cria valores só pra um valor da coluna 
    for proto in arr_protocols:
        # txt_df[] = txt_df.loc[txt_df['protocols'] == str(eth:ethertype:ip:tcp)]
        time_df = txt_df[txt_df['protocols'].str.contains(str(proto))]
        # print(time_df)
    




    # time_req_df = txt_df[txt_df['req_resp'].str.contains("GET")]
    # time_req_df["Time"] = time_req_df["Time"].apply(pd.to_numeric)
    # print(time_req_df)
    # time_resp_df = txt_df[txt_df['req_resp'].str.contains("OK")]
    # print(time_resp_df)
    # time_resp_df["Time"] = time_resp_df["Time"].apply(pd.to_numeric)
    
        np.savetxt('/home/carl/New_Results/Files/flow_'+proto+'_time.txt', time_df["Time"], delimiter=',', fmt='%f')
    # np.savetxt('/home/carl/New_Results/Files/'+app_protocol+'_req_time.txt', time_req_df["Time"], delimiter=',', fmt='%f')
        size_df = txt_df[txt_df['protocols'].str.contains(str(proto))]
        if const_Size == False:
        # Abrindo arquivos .txt
            
            # size_req_df["Size"] = size_req_df["Size"].apply(pd.to_numeric)

            # size_resp_df = txt_df[txt_df['req_resp'].str.contains("HTTP/1.1")]
            # size_resp_df["Size"] = size_resp_df["Size"].apply(pd.to_numeric)
            # size_req_df.round(5)
            # size_df.round(5)
            np.savetxt('/home/carl/New_Results/Files/flow_'+proto+'_size.txt', size_df["Size"], delimiter=',', fmt='%f')
            # np.savetxt('/home/carl/New_Results/Files/'+app_protocol+'_resp_size.txt', size_resp_df["Size"], delimiter=',', fmt='%f')
        else:
            # size_df = txt_df[txt_df['protocols'].str.contains(str(proto))]
            
            if type_Size == "mean_Trace":
                size = np.mean(size_df["Size"])
            if type_Size == "const_Value":
                size = 500
            
            
            arr_Size = np.empty(len(size_df["Size"])-1)
            arr_Size = [size for x in range(len(size_df["Size"])-1)]

            np.savetxt('/home/carl/New_Results/Files/flow_'+proto+'_size.txt', arr_Size, delimiter=',', fmt='%f')
        # timeStopSimulation =  time_resp_df["Time"].iloc[-1]
        # print(timeStopSimulation)
        # nRequestPackets = len(time_req_df["Time"])
        # nResponsePackets = len(time_resp_df["Time"])
    



def main(argv):
    # Determina se os tamanhos de pacotes serão constantes caso True ou se seguirão o padrão do trace caso False
    const_Size = True
    # Tipo de tamanho de pacote: "const_Value"(Valor específico) | "mean_Trace"(Usa a média do tamanho dos pacotes do trace)
    type_Size = "const_Value"

    read_filter(const_Size, type_Size)

    # traffic = "send"
    # parameter = "Size"

    #     # Condição de escolha do método de geração de variáveis aleatórias 
    #     # diretamente por uma distribuição de probabiidade
    #     if mt_RG == "PD":
    #         # Chamando a função wgwnet_PD() e retornando valor gerado para uma variável auxiliar
    #         aux_packet = wgwnet_PD(parameter, traffic)
    # # Condição de escolha do método de geração de variáveis aleatórias 
    #     # baseado nos dados do trace
    #     if first_trace_size == 0 and (mt_RG == "ecdf" or mt_RG == "tcdf"):
            
    #         if first_trace_size == 0:
    #             # Definindo o método de leitura do arquivo trace
    #             if reader == "txt":
    #                 read_txt(parameter, traffic, app_protocol)

if __name__ == '__main__':
    main(sys.argv)