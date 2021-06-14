#!/usr/bin/env bash

# Diretório principal da pasta
src="/home/carl/FWGNet/new_FWGNet/"
# Definição se o tamanho dos pacotes será constante
const_Size="True"
# Tipo de tamanho de pacote: "const_Value"(Valor específico) | "mean_Trace"(Usa a média do tamanho dos pacotes do trace)
type_Size="const_Value"
# Definição para salvar gráficos 
save_Graph="True"
# Definição do estudo de caso (experimento) abordado "p2p" | "csma" | "wifi" | "p2p-csma" | "csma-wifi" | "csma-p2p-wifi"
case_Study="csma"
# Definição do tipo de rede "p2p" | "multi"
# type_Net="multi" 
# Definição do tipo de topologia "duo" | "multi"
type_Topol="mono"
# Definição do Intervalo de Confiança
# "99.80%";"99.85%";"99.90%";"99.95%";"99.99%"
IC="95.0"
# Definição para obter ou não uma captura
get_pcap="False"
# Definição da interface a ser analisada
interface="wlp61s0" 
# Definição da ativação do verbose
str_verbose="True"
# Definição da taxa da comunicação
rate="4096kb/s"
if [[ $case_Study = "csma" ]]
then
    rate="4096Kbps"
fi
# Definição do atraso do canal
lat="0.2ms"

# Captura do tráfego (If get_pcap == "True")
if [[ $get_pcap = "True" ]]
then
    # Exportando interface
    export interface
    # Start capture traffic module
    # ./Shell/capture.sh
    echo "End Capture Module"
fi
# Abertura de arquivo .pcap existente get_pcap == "False" get a exist file
if [[ $get_pcap = "False" ]]
then
    # .pcap file directory
    # dir_pcap_file="/media/carl/95d90125-d7fb-4efd-8875-ceead818ba80/Traces/Case-Study-1/small-tcp-http.pcap"
    dir_pcap_file="/media/carl/95d90125-d7fb-4efd-8875-ceead818ba80/Traces/02d239902051fae4acd5e7bd7bcb5494.pcap"
    # Definição do tipo de tráfego observado
    filter_pcap="real"
    # Exportando variáveis
    export dir_pcap_file
    export filter_pcap
    export case_Study
    # Start filter module (real .pcap file)
    # ./Shell/filter.sh
    echo "End Real Filter Module"
fi


# Start workload generation module
# python Python/workload_generation.py "$const_Size" "$type_Size" "$save_Graph" "$case_Study" "$IC" "$type_Topol"
echo "End Workload Generation Module"


# Variável para obter o número de parâmetros, colunas, do arquivo "proto_ips.txt"
str_n_param=$(awk -F' ' '{print NF; exit}' < /home/carl/New_Results/Files/${case_Study}_proto_ips.txt)
# Variável que obtem o número de linhas do arquivo "proto_ips.txt"
str_n_row=$(wc -l < /home/carl/New_Results/Files/${case_Study}_proto_ips.txt)
# Variável para obter o número de parametros, colunas, dos arquivos filtrados do trace
str_n_file_param="2"

# Start NS3 module 
# Definindo diretório do NS3
dir_NS3="/home/carl/repos/ns-3-allinone/ns-3.31/"
# Definindo nome do arquivo de simulação (sem ".cc")
file_Simulation="multi-eth"
# Definindo nome do arquivo myapp
file_Myapp="myapp.h"
# Definindo diretório original do arquivo de simulação
file_Dir="/home/carl/FWGNet/new_FWGNet/C++/"


# Copy file to scratch
sudo cp ${file_Dir}${file_Simulation}.cc ${dir_NS3}scratch
# Set permissions to file in scratch
sudo chmod 777 ${dir_NS3}scratch/${file_Simulation}.cc

# Copy file to scratch
sudo cp ${file_Dir}${file_Myapp} ${dir_NS3}
# Set permissions to file in scratch
sudo chmod 777 ${dir_NS3}${file_Myapp}

# Open NS3 directory
cd ${dir_NS3} 

# Run simulation file 
sudo ./waf --run "scratch/${file_Simulation} --str_n_param=${str_n_param} --str_n_row=${str_n_row} --str_n_file_param=${str_n_file_param} --str_verbose=${str_verbose} --rate=${rate} --lat=${lat} --case_Study=${case_Study}" # --phyMode=DsssRate2Mbps --rss=-50 --packetSize=500 --numPackets=100 --interval=2 "
# Debbuger #
# sudo ./waf --run scratch/${file_Simulation} --command-template="g++ %s"

# Removendo arquivos existentes da pasta do NS3
sudo rm ${dir_NS3}scratch/${file_Simulation}.cc
sudo rm ${dir_NS3}${file_Myapp}

echo "End Simulation Module"

# Open source directory
cd ${src}
# Start filter module (simulated .pcap file)
# .pcap file directory
dir_pcap_file="/home/carl/repos/ns-3-allinone/ns-3.31/"${case_Study}".pcap"
export dir_pcap_file
# Atualizand ovariável para filtro em arquivos do NS3
filter_pcap="ns3"
export filter_pcap
# Start filter module (real .pcap file)
# ./Shell/filter.sh
echo "End Simulated Filter Module"
# Start compare module
# python Python/compare.py "$const_Size" "$type_Size" "$save_Graph" "$case_Study" "$IC"
echo "End Compare Module"
echo "End Framework"