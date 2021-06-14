#!/usr/bin/env bash

# Filter .pcap files 
# Definindo diretório para salvar arquivos filtrados
dir_tr_filter="/home/carl/New_Results/Filter_Traces/"
# Filtrando arquivo .pcap de uma rede real
if [[ $filter_pcap = "real" ]]
then
    # Start Filter config
    sudo termshark -r ${dir_pcap_file} -T fields -E separator=/t -e ip.src -e ip.dst -e frame.time_relative -e frame.len -e frame.protocols -e tcp.len -e udp.length > ${dir_tr_filter}${case_Study}_trace.txt
    # Habilitando permissões ao arquivo gerado
    sudo chmod 777 ${dir_tr_filter}${case_Study}_trace.txt
    # Preparando arquivo para ser legível como DataFrame
    sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "${dir_tr_filter}${case_Study}_trace.txt"
    sudo sed -i ':a;N;$!ba;s/\t/";"/g' "${dir_tr_filter}${case_Study}_trace.txt"
    # Adicionando aspas no começo e no final do arquivo
    sudo sed -i '1s/^/"/' "${dir_tr_filter}${case_Study}_trace.txt"
    sudo sed -i '${s/$/"/}' "${dir_tr_filter}${case_Study}_trace.txt"
fi
# Filtrando arquivo .pcap de uma rede simulada
if [[ $filter_pcap = "ns3" ]]
then
    # Start Filter config
    sudo termshark -r ${dir_pcap_file} -T fields -E separator=/t -e ip.src -e ip.dst -e frame.time_relative -e frame.len -e frame.protocols -e tcp.len -e udp.length > ${dir_tr_filter}ns3_${case_Study}_trace.txt
    # Habilitando permissões ao arquivo gerado
    sudo chmod 777 ${dir_tr_filter}${case_Study}_trace.txt
    # Preparando arquivo para ser legível como DataFrame
    sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "${dir_tr_filter}${case_Study}_ns3_trace.txt"
    sudo sed -i ':a;N;$!ba;s/\t/";"/g' "${dir_tr_filter}${case_Study}_ns3_trace.txt"
    # Adicionando aspas no começo e no final do arquivo
    sudo sed -i '1s/^/"/' "${dir_tr_filter}${case_Study}_ns3_trace.txt"
    sudo sed -i '${s/$/"/}' "${dir_tr_filter}${case_Study}_ns3_trace.txt"
fi
