#!/usr/bin/env bash

# Filter .pcap files 

# NS3 dir
dir_NS3="repos/ns-3-allinone/ns-3.30/"

# Por enquanto
app_protocol="http"

# Start Filter config
sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e ip.src -e ip.dst -e frame.time_relative -e frame.len -e frame.protocols > ${dir_NS3}scratch/${app_protocol}_trace.txt

# Habilitando permissões ao arquivo gerado
sudo chmod 777 ${dir_NS3}scratch/${app_protocol}_trace.txt
# Preparando arquivo para ser legível como DataFrame
sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "${dir_NS3}scratch/${app_protocol}_trace.txt"
sudo sed -i ':a;N;$!ba;s/\t/";"/g' "${dir_NS3}scratch/${app_protocol}_trace.txt"
# Adicionando aspas no começo e no final do arquivo
sudo sed -i '1s/^/"/' "${dir_NS3}scratch/${app_protocol}_trace.txt"
sudo sed -i '${s/$/"/}' "${dir_NS3}scratch/${app_protocol}_trace.txt"

