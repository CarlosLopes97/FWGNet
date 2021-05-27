#!/usr/bin/env bash
app_protocol="http"
rd="pcap"
if [[ $app_protocol = "http" ]]
then
    dir_tr_file="/media/carl/95d90125-d7fb-4efd-8875-ceead818ba80/Traces/Case-Study-1/small-tcp-http.${rd}"
    # dir_tr_file="/media/carl/95d90125-d7fb-4efd-8875-ceead818ba80/Traces/Case-Study-1/http-p2p.${rd}"
fi

sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e ip.src -e ip.dst -e frame.protocols -e _ws.expert.message -e frame.time_relative -e tcp.len  > ${app_protocol}_trace.txt
# Filtrando as linhas que são de request e response
# sudo awk '/OK/||/GET/' "${app_protocol}_trace.txt" > tempT && mv tempT "${app_protocol}_trace.txt"
# Habilitando permissões ao arquivo gerado
sudo chmod 777 ${app_protocol}_trace.txt
sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "${app_protocol}_trace.txt"
sudo sed -i ':a;N;$!ba;s/\t/";"/g' "${app_protocol}_trace.txt"
# Adicionando aspas no começo e no final do arquivo
sudo sed -i '1s/^/"/' "${app_protocol}_trace.txt"
sudo sed -i '${s/$/"/}' "${app_protocol}_trace.txt"

# tshark -r ${dir_tr_file} -qz io,phs
tshark -r ${dir_tr_file} -T fields -e frame.number -e frame.time_relative -e ip.src -e ip.dst -e frame.protocols -e frame.len -E header=y -E quote=n -E occurrence=f