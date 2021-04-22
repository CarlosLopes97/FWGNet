#!/usr/bin/env bash

rd="pcap"
# rd="xml"
app_protocol=$1

# dir_tr_file="Results/Traces/"${app_protocol}"_eth-myapp-py.pcap-0-0.${rd}"
dir_tr_file="Results/Traces/"${app_protocol}"_wifi-myapp-py.pcap-0-0.${rd}"

if [[ $rd = "pcap" ]]
then
    # Lendo arquivo .pcap
    cd "../../../"
    
    # sudo cp ${dir_tr_file} repos/ns-3-allinone/ns-3.30/
    # Tornando arquivo editável e legível
    sudo chmod 777 ${dir_tr_file}
    if [[ $app_protocol = "tcp" ]]
    then
        # Criando filtro por ts_VAL para somar os tamanhos dos pacotes no tcp
        # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t  -e tcp.options.timestamp.tsval > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_tsval.txt
        # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t  -e tcp.len > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size.txt
        
        # # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_tsval.txt
        # # Obtendo tempos de envio dos pacotes após a simulação
        # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e frame.time_relative > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time.txt
        sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e frame.time_relative -e tcp.len > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt
        sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt
        sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
        sudo sed -i ':a;N;$!ba;s/\t/";"/g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
        sudo sed -i '1s/^/"/' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
        sudo sed -i '${s/$/"/}' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"

    fi
    
    # if [[ $app_protocol = "http" ]]
    # then
    #     # Criando filtro por ts_VAL para somar os tamanhos dos pacotes no tcp
    #     # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e tcp.options.timestamp.tsval > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_tsval.txt
    #     # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e tcp.len > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size.txt
    #     # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size.txt
    #     # sudo sed -i ':a;N;$!ba;s/\n0//g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size.txt"
    #     # sudo sed -i ':a;N;$!ba;s/\n\n//g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size.txt"
    #     # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e frame.time_relative > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time.txt
    #     # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e tcp.dstport > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_port.txt
    #     # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time.txt
    #     # # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_tsval.txt
    #     # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_port.txt
    #     # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_proto.txt
    #     # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e frame.coloring_rule.name > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_proto.txt
        

    #     sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e frame.time_relative -e tcp.dstport -e tcp.len > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt
    #     # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size_req.txt

    #     sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt
    #     # Preparando arquivo para ser legível como DataFrame
    #     sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
    #     sudo sed -i ':a;N;$!ba;s/\t/";"/g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
    #     # Adicionando aspas no começo e no final do arquivo
    #     sudo sed -i '1s/^/"/' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
    #     sudo sed -i '${s/$/"/}' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
    # fi

    if [[ $app_protocol = "ftp" ]] || [[ $app_protocol = "hls" ]] || [[ $app_protocol = "http" ]] 
    then
        # Criando filtro por ts_VAL para somar os tamanhos dos pacotes no tcp
        # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e tcp.options.timestamp.tsval > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_tsval.txt
        # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e tcp.len > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size.txt
        # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size.txt
        # sudo sed -i ':a;N;$!ba;s/\n0//g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size.txt"
        # sudo sed -i ':a;N;$!ba;s/\n\n//g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size.txt"
        # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e frame.time_relative > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time.txt
        # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e tcp.dstport > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_port.txt

        # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e frame.coloring_rule.name > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_proto.txt
        

        sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e frame.time_relative -e tcp.dstport -e tcp.len > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt
        # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size_req.txt
        # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time.txt
        # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_tsval.txt
        # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_port.txt
        # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_proto.txt
        sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt
        # Preparando arquivo para ser legível como DataFrame
        sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
        sudo sed -i ':a;N;$!ba;s/\t/";"/g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
        # Adicionando aspas no começo e no final do arquivo
        sudo sed -i '1s/^/"/' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
        sudo sed -i '${s/$/"/}' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
        
        # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e tcp.srcport -e tcp.len > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size_send.txt
        # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e tcp.srcport -e frame.time_relative > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time_send.txt
        
        
        
        
        

        # sudo awk '/\t17\t/' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size_send.txt" > tempS && mv tempS "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size_send.txt"
        # sudo awk '/0\t/' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size_send.txt" > tempS && mv tempS "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size_send.txt"
        # sudo sed -i ':a;N;$!ba;s/0\t17\t//g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size_send.txt"
        # sudo sed -i '1\ts/^/"/' "repos/ns-3-allinone/ns-3.30/scratch/req_resp_${app_protocol}_size_send.txt"
        # sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size_send.txt"
        # sudo sed -i ':a;N;$!ba;s/\t/";"/g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size_send.txt"
        # sudo sed -i '1s/^/"/' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size_send.txt"
        # sudo sed -i '${s/$/"/}' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size_send.txt"
          
        
        
        # sudo awk '/\t17\t/' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time_send.txt" > tempS && mv tempS "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time_send.txt"
        # sudo awk '/0\t/' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time_send.txt" > tempS && mv tempS "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time_send.txt"
        # sudo sed -i ':a;N;$!ba;s/0\t17\t//g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time_send.txt"
        # sudo sed -i '1\ts/^/"/' "repos/ns-3-allinone/ns-3.30/scratch/req_resp_${app_protocol}_time_send.txt"
        # sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time_send.txt"
        # sudo sed -i ':a;N;$!ba;s/\t/";"/g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time_send.txt"
        # sudo sed -i '1s/^/"/' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time_send.txt"
        # sudo sed -i '${s/$/"/}' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time_send.txt"
        
        # Obtendo tempos de envio dos pacotes após a simulação
        
    fi

    if [[ $app_protocol = "udp" ]] || [[ $app_protocol = "802_11" ]] 
    then
        # Criando filtro por data.len para obter os tamanhos dos pacotes no udp
        # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t  -e data.len > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size.txt
        # Obtendo tempos de envio dos pacotes após a simulação
        sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e frame.time_relative -e data.len > repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt
        sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt
        sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
        sudo sed -i ':a;N;$!ba;s/\t/";"/g' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
        sudo sed -i '1s/^/"/' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
        sudo sed -i '${s/$/"/}' "repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}.txt"
    fi
    
    
    
    # Tornando arquivos editáveis e legíveis
    
    # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_size.txt
    # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/compare_${app_protocol}_time.txt

    # sed -i 's/,/./g' repos/ns-3-allinone/ns-3.30/scratch/size.txt
fi
echo "End of pos-simulation."