#!/usr/bin/env bash
# Diretório do arquivo do NS3

file="myapp-simple-ht-hidden-stations.py"
# file="myapp-fifth-req.py"

# file="myapp-fifth.py"
# file="tcp-bulk-send.py"
# file="fifth-random"
# file="lab2"

# Definindo tipo da linguagem do NS3 a ser utilizada
lgp="python"
# lgp="c++"

# Definindo o tipo de leitura do arquivo do trace
rd="pcap"
# rd="xml"

# Definindo se o NS3 irá usar o cmd para obter informações externas
cmd="true"
# cmd="false"

# Definindo o protocolo de comunicação a ser realizado
# app_protocol="tcp"
# app_protocol="udp"
# app_protocol="http"
# app_protocol="ftp"
# app_protocol="hls"
app_protocol="802_11"

# Definindo se o tamanho dos pacotes é constante ou não
size_const="True"
mt_const="Value"
# mt_const="Trace"
# validation="False"
validation="True"
run=1
# Declare an array of string with type
# declare -a mt_RG=("PD" "ecdf" "tcdf")
declare -a mt_RG=("ecdf")
# "99.80%";"99.85%";"99.90%";"99.95%";"99.99%"
IC="95.0"

# Definindo diretório do arquivo trace a ser utilizado pela simulação
if [[ $app_protocol = "tcp" ]]
then
    dir_tr_file="/media/carl/95d90125-d7fb-4efd-8875-ceead818ba80/Traces/87245236944caa854713d71b7939cce8.pcap"
fi
if [[ $app_protocol = "udp" ]]
then
    dir_tr_file="/media/carl/95d90125-d7fb-4efd-8875-ceead818ba80/Traces/Case-Study-2/UDP-files/udp-Client.${rd}"
fi
if [[ $app_protocol = "http" ]]
then
    # dir_tr_file="Traces/Case-Study-1/http-p2p.${rd}"
    dir_tr_file="/media/carl/95d90125-d7fb-4efd-8875-ceead818ba80/Traces/Case-Study-1/small-tcp-http.${rd}"
fi
if [[ $app_protocol = "ftp" ]]
then
    dir_tr_file="/media/carl/95d90125-d7fb-4efd-8875-ceead818ba80/Traces/Case-Study-2/TCP-files/ftp-tcp-b0.${rd}"
fi
if [[ $app_protocol = "hls" ]]
then
    dir_tr_file="/media/carl/95d90125-d7fb-4efd-8875-ceead818ba80/Traces/sf19eu-ikeriri-traces/hls.pcap"
fi

if [[ $app_protocol = "802_11" ]]
then
    dir_tr_file="/media/carl/95d90125-d7fb-4efd-8875-ceead818ba80/Traces/Vatican1/probes-2013-02-24.${rd}0"
fi


# 
#



# Condição que define pela linguagem do NS3 qual arquivo será copiado para área de trabalho do NS3
if [[ $lgp = "c++" ]]
then
    echo File ${file}.cc
    
    # Copy file to scratch
    sudo cp labs-example/${file}.cc ../repos/ns-3-allinone/ns-3.30/scratch/

    # Set permissions to file in scratch
    sudo chmod 777 ../repos/ns-3-allinone/ns-3.30/scratch/${file}.cc
    
fi
if [[ $lgp = "python" ]]
then
    echo File ${file}

    # Copy file to scratch
    sudo cp python-examples/${file} ../repos/ns-3-allinone/ns-3.30/scratch/

    # Set permissions to file in scratch
    sudo chmod 777 ../repos/ns-3-allinone/ns-3.30/scratch/${file}
fi

# Condição que copia o arquivo de trace para a área de trabalho do NS3 e define as configurações de filtros
if [[ $rd = "pcap" ]]
then
    # Lendo arquivo .pcap 
    cd "../"
    
    sudo cp ${dir_tr_file} repos/ns-3-allinone/ns-3.30/
    # Tornando arquivo editável e legível
    sudo chmod 777 ${dir_tr_file}
    
    # Realização do filtro para o protocolo FTP
    if [[ $app_protocol = "ftp" ]]
    then
        # Filtro para Requisições e Respostas
        # Filtro para envio de dados
        # Obtendo Tempo de envio dos pacotes
        sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e ftp.request -e  tls.record.content_type  -e frame.time_relative -e tcp.len > repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt
        # Habilitando permissões ao arquivo gerado
        sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt
        # sudo cat repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_aux_time.txt | sed -i -e '/\t22\t/p' -e '0,/\t22\t/d' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_time.txt"
        # Removendo requisições e respostas
        # sudo sed -i ':a N;$!ba; s/\(.*0\t\).*/\1/' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_time.txt"
        # sudo sed -i 's/^.*\t//' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_time.txt"
        # # Deletando a ultima linha do arquivo
        # sudo sed -i '$d' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_time.txt"
        # Preparando arquivo para ser legível como DataFrame
        sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        sudo sed -i ':a;N;$!ba;s/\t/";"/g' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        # Adicionando aspas no começo e no final do arquivo
        sudo sed -i '1s/^/"/' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        sudo sed -i '${s/$/"/}' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        
        
    fi
# Realização do filtro para o protocolo HTTP
    if [[ $app_protocol = "http" ]]
    then
        # "GET / HTTP/1.1\r\n" para Requisições
        # "HTTP/1.1 200 OK\r\n" para Envio de dados
        
        # Atribuindo valor do tempo de envio cada pacote em um arquivo .txt
        # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e frame.time_relative > repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_time.txt
        # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_time.txt
        
        # Atribuindo valor do atributo "ws.expert.message" que define os tipos de requisições (Request ou Response) em um arquivo .txt 
        sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e _ws.expert.message -e frame.time_relative -e tcp.len  > repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt
        # Filtrando as linhas que são de request e response
        # sudo awk '/OK/||/GET/' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt" > tempT && mv tempT "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        # Habilitando permissões ao arquivo gerado
        sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt
        sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        sudo sed -i ':a;N;$!ba;s/\t/";"/g' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        # Adicionando aspas no começo e no final do arquivo
        sudo sed -i '1s/^/"/' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        sudo sed -i '${s/$/"/}' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
    fi
# Realização do filtro para o protocolo HTTP
    if [[ $app_protocol = "tcp" ]]
    then
        # Atribuindo valor do tempo de envio cada pacote em um arquivo .txt
        sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e frame.time_relative -e tcp.len > repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt
        # Habilitando permissões ao arquivo gerado
        sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt
    fi
# Realização do filtro para o protocolo HTTP
    if [[ $app_protocol = "udp" ]]
    then
        # Atribuindo valor do tempo de envio cada pacote em um arquivo .txt
        sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e frame.time_relative -e udp.length > repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt
        # Habilitando permissões ao arquivos gerados
        sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt
        # Preparando arquivo para ser legível como DataFrame
        sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        sudo sed -i ':a;N;$!ba;s/\t/";"/g' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        # Adicionando aspas no começo e no final do arquivo
        sudo sed -i '1s/^/"/' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        sudo sed -i '${s/$/"/}' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
    fi
    
    # Realização do filtro para o protocolo HLS
    if [[ $app_protocol = "hls" ]]
    then
        # Filtro para Requisições e Respostas
        # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e frame.time_relative -e _ws.expert > repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_time.txt
        # sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_time.txt
        # sudo sed -i '$ s/.$//' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_time.txt"
        # Atribuindo valor do atributo "ws.expert.message" que define os tipos de requisições (Request ou Response) em um arquivo .txt 
        sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e _ws.expert -e tcp.flags.push -e frame.time_relative -e tcp.len > repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt
        sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt

        # Filtrando as linhas que são de request e response
        sudo sed -i ':a;N;$!ba;s/\n\t0/\nTCP\t0/g' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        sudo awk '/OK/||/GET/||/Partial/||/TCP/' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt" > tempT && mv tempT "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        
        # Preparando arquivo para ser legível como DataFrame
        sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        sudo sed -i ':a;N;$!ba;s/\t/";"/g' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        # Adicionando aspas no começo e no final do arquivo
        sudo sed -i '1s/^/"/' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        sudo sed -i '${s/$/"/}' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
    fi
    
    if [[ $app_protocol = "802_11" ]]
    then
        # Atribuindo valor do tempo de envio cada pacote em um arquivo .txt
        # sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e wlan.ta -e frame.time_relative -e frame.cap_len > repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt
        sudo termshark -r ${dir_tr_file} -T fields -E separator=/t -e frame.time_relative -e frame.cap_len > repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt
        # Habilitando permissões ao arquivos gerados
        sudo chmod 777 repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt
        # Preparando arquivo para ser legível como DataFrame
        sudo sed -i ':a;N;$!ba;s/\n/"\n"/g' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        sudo sed -i ':a;N;$!ba;s/\t/";"/g' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        # Adicionando aspas no começo e no final do arquivo
        sudo sed -i '1s/^/"/' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
        sudo sed -i '${s/$/"/}' "repos/ns-3-allinone/ns-3.30/scratch/${app_protocol}_trace.txt"
    fi
fi

# Condição para criar entradas para o NS3 por cmd com trace .pcap
if [[ $cmd = "true" ]] && [[ $rd = "pcap" ]]
then
    # Direcionando a função para a área de trabalho do NS3
    cd "repos/ns-3-allinone/ns-3.30/" 
    # Definindo Tamanho do arquivo
    # sizeFile=$(wc -l < scratch/${app_protocol}_time.txt)
    sizeFile=100
    # # Definindo último tempo do arquivo
    # lastTime=$(tail -n 1 scratch/${app_protocol}_time.txt)
    lastTime=100
    
    # Condição de criação das entradas pro protocolo HTTP
    if [[ $app_protocol = "http" ]]
    then
        # Definindo tamanho do arquivo de requisições
        # request_sizeFile=$(grep -c 'GET' scratch/${app_protocol}_trace.txt)
        request_sizeFile=0
        # Definindo tamanho do arquivo de respostas
        # response_sizeFile=$(grep -c 'OK' scratch/${app_protocol}_trace.txt)
        response_sizeFile=0
        # Escrevendo saídas no terminal
        # echo "request Size: " $request_sizeFile
        # echo "response Size: " $response_sizeFile
    fi
    # Condição de criação das entradas pro protocolo FTP
    if [[ $app_protocol = "ftp" ]]
    then
        # Definindo tamanho do arquivo de requisições
        # request_sizeFile=$(grep -c '1' scratch/req_resp_${app_protocol}_time.txt)
        request_sizeFile=0
        # Definindo tamanho do arquivo de respostas
        # response_sizeFile=$(grep -c '0' scratch/req_resp_${app_protocol}_time.txt)
        response_sizeFile=0
        # Escrevendo saídas no terminal
        # echo "request Size: " $request_sizeFile
        # echo "response Size: " $response_sizeFile
        # sizeFile=$(( sizeFile-(request_sizeFile+response_sizeFile) ))
        sizeFile=0
    fi
    if [[ $app_protocol = "hls" ]]
    then
        # Definindo tamanho do arquivo de requisições
        # request_sizeFile=$(grep -c 'GET' scratch/${app_protocol}_trace.txt)
        request_sizeFile=0
        # Definindo tamanho do arquivo de respostas
        # response_sizeFile=$(grep -c 'Partial' scratch/${app_protocol}_trace.txt)
        response_sizeFile=0
        # Escrevendo saídas no terminal
        # echo "request Size: " $request_sizeFile
        # echo "response Size: " $response_sizeFile
        # sizeFile=$(( sizeFile-(request_sizeFile+response_sizeFile) ))
    fi
    # Escrevendo saídas no terminal
    # echo Size File $sizeFile
    # echo Last Time $lastTime
# Executando NS3 com entradas pela linguagem C++
    if [[ $lgp = "c++" ]] 
    then
        ./waf --run "scratch/${file} --nPackets=$sizeFile --timeStopSimulation=$lastTime --app_protocol=$app_protocol"
    fi
    # Executando NS3 com entradas pela linguagem Python
    if [[ $lgp = "python" ]]
    then
        # Execução do NS3 para o protocolo HTTP
        if [[ $app_protocol = "http" ]] 
        then
            if [[ $validation = "True" ]] 
            then
                ./waf --pyrun "scratch/${file}  --nRequestPackets=$request_sizeFile --IC=$IC --const_Size=$size_const --mt_const=$mt_const --nResponsePackets=$response_sizeFile --timeStopSimulation=$lastTime --app_protocol=$app_protocol --validation=$validation --run=$run --mt_RG=$mt_RG"
            fi

            if [[ $validation = "False" ]] 
            then
                # Iterate the string array using for loop
                for val_mt_RG in ${mt_RG[@]};
                do
                    # for i_run in $run
                    for ((i_run=1; i_run<=run; i_run++));
                    do  
                        request_sizeFile=5000
                        response_sizeFile=5000
                        lastTime=3600
                        ./waf --pyrun "scratch/${file}  --nRequestPackets=$request_sizeFile --IC=$IC --const_Size=$size_const --mt_const=$mt_const --nResponsePackets=$response_sizeFile --timeStopSimulation=$lastTime --app_protocol=$app_protocol --validation=$validation --run=$i_run --mt_RG=$val_mt_RG"
                    done
                done
            fi
        fi
        # Execução do NS3 para o protocolo FTP e HLS
        if [[ $app_protocol = "ftp" ]] || [[ $app_protocol = "hls" ]]
        then
            # echo $lastTime
            ./waf --pyrun "scratch/${file} --nPackets=$sizeFile --nRequestPackets=$request_sizeFile --IC=$IC --timeStopSimulation=$lastTime --const_Size=$size_const --mt_const=$mt_const --nResponsePackets=$response_sizeFile --app_protocol=$app_protocol --validation=$validation --run=$run --mt_RG=$mt_RG"
        fi

        # Execução do NS3 para o protocolo UDP ou TCP
        if [[ $app_protocol = "udp" ]] || [[ $app_protocol = "tcp" ]] || [[ $app_protocol = "802_11" ]]
        then
            # Definindo tamanho do arquivo
            # sizeFile=$(wc -l < scratch/${app_protocol}_time.txt)
            # # Definindo último tempo do arquivo
            # lastTime=$(tail -n 1 scratch/${app_protocol}_time.txt)
            ./waf --pyrun "scratch/${file} --nPackets=$sizeFile --timeStopSimulation=$lastTime --IC=$IC --const_Size=$size_const --mt_const=$mt_const --app_protocol=$app_protocol --validation=$validation --run=$run --mt_RG=$mt_RG"
        fi
    fi
fi


# Condição para executar o NS3 sem entradas
if [[ $cmd = "false" ]]
then
    if [[ $lgp = "c++" ]] 
    then
        # cd desired/directory
        cd "../repos/ns-3-allinone/ns-3.30/" 
        ./waf --run "scratch/${file}"
    fi
    if [[ $lgp = "python" ]] 
    then
        # cd desired/directory
        cd "repos/ns-3-allinone/ns-3.30/" 
        ./waf --pyrun "scratch/${file}"
    fi
fi
# Escrevendo no terminal a finalização da simulação
echo "End of Simulation"

