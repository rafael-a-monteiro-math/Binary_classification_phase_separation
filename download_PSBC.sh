#!/bin/bash
## AUTHOR : RAFAEL MONTEIRO
## Untar all the files in a folder
author="RAFAEL MONTEIRO"
paper="Binary classification as a phase separation process"

##### MD5
##PSBC_Notebooks.tar.gz
md5_PSBC_Notebooks="59b88cd156308367b5c744c1e97a569e"

##PSBC_1D.tar.gz,  	405.0 MB	
md5_PSBC_1D="492d121bd9c9295577e6424fefc3c1a8" 

##PSBC_computational_statistics.tar.gz,  	345.3 MB
md5_PSBC_computational_statistics="e0beda0b9c1abdbe393c7e920cfce240"

##PSBC_Examples.tar.gz, 42.1 MB	
md5_PSBC_Examples="958bad286bb9914456c6d3ea55189bcf" 

##PSBC_Extras.tar.gz, 421.3 MB	
md5_PSBC_Extras="d745a43e73887c3d64f49f398c8ff0ae" 	

##PSBC_MNIST_Neumann_w1s.tar.gz, 925.9 MB
md5_PSBC_MNIST_Neumann_w1s="c863865dc815264fea6eba4dda90155e" 	

##PSBC_MNIST_Neumann_wNts.tar.gz, 1.2 GB
md5_PSBC_MNIST_Neumann_wNts="861c0149dffb85aa08292ec690269c98" 	

##PSBC_MNIST_Non_diff.tar.gz, 	823.1 MB
md5_PSBC_MNIST_Non_diff="f68c2f0db75ac18b1a49c324cf899fa5" 

##PSBC_MNIST_Periodic_w1s.tar.gz, 926.6 MB	
md5_PSBC_MNIST_Periodic_w1s="25678fa3d282aed7711117fe74403ceb" 	

##PSBC_MNIST_Periodic_wNts.tar.gz, 1.2 GB
md5_PSBC_MNIST_Periodic_wNts="6e3cd2d0f7fc1138e346052d79bcdf52" 

########################################################################
echo -e "This is a companion script to the paper \n\n\t$paper, \n\n\tby $author"

echo -e "\n\nThis script download all the files in the statistics folder\
and organizes them."

number_questions=(1 2 3 4)

questions="\n\nYou have the following options:\n \
\n Enter 1 to download all examples;\
\n Enter 2 to download all statistics;\
\n Enter 3 to download all jupyter notebooks;\
\n Enter 4 to download all raw data;
\n Enter any other value to finish."
echo -e $questions
read var
########################################################################
while [[ "${number_questions[@]}" =~ "$var" ]]
do    
    case "$var" in
    ########################################################################
    1 ) 
        echo -e "\nYou typed $var\n"
        echo -e "\n Downloading the Examples folder \(42.1 MB\)\n"
        wget -O PSBC_Examples.tar.gz -c https://zenodo.org/record/4005131/files/PSBC_Examples.tar.gz?download=1
        echo -e "\n Checking the md5sum"
        #getting the output of md5sum
        md5_now=$(md5 PSBC_Examples.tar.gz) 
        # splitting the output
        md5_now=(${md5_now//=/ })    
        ## getting just the number
        md5_now=${md5_now[2]}
        # comparing it to the number in the website
        if [ $md5_now == $md5_PSBC_Examples ]
        then
            echo -e "\n md5sum check complete: the file looks good!"
            echo -e "\nDecompressing tar ball"
            tar -xzvf PSBC_Examples.tar.gz
        else
            echo -e "\n md5sum check complete: the file is corrupted"
        fi
        echo -e "$questions"
        read var 
        ;;
    ########################################################################
    2 ) 
        echo -e "\nYou typed $var"
        echo -e "\n Downloading computational statitics \(345.3 MB\)\n"
        wget -O PSBC_computational_statistics.tar.gz -c https://zenodo.org/record/4005131/files/PSBC_computational_statistics.tar.gz?download=1
        echo -e "\n Checking the md5sum"
        #getting the output of md5sum
        md5_now=$(md5 PSBC_computational_statistics.tar.gz) 
        # splitting the output
        md5_now=(${md5_now//=/ })    
        ## getting just the number
        md5_now=${md5_now[2]}
        # comparing it to the number in the website
        if [ $md5_now == $md5_PSBC_computational_statistics ]
        then
            echo -e "\n md5sum check complete: the file looks good!"
            echo -e "\nDecompressing tar ball"
            tar -xzvf PSBC_computational_statistics.tar.gz
        else
            echo -e "\n md5sum check complete: the file is corrupted"
        fi
        echo -e "$questions"
        read var 
        ;;
    ########################################################################
    3 ) 
        echo -e "you typed $var"
        echo -e "\n Downloading jupyter-notebooks\n"
        wget -O PSBC_Notebooks.tar.gz -c https://github.com/rafael-a-monteiro-math/Binary_classification_phase_separation/blob/master/PSBC_Notebooks.tar.gz
        echo -e "\n Checking the md5sum"
        #getting the output of md5sum
        md5_now=$(md5 PSBC_Notebooks.tar.gz) 
        # splitting the output
        md5_now=(${md5_now//=/ })    
        ## getting just the number
        md5_now=${md5_now[2]}
        # comparing it to the number in the website
        if [ $md5_now == $md5_PSBC_Notebooks ]
        then
            echo -e "\n md5sum check complete: the file looks good!"
            echo -e "\nDecompressing tar ball"
            tar -xzvf PSBC_Notebooks.tar.gz
        else
            echo -e "\n md5sum check complete: the file is corrupted"
        fi
        
        echo -e "$questions"
        read var 
        ;;
    ########################################################################
    4 ) 
        echo -e "you typed $var"
        echo -e"\n Creating PSBC folder"
        mkdir PSBC
        
        ## DOWNLOADING DATA
        echo -e "\n Downloading all the trained data \(>5GB\)\n"
        
        echo -e "\nPSBC_Extras.tar.gz \(421.3 MB\)	\n"
        wget -O PSBC_Extras.tar.gz -c https://zenodo.org/record/4005131/files/PSBC_Extras.tar.gz?download=1
        
        echo -e "\nPSBC_MNIST_Neumann_w1s.tar.gz \(925.9 MB\)\n"
        wget -O PSBC/PSBC_MNIST_Neumann_w1s.tar.gz -c https://zenodo.org/record/4005131/files/PSBC_MNIST_Neumann_w1s.tar.gz?download=1
        
        echo -e "\nPSBC_MNIST_Neumann_wNts.tar.gz \(1.2 GB\)\n"
        wget -O PSBC/PSBC_MNIST_Neumann_wNts.tar.gz -c https://zenodo.org/record/4012004/files/PSBC_MNIST_Neumann_wNts.tar.gz?download=1
        
        echo -e "\nPSBC_MNIST_Non_diff.tar.gz \(823.1 MB\)\n"
        wget -O PSBC/PSBC_MNIST_Non_diff.tar.gz -c https://zenodo.org/record/4012004/files/PSBC_MNIST_Non_diff.tar.gz?download=1
        
        echo -e "\nPSBC_MNIST_Periodic_w1s.tar.gz \(926.6 MB\)\n"
        wget -O PSBC/PSBC_MNIST_Periodic_w1s.tar.gz -c https://zenodo.org/record/4012004/files/PSBC_MNIST_Periodic_w1s.tar.gz?download=1
        
        echo -e "\nPSBC_MNIST_Periodic_wNts.tar.gz \(1.2 GB\)\n"
        wget -O PSBC/PSBC_MNIST_Periodic_wNts.tar.gz -c https://zenodo.org/record/4012004/files/PSBC_MNIST_Periodic_wNts.tar.gz?download=1
        
        ### CHECKING MD5SUM

        echo -e "\n Checking the md5sum"
        #######################################
        echo -e "\n FILE: PSBC_Extras.tar.gz"
        md5_now=$(md5 PSBC_Extras.tar.gz) 
        # splitting the output
        md5_now=(${md5_now//=/ })    
        ## getting just the number
        md5_now=${md5_now[2]}
        # comparing it to the number in the website
        if [ $md5_now == $md5_PSBC_Extras ]
        then
            echo -e "\n md5sum check complete: the file PSBC_Extras.tar.gz looks good!"
            echo -e "\nDecompressing tar ball"
            tar -xzvf PSBC_Extras.tar.gz
        else
            echo -e "\n md5sum check complete: the file md5_PSBC_Extras.tar.gz is corrupted"
        fi
        #######################################
        ### Move to PSBC folder
        #######################################
        cd PSBC
        #######################################
        echo -e "\n FILE: PSBC_MNIST_Neumann_w1s.tar.gz"
        md5_now=$(md5 PSBC_MNIST_Neumann_w1s.tar.gz) 
        # splitting the output
        md5_now=(${md5_now//=/ })    
        ## getting just the number
        md5_now=${md5_now[2]}
        # comparing it to the number in the website
        if [ $md5_now == $md5_PSBC_MNIST_Neumann_w1s ]
        then
            echo -e "\n md5sum check complete: the file PSBC_MNIST_Neumann_w1s.tar.gz looks good!"
            echo -e "\nDecompressing tar ball"
            tar -xzvf PSBC_MNIST_Neumann_w1s.tar.gz
        else
            echo -e "\n md5sum check complete: the file PSBC_MNIST_Neumann_w1s.tar.gz is corrupted"
        fi
        #######################################
        echo -e "\n FILE: PSBC_MNIST_Neumann_wNts.tar.gz"
        md5_now=$(md5 PSBC_MNIST_Neumann_wNts.tar.gz) 
        # splitting the output
        md5_now=(${md5_now//=/ })    
        ## getting just the number
        md5_now=${md5_now[2]}
        # comparing it to the number in the website
        if [ $md5_now == $md5_PSBC_MNIST_Neumann_wNts ]
        then
            echo -e "\n md5sum check complete: the file PSBC_MNIST_Neumann_wNts.tar.gz looks good!"
            echo -e "\nDecompressing tar ball"
            tar -xzvf PSBC_MNIST_Neumann_wNts.tar.gz
        else
            echo -e "\n md5sum check complete: the file PSBC_MNIST_Neumann_wNts.tar.gz is corrupted"
        fi
        #######################################
        echo -e "\n FILE: PSBC_MNIST_Non_diff.tar.gz"
        md5_now=$(md5 PSBC_MNIST_Non_diff.tar.gz) 
        # splitting the output
        md5_now=(${md5_now//=/ })    
        ## getting just the number
        md5_now=${md5_now[2]}
        # comparing it to the number in the website
        if [ $md5_now == $md5_PSBC_MNIST_Non_diff ]
        then
            echo -e "\n md5sum check complete: the file PSBC_MNIST_Non_diff.tar.gz looks good!"
            echo -e "\nDecompressing tar ball"
            tar -xzvf PSBC_MNIST_Non_diff.tar.gz
        else
            echo -e "\n md5sum check complete: the file PSBC_MNIST_Non_diff.tar.gz is corrupted"
        fi
        #######################################
        echo -e "\n FILE: PSBC_MNIST_Periodic_w1s.tar.gz"
        md5_now=$(md5 PSBC_MNIST_Periodic_w1s.tar.gz) 
        # splitting the output
        md5_now=(${md5_now//=/ })    
        ## getting just the number
        md5_now=${md5_now[2]}
        # comparing it to the number in the website
        if [ $md5_now == $md5_PSBC_MNIST_Periodic_w1s ]
        then
            echo -e "\n md5sum check complete: the file PSBC_MNIST_Periodic_w1s.tar.gz looks good!"
            echo -e "\nDecompressing tar ball"
            tar -xzvf PSBC_MNIST_Periodic_w1s.tar.gz
        else
            echo -e "\n md5sum check complete: the file PSBC_MNIST_Periodic_w1s.tar.gz is corrupted"
        fi
        #######################################
        echo -e "\n FILE: PSBC_MNIST_Periodic_wNts.tar.gz"
        md5_now=$(md5 PSBC_MNIST_Periodic_wNts.tar.gz) 
        # splitting the output
        md5_now=(${md5_now//=/ })    
        ## getting just the number
        md5_now=${md5_now[2]}
        # comparing it to the number in the website
        if [ $md5_now == $md5_PSBC_MNIST_Periodic_wNts ]
        then
            echo -e "\n md5sum check complete: the file PSBC_MNIST_Periodic_wNts.tar.gz looks good!"
            echo -e "\nDecompressing tar ball"
            tar -xzvf PSBC_MNIST_Periodic_wNts.tar.gz
        else
            echo -e "\n md5sum check complete: the file PSBC_MNIST_Periodic_wNts.tar.gz is corrupted"
        fi
        #######################################
        echo -e "$questions"
        read var 
        ;;
    ########################################################################
    esac
done

echo -e "\nEnd of file!"