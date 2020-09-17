# Binary Classification as a Phase Separation Process
This is a Github with python code used in the paper 

<b><a href=https://arxiv.org/abs/2009.02467>Binary Classification as a Phase Separation Process</a></b>, by <b> <a href=https://sites.google.com/view/rafaelmonteiro-math/home>Rafael Monteiro </a></b>. A preprint to the above paper is in arXiv at <a href=https://arxiv.org/abs/2009.02467>https://arxiv.org/abs/2009.02467</a>.

In the <a href=https://rafael-a-monteiro-math.github.io/Binary_classification_phase_separation/index.html>webpage</a> you will find a short tutorial  with  examples of how to use/apply the model. The tutorial can be downloaded <a href=https://github.com/rafael-a-monteiro-math/Binary_classification_phase_separation/blob/master/Website.pdf>here</a>, or from the aforementioned website.

Further information can be found in the file <a href=https://github.com/rafael-a-monteiro-math/Binary_classification_phase_separation/blob/master/README.pdf>README.pdf</a>, which contains information on how to run the model and also on the trained models's parameters used in the paper.

### Available Data and verifiability:
The dataset used in the paper is public: it is known as the <a href=http://yann.lecun.com/exdb/mnist/> MNIST database </a>.

Raw data and Computational statistics  are available at 
<li><a href=https://zenodo.org/record/4005131#.X1nsuR9fhFQ>10.5281/zenodo.4005131</a></li>

I have also included the script <a href=https://github.com/rafael-a-monteiro-math/Binary_classification_phase_separation/blob/master/download_PSBC.sh>download_PSBC.sh</a> in this github that you can use to download all the data automatically: it gives you three options:
  1. download examples;
  2. download statistics;
  3. download all jupyter notebooks;
  4. download all raw data;

There is a tutorial on how to run the script in the file <a href=https://github.com/rafael-a-monteiro-math/Binary_classification_phase_separation/blob/master/README.pdf>README.pdf</a>.

### Remarks:
All PSBCs applied to the MNIST database have been trained on a super computer at <a href=https://unit.aist.go.jp/matham-oil/index_en.htm>MathAM-OIL</a>, using a variable number  of cores ranging from  5 to 22, over a single node. Smaller models have been trained on simple laptops (MacBook Pro with Intel Core i7 processor, 4 Cores, with speed 2.5 GHz).

As proved in the paper, it is possible to parallelize the non-diffusive PSBC: this is a very interesting case (both mathematically and computationally) that has not been implemented; the author welcomes those who want to contribute with that. Other questions and related open problems can be found in the paper <a href=https://arxiv.org/abs/2009.02467>Binary Classification as a Phase Separation Process</a>.


### Dependencies:
Main modules and versions are, respectively:

  1. python_version 3.7.0
  2. sklearn 0.22.2.post1
  3. matplotlib 3.2.1
  4. numpy 1.18.1
  5. scipy 1.3.0
  6. sympy 1.3
  7. pandas 1.0.1
  8. keras 2.3.1
  9. tensorflow 1.14.0
  10. shutil 1.14.0
  11. copy 1.14.0
  12. pickle 4.0
  

<br>
--------------------------------------------------
</br>
<ul>
<li>Rafael Monteiro</li>
<li>Mathematics for Advanced Materials - Open Innovation Laboratory, Japan</li>
<li>PhD Applied Mathematics</li>
<li>https://sites.google.com/view/rafaelmonteiro-math/home</li>
<li>email:<a href=rafael.a.monteiro.math@gmail.com>rafael.a.monteiro.math  AT gmail.com</a>, or <a href=monteirodasilva-rafael@aist.go.jp>monteirodasilva-rafael AT aist.go.jp</a></li>
</ul>
